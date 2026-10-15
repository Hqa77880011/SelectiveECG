import json
from pathlib import Path
import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.linear_model import LogisticRegression
from torch.nn import functional as F


def json_ready(value):
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and (not np.isfinite(value)):
        return None
    if isinstance(value, dict):
        return {str(k): json_ready(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [json_ready(v) for v in value]
    return value


def patient_weights(ids):
    _, inverse, counts = np.unique(ids, return_inverse=True, return_counts=True)
    return 1.0 / counts[inverse]


def check_groups(ids, splits):
    for value in np.unique(ids):
        if len(np.unique(splits[ids == value])) != 1:
            raise ValueError(f"Patient/group {value} occurs in multiple partitions")


def fit(model, train, validation, objective, cfg, folder, monitor=None):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg["learning_rate"], weight_decay=cfg["weight_decay"]
    )
    generator = torch.Generator().manual_seed(cfg["seed"])
    best, wait, history = (float("inf"), 0, [])
    if not len(train[0]) or not len(validation[0]):
        raise ValueError("Empty train/validation split")
    for epoch in range(cfg["epochs"]):
        model.train()
        for ix in torch.randperm(len(train[0]), generator=generator).split(
            cfg["batch_size"]
        ):
            numerator, denominator = objective(model, *[x[ix] for x in train])
            if denominator.item() == 0:
                continue
            optimizer.zero_grad()
            (numerator / denominator).backward()
            optimizer.step()
        model.eval()
        numerator_sum = denominator_sum = 0.0
        with torch.no_grad():
            for ix in torch.arange(len(validation[0])).split(cfg["batch_size"]):
                numerator, denominator = objective(model, *[x[ix] for x in validation])
                numerator_sum += numerator.item()
                denominator_sum += denominator.item()
        if not denominator_sum:
            raise ValueError("Validation has no supported target")
        validation_loss = numerator_sum / denominator_sum
        value = monitor(model, validation) if monitor is not None else validation_loss
        history.append(
            {
                "epoch": epoch + 1,
                "validation_loss": validation_loss,
                "selection_value": value,
            }
        )
        if value < best:
            best, wait = (value, 0)
            torch.save(
                {"state": model.state_dict(), "config": cfg}, folder / "model.pt"
            )
        else:
            wait += 1
        if wait >= cfg["patience"]:
            break
    (folder / "training.json").write_text(
        json.dumps(json_ready(history), indent=2, allow_nan=False)
    )


def probabilities(logits, temperature):
    return torch.sigmoid(
        torch.as_tensor(logits, dtype=torch.float64)
        / torch.as_tensor(temperature, dtype=torch.float64)
    ).numpy()


def temperature(logits, y, weights=None):
    logits = torch.tensor(logits, dtype=torch.float64)
    y = torch.tensor(y, dtype=torch.float64)
    weights = (
        torch.ones_like(y)
        if weights is None
        else torch.tensor(weights, dtype=torch.float64)
    )
    known = torch.isfinite(y) & (weights > 0)
    if not known.any():
        return 1.0
    logt = torch.zeros((), dtype=torch.float64, requires_grad=True)
    opt = torch.optim.LBFGS([logt], max_iter=60, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        loss = (
            F.binary_cross_entropy_with_logits(
                logits[known] / logt.exp().clamp(0.05, 20), y[known], reduction="none"
            )
            * weights[known]
        ).sum() / weights[known].sum()
        loss.backward()
        return loss

    opt.step(closure)
    return float(logt.detach().exp().clamp(0.05, 20))


def binary_metrics(y, p, weights=None):
    y, p = (np.asarray(y), np.asarray(p))
    w = np.ones(len(y)) if weights is None else np.asarray(weights)
    known = np.isfinite(y) & np.isfinite(p) & (w > 0)
    y, p, w = (y[known], p[known], w[known])
    if not len(y):
        return {"n": 0, "auroc": None, "auprc": None, "brier": None, "ece": None}
    result = {
        "n": len(y),
        "positives": int(y.sum()),
        "brier": float(np.average((p - y) ** 2, weights=w)),
    }
    result["auroc"] = (
        float(roc_auc_score(y, p, sample_weight=w)) if len(np.unique(y)) == 2 else None
    )
    result["auprc"] = (
        float(average_precision_score(y, p, sample_weight=w)) if y.sum() > 0 else None
    )
    ece = 0.0
    for b in range(10):
        mask = (p >= b / 10) & (p < (b + 1) / 10 if b < 9 else p <= 1)
        if mask.any():
            ece += (
                w[mask].sum()
                / w.sum()
                * abs(np.average(y[mask] - p[mask], weights=w[mask]))
            )
    result["ece"] = float(ece)
    if len(np.unique(y)) == 2:
        regression = LogisticRegression(C=1000000.0).fit(
            np.log(np.clip(p, 1e-07, 1 - 1e-07) / (1 - np.clip(p, 1e-07, 1 - 1e-07)))[
                :, None
            ],
            y,
            sample_weight=w,
        )
        result.update(
            calibration_intercept=float(regression.intercept_[0]),
            calibration_slope=float(regression.coef_[0, 0]),
        )
    return result


def ratio(num, den):
    return float(num / den) if den else None


def bootstrap(groups, function, draws, seed):
    rng = np.random.default_rng(seed)
    unique = np.unique(groups)
    values = []
    for _ in range(draws):
        chosen = rng.choice(unique, len(unique), replace=True)
        ix = np.concatenate([np.flatnonzero(groups == group) for group in chosen])
        value = function(ix)
        if value is not None and np.isfinite(value):
            values.append(value)
    return np.quantile(values, [0.025, 0.975]).tolist() if values else [None, None]
