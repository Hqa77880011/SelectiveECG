import argparse
import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy.signal import resample_poly
from torch import nn
from torch.nn import functional as F
from .common import (
    binary_metrics,
    check_groups,
    fit,
    patient_weights,
    probabilities,
    ratio,
    temperature,
)


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


QUERIES = ["AFIB", "CRBBB", "CLBBB", "1AVB"]
LEADS = ["I", "II", "III", "AVR", "AVL", "AVF", "V1", "V2", "V3", "V4", "V5", "V6"]


class Block(nn.Module):

    def __init__(self, ci, co, stride=1):
        super().__init__()
        self.convs = nn.Sequential(
            nn.Conv1d(ci, co, 3, stride=stride, padding=1, bias=False),
            nn.BatchNorm1d(co),
            nn.ReLU(),
            nn.Conv1d(co, co, 3, padding=1, bias=False),
            nn.BatchNorm1d(co),
        )
        self.skip = (
            nn.Identity()
            if ci == co and stride == 1
            else nn.Sequential(
                nn.Conv1d(ci, co, 1, stride=stride, bias=False), nn.BatchNorm1d(co)
            )
        )

    def forward(self, x):
        return F.relu(self.convs(x) + self.skip(x))


class ResNet(nn.Module):

    def __init__(self, observation_classes=0):
        super().__init__()
        layers = [
            nn.Conv1d(24, 64, 7, stride=2, padding=3, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(3, 2, 1),
        ]
        ci = 64
        for width in [64, 128, 256, 512]:
            layers.extend(
                [Block(ci, width, 1 if ci == width else 2), Block(width, width)]
            )
            ci = width
        self.encoder = nn.Sequential(*layers, nn.AdaptiveAvgPool1d(1))
        self.canonical = nn.Linear(512, 4)
        self.observation = (
            nn.Linear(512, observation_classes) if observation_classes else None
        )

    def forward(self, x):
        h = self.encoder(x).squeeze(-1)
        return (
            self.canonical(h),
            self.observation(h) if self.observation is not None else None,
        )


def prepare(args, cfg):
    import math
    import wfdb

    manifest = pd.read_csv(args.manifest, dtype={"patient": str, "source": str})
    check_groups(manifest.patient.to_numpy(), manifest.split.to_numpy())
    mappings = (
        json.loads(Path(args.mapping).read_text())
        if args.mapping
        else {code: [code] for code in QUERIES}
    )
    observation_codes = sorted(
        {
            f"{r.source}:{code}"
            for r in manifest.itertuples(index=False)
            for code in json.loads(r.codes)
        }
    )
    rows = []
    for row in manifest.itertuples(index=False):
        signal, fields = wfdb.rdsamp(row.record)
        sf = int(fields["fs"])
        names = [name.upper() for name in fields["sig_name"]]
        target = np.full((len(signal), 12), np.nan)
        for j, lead in enumerate(LEADS):
            if lead in names:
                col = names.index(lead)
                factor = {"MV": 1.0, "UV": 0.001, "V": 1000.0}.get(
                    fields["units"][col].upper()
                )
                if factor is None:
                    raise ValueError("Unknown physical ECG unit")
                target[:, j] = signal[:, col] * factor
        valid = np.isfinite(target)
        values = resample_poly(
            np.nan_to_num(target),
            100 // math.gcd(sf, 100),
            sf // math.gcd(sf, 100),
            axis=0,
        )
        masks = (
            resample_poly(
                valid.astype(float),
                100 // math.gcd(sf, 100),
                sf // math.gcd(sf, 100),
                axis=0,
            )
            > 0.99
        )
        codes = set(json.loads(row.codes))
        y = np.full(4, np.nan)
        for k, query in enumerate(QUERIES):
            if codes.intersection(mappings[query]):
                y[k] = 1
            elif "NORM" in codes and (
                not codes.intersection(
                    set().union(*[set(v) for v in mappings.values()])
                )
            ):
                y[k] = 0
        obs = np.full(len(observation_codes), np.nan)
        for i, code in enumerate(observation_codes):
            if code.startswith(row.source + ":"):
                if code.split(":", 1)[1] in codes:
                    obs[i] = 1
                elif str(getattr(row, "closed_world", False)).lower() in [
                    "true",
                    "1",
                    "1.0",
                ]:
                    obs[i] = 0
        eligibility = np.array(json.loads(row.eligibility), bool)
        if eligibility.shape != (4,):
            raise ValueError("Eligibility must contain four policy-provided bits")
        chunks = []
        invalid_leads = np.zeros(12, bool)
        for begin in range(0, len(values), 1000):
            x = np.zeros((1000, 12))
            mask = np.zeros((1000, 12), bool)
            end = min(len(values), begin + 1000)
            x[: end - begin] = values[begin:end]
            mask[: end - begin] = masks[begin:end]
            for lead in range(12):
                flat = any(
                    (np.ptp(x[s : s + 200, lead]) < 0.02 for s in range(0, 801, 100))
                )
                absent = np.mean(~mask[:, lead]) > 0.1
                rail = getattr(row, "rail_mv", np.nan)
                clipped = (
                    np.isfinite(rail) and np.mean(np.abs(x[:, lead]) >= rail) > 0.01
                )
                invalid_leads[lead] |= flat or absent or clipped
            chunks.append((x.T, mask.T))
        quality = np.sum(invalid_leads) < 3
        for window, (x, mask) in enumerate(chunks):
            rows.append(
                (
                    x,
                    mask,
                    y,
                    obs,
                    row.patient,
                    row.source,
                    row.split,
                    str(row.record),
                    eligibility,
                    quality,
                    window,
                )
            )
    target = Path(args.data)
    target.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        target,
        **{
            name: np.asarray([r[i] for r in rows])
            for i, name in enumerate(
                [
                    "x",
                    "mask",
                    "y",
                    "obs",
                    "patient",
                    "source",
                    "split",
                    "record",
                    "eligibility",
                    "quality",
                    "window",
                ]
            )
        },
        observation_codes=np.array(observation_codes),
    )


def decisions(p, eligible, quality, t, gamma, labels, weights, cfg):
    accept = eligible & quality[:, None] & (np.maximum(p, 1 - p) >= gamma)
    released = accept.all(1)
    d = p >= t
    known = np.isfinite(labels)
    y = np.nan_to_num(labels)
    w = weights[:, None] * known
    cost = (
        cfg["false_negative_cost"] * y * ~d + cfg["false_positive_cost"] * (1 - y) * d
    )
    workload = float(np.average(~released, weights=weights))
    risk = ratio(np.sum(w * released[:, None] * cost), w.sum())
    pos = w * y
    positive_acceptance = np.array(
        [
            (
                ratio(np.sum(pos[:, k] * released), pos[:, k].sum())
                if pos[:, k].sum()
                else np.nan
            )
            for k in range(4)
        ]
    )
    return (released, d, workload, risk, positive_acceptance, accept)


def select_policy(p, eligible, quality, y, w, cfg):
    t = np.full(4, 0.5)
    gamma = np.full(4, 0.5)

    def score(t, gamma):
        _, d, work, risk, positive, _ = decisions(
            p, eligible, quality, t, gamma, y, w, cfg
        )
        deficit = np.maximum(
            0, cfg["positive_acceptance_floor"] - positive[np.isfinite(positive)]
        ).sum()
        violation = max(0, work - cfg["review_budget"]) + deficit
        misses = np.nansum(w[:, None] * np.nan_to_num(y) * ~d)
        return (violation, risk + cfg["review_cost"] * work, work, misses)

    best = score(t, gamma)
    for _ in range(5):
        changed = False
        for k in range(4):
            candidates = []
            for tk in np.arange(0.1, 0.901, 0.05):
                for gk in np.arange(0.5, 0.951, 0.05):
                    nt = t.copy()
                    ng = gamma.copy()
                    nt[k] = tk
                    ng[k] = gk
                    candidates.append((score(nt, ng), nt, ng))
            value, nt, ng = min(candidates, key=lambda row: row[0])
            if value < best:
                best = value
                t = nt
                gamma = ng
                changed = True
        if not changed:
            break
    return {
        "feasible": best[0] < 1e-10,
        "thresholds": t.tolist(),
        "confidence": gamma.tolist(),
        "development_objective": best[1],
        "constraint_violation": best[0],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["prepare", "train", "evaluate"])
    parser.add_argument("--config", default="config.json")
    parser.add_argument("--data", default="data/ecg.npz")
    parser.add_argument("--output", default="outputs")
    parser.add_argument("--manifest")
    parser.add_argument("--mapping")
    parser.add_argument(
        "--variant",
        choices=["masked", "observation-head", "unknown-negative", "intersection"],
        default="masked",
    )
    parser.add_argument("--omit-positive", type=float, default=0)
    args = parser.parse_args()
    cfg = json.loads(Path(args.config).read_text())
    torch.manual_seed(cfg["seed"])
    if args.command == "prepare":
        prepare(args, cfg)
        return
    data = np.load(args.data)
    check_groups(data["patient"], data["split"])
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    if args.command == "train":
        train = np.flatnonzero(data["split"] == "train")
        means = np.array(
            [
                data["x"][train, c, :][data["mask"][train, c, :]].mean()
                for c in range(12)
            ]
        )
        scales = np.array(
            [data["x"][train, c, :][data["mask"][train, c, :]].std() for c in range(12)]
        ).clip(1e-06)
        if not np.isfinite(means).all():
            raise ValueError("Every lead needs observed training samples")
        np.savez(out / "normalization.npz", means=means, scales=scales)
        y = data["y"].copy()
        rng = np.random.default_rng(cfg["seed"])
        omit = (
            (y == 1)
            & (rng.random(y.shape) < args.omit_positive)
            & (data["split"] == "train")[:, None]
        )
        y[omit] = np.nan
        if args.variant == "unknown-negative":
            y[data["split"] == "train"] = np.nan_to_num(y[data["split"] == "train"])
        support = np.ones(4, bool)
        if args.variant == "intersection":
            support = np.array(
                [
                    all(
                        (
                            np.isfinite(
                                y[
                                    (data["source"] == source)
                                    & (data["split"] == "train"),
                                    k,
                                ]
                            ).any()
                            for source in np.unique(data["source"][train])
                        )
                    )
                    for k in range(4)
                ]
            )
            y[:, ~support] = np.nan
        cfg.update(
            variant=args.variant,
            support=support.tolist(),
            observation_classes=(
                len(data["observation_codes"])
                if args.variant == "observation-head"
                else 0
            ),
        )
        model = ResNet(cfg["observation_classes"])

        def items(split):
            ix = np.flatnonzero(data["split"] == split)
            x = (data["x"][ix] - means[None, :, None]) / scales[None, :, None]
            mask = data["mask"][ix]
            x = np.concatenate([np.where(mask, x, 0), mask], axis=1)
            auxiliary = patient_weights(data["patient"][ix])
            for source in np.unique(data["source"][ix]):
                same = data["source"][ix] == source
                auxiliary[same] /= len(np.unique(data["patient"][ix][same]))
            return tuple(
                (
                    torch.tensor(a, dtype=torch.float32)
                    for a in [
                        x,
                        y[ix],
                        data["obs"][ix],
                        patient_weights(data["patient"][ix]),
                        auxiliary,
                    ]
                )
            )

        def objective(m, x, target, obs, w, auxiliary):
            canonical, observation = m(x)
            known = torch.isfinite(target)
            weights = w[:, None] * known
            numerator = (
                F.binary_cross_entropy_with_logits(
                    canonical, torch.nan_to_num(target), reduction="none"
                )
                * weights
            ).sum()
            denominator = weights.sum()
            if observation is not None:
                om = torch.isfinite(obs)
                ow = auxiliary[:, None] * om
                num = (
                    F.binary_cross_entropy_with_logits(
                        observation, torch.nan_to_num(obs), reduction="none"
                    )
                    * ow
                ).sum()
                if ow.sum() > 0:
                    numerator += (
                        cfg["source_head_weight"] * num / ow.sum() * denominator
                    )
            return (numerator, denominator)

        fit(model, items("train"), items("validation"), objective, cfg, out)
        return
    checkpoint = torch.load(out / "model.pt", map_location="cpu", weights_only=False)
    cfg = checkpoint["config"]
    model = ResNet(cfg["observation_classes"])
    model.load_state_dict(checkpoint["state"])
    model.eval()
    norm = np.load(out / "normalization.npz")
    logits = []
    with torch.no_grad():
        for ix in np.array_split(
            np.arange(len(data["y"])), max(1, int(np.ceil(len(data["y"]) / 64)))
        ):
            mask = data["mask"][ix]
            x = (data["x"][ix] - norm["means"][None, :, None]) / norm["scales"][
                None, :, None
            ]
            x = np.concatenate([np.where(mask, x, 0), mask], 1)
            logits.append(model(torch.tensor(x, dtype=torch.float32))[0].numpy())
    logits = np.concatenate(logits)
    records = []
    for record in np.unique(data["record"]):
        ix = np.flatnonzero(data["record"] == record)
        i = ix[0]
        records.append(
            (
                logits[ix].mean(0),
                data["y"][i],
                data["patient"][i],
                data["split"][i],
                data["eligibility"][i] & np.array(cfg["support"]),
                data["quality"][i],
                record,
            )
        )
    l, y, patients, splits, eligible, quality, record_ids = [
        np.array([r[i] for r in records]) for i in range(7)
    ]
    cal = splits == "calibration"
    policy = splits == "policy"
    test = splits == "test"
    if not cal.any() or not policy.any() or (not test.any()):
        raise ValueError("Separate calibration, policy and test records are required")
    temperatures = np.array([temperature(l[cal, k], y[cal, k]) for k in range(4)])
    probs = probabilities(l, temperatures)
    operating = select_policy(
        probs[policy],
        eligible[policy],
        quality[policy],
        y[policy],
        patient_weights(patients[policy]),
        cfg,
    )
    result = {"policy": operating, "temperatures": temperatures.tolist(), "queries": {}}
    if not operating["feasible"]:
        (out / "metrics.json").write_text(
            json.dumps(json_ready(result), indent=2, allow_nan=False)
        )
        return
    w = patient_weights(patients[test])
    pred = probs[test]
    yy = y[test]
    released, decision, workload, risk, positive, accept = decisions(
        pred,
        eligible[test],
        quality[test],
        np.array(operating["thresholds"]),
        np.array(operating["confidence"]),
        yy,
        w,
        cfg,
    )
    for k, name in enumerate(QUERIES):
        known = np.isfinite(yy[:, k])
        target = np.nan_to_num(yy[:, k])
        wk = w * known
        pos = wk * target
        neg = wk * (1 - target)
        result["queries"][name] = {
            "all": binary_metrics(yy[:, k], pred[:, k], w),
            "accepted": binary_metrics(yy[released, k], pred[released, k], w[released]),
            "unreviewed_miss": ratio(
                np.sum(pos * released * ~decision[:, k]), pos.sum()
            ),
            "routing_sensitivity": (
                1 - ratio(np.sum(pos * released * ~decision[:, k]), pos.sum())
                if pos.sum()
                else None
            ),
            "positive_acceptance": ratio(np.sum(pos * released), pos.sum()),
            "accepted_sensitivity": ratio(
                np.sum(pos * released * decision[:, k]), np.sum(pos * released)
            ),
            "accepted_specificity": ratio(
                np.sum(neg * released * ~decision[:, k]), np.sum(neg * released)
            ),
            "false_positive_all_negatives": ratio(
                np.sum(neg * released * decision[:, k]), neg.sum()
            ),
        }
    result.update(
        review=workload,
        automatic_coverage=1 - workload,
        decision_risk=risk,
        patients_any_review=float(
            np.mean(
                [
                    np.any(~released[patients[test] == v])
                    for v in np.unique(patients[test])
                ]
            )
        ),
        report_count=int(test.sum()),
        semantic_review=float(np.average(~eligible[test].all(1), weights=w)),
        quality_review=float(
            np.average(eligible[test].all(1) & ~quality[test], weights=w)
        ),
    )
    (out / "metrics.json").write_text(
        json.dumps(json_ready(result), indent=2, allow_nan=False)
    )
    np.savez(
        out / "reports.npz",
        patient=patients[test],
        record=record_ids[test],
        p=pred,
        y=yy,
        released=released,
        decision=decision,
        accepted_queries=accept,
    )
    plt.bar(QUERIES, [result["queries"][k]["unreviewed_miss"] or 0 for k in QUERIES])
    plt.ylabel("Unreviewed misses / supported positives")
    plt.tight_layout()
    plt.savefig(out / "misses.png")
    plt.close()


if __name__ == "__main__":
    main()
