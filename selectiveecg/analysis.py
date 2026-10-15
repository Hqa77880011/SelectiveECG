import argparse
import json
from pathlib import Path
import numpy as np
from .common import bootstrap, patient_weights, ratio


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="outputs/reports.npz")
    parser.add_argument("--output", default="outputs/patient_analysis.json")
    parser.add_argument("--draws", type=int, default=2000)
    args = parser.parse_args()
    data = np.load(args.input)
    weights = patient_weights(data["patient"])
    result = {}
    for query in range(data["y"].shape[1]):

        def missed(ix):
            positive = np.isfinite(data["y"][ix, query]) & (data["y"][ix, query] == 1)
            return ratio(
                np.sum(
                    weights[ix]
                    * positive
                    * data["released"][ix]
                    * ~data["decision"][ix, query]
                ),
                np.sum(weights[ix] * positive),
            )

        result[str(query)] = {
            "unreviewed_miss_ci95": bootstrap(data["patient"], missed, args.draws, 11)
        }
    Path(args.output).write_text(
        json.dumps(json_ready(result), indent=2, allow_nan=False)
    )


if __name__ == "__main__":
    main()
