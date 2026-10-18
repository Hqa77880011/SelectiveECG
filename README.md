# SelectiveECG

Implementation of selective ECG interpretation under label-policy shift.

## Installation

Use Python 3.10+. Create and activate a virtual environment before installing:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

In PowerShell, replace the activation line with `.venv\Scripts\Activate.ps1`.

## Data and preprocessing

Sources are [PTB-XL 1.0.3](https://physionet.org/content/ptb-xl/1.0.3/) and the `cpsc_2018` subset of [Challenge 2020](https://physionet.org/content/challenge-2020/1.0.2/). Export `records.csv` with `patient,source,record,codes,eligibility,split`. `record` is a WFDB basename; `codes` is a JSON list of documented annotation codes; `eligibility` is a four-bit JSON list from the declared output policy, never from reference labels. Query order is AFIB, CRBBB, CLBBB, 1AVB. `--mapping` accepts a JSON query-to-positive-code-list mapping for a verified external vocabulary. An explicit NORM with no contradictory target code supplies negatives; all other unsupported query targets remain unknown.

Use PTB-XL folds 1–8 for train plus patient-disjoint validation, fold 9 split by patient into `calibration` and `policy`, and fold 10 as `test`. All repeated recordings remain grouped. The optional `closed_world` column applies only to explicitly declared source observation-head targets. `rail_mv`, when supplied from acquisition metadata, enables the clipping rule. Preparation reads physical waveform units, harmonizes the 12-lead order, resamples to 100 Hz and keeps missing/padding masks. Longer recordings use a fixed mean of 10-second window logits. Quality review requires at least three invalid leads, based on missingness, two-second flatline and known acquisition rails.

```bash
python -m selectiveecg prepare --manifest records.csv --config config.json
```

## Training and inference

```bash
python -m selectiveecg train --config config.json --output outputs --variant masked
```

## Evaluation

```bash
python -m selectiveecg evaluate --config config.json --output outputs
python -m selectiveecg.analysis --input outputs/reports.npz --output outputs/patient_analysis.json --draws 2000
```

## Ablations

Supported controls: `masked`, `observation-head`, `unknown-negative`, `intersection`. Reuse the same inputs, seed and budget. For retrained controls, evaluate the corresponding saved model.

```bash
python -m selectiveecg train --config config.json --output outputs/masked --variant masked
```

## Outputs

`data/ecg.npz` holds waveforms/masks. `outputs/` contains model/normalization/training records, `reports.npz`, `metrics.json`, `misses.png` and optional patient analysis.

Keep `config.json` with the run; it contains model/data settings and the seed where the CLI does not expose one. Use separate output directories for different seeds/conditions.

Source inspection: Python 3.13.5, numpy 2.1.3, pandas 2.2.3, scipy 1.15.3, scikit-learn 1.6.1, torch 2.7.1+cu118, matplotlib 3.10.0 (installed versions inspected). Checks covered Python syntax, configuration parsing and CLI arguments. No training, full evaluation or CI was run in this update.
