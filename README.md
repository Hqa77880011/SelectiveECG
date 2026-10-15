# SelectiveECG

Implementation of selective ECG interpretation under label-policy shift.

```bash
python -m pip install -e .
python -m selectiveecg prepare --manifest records.csv
python -m selectiveecg train --variant masked
python -m selectiveecg evaluate
```

Sources are [PTB-XL 1.0.3](https://physionet.org/content/ptb-xl/1.0.3/) and the `cpsc_2018` subset of [Challenge 2020](https://physionet.org/content/challenge-2020/1.0.2/). Export `records.csv` with `patient,source,record,codes,eligibility,split`. `record` is a WFDB basename; `codes` is a JSON list of documented annotation codes; `eligibility` is a four-bit JSON list from the declared output policy, never from reference labels. Query order is AFIB, CRBBB, CLBBB, 1AVB. `--mapping` accepts a JSON query-to-positive-code-list mapping for a verified external vocabulary. An explicit NORM with no contradictory target code supplies negatives; all other unsupported query targets remain unknown.

Use PTB-XL folds 1–8 for train plus patient-disjoint validation, fold 9 split by patient into `calibration` and `policy`, and fold 10 as `test`. All repeated recordings remain grouped. The optional `closed_world` column applies only to explicitly declared source observation-head targets. `rail_mv`, when supplied from acquisition metadata, enables the clipping rule. Preparation reads physical waveform units, harmonizes the 12-lead order, resamples to 100 Hz and keeps missing/padding masks. Longer recordings use a fixed mean of 10-second window logits. Quality review requires at least three invalid leads, based on missingness, two-second flatline and known acquisition rails.

The ResNet-18 encoder uses four width stages and two residual blocks per stage; values and availability masks form its 24 input channels. Canonical BCE masks unknown targets and weights recordings by patient. `--variant observation-head` adds documented source-vocabulary supervision; `unknown-negative` and `intersection` implement the corresponding training controls. `--omit-positive 0.2`, `0.4` or `0.6` performs a seeded training-only omission intervention. Each variant needs its own output directory. The intersection control retains the complete request set and marks untrained queries semantically unresolved.

Calibration is query-wise supported-label temperature scaling. Operating-policy patients select probability and confidence thresholds by five-pass deterministic coordinate search. Selection minimizes weighted decision cost plus review cost subject to the report budget and per-query positive acceptance floor. Any unresolved query reviews the complete report. An empty feasible set is reported; test thresholds are never adjusted to hide it.

`config.json` controls seed, optimizer, source-head weight, false-negative/false-positive/review costs, review budget and positive-acceptance floor. Outputs include model/normalization files, history, policy and query metrics, report-level predictions and `misses.png`. Metrics retain separate denominators for unreviewed misses, routing sensitivity, positive acceptance, accepted-set sensitivity/specificity and all-negative false-positive reporting. AUROC/AUPRC/Brier/ECE and calibration intercept/slope use only supported references; accepted subsets are reported separately. Workload counts complete ECG reports once and also reports patients with any reviewed encounter. Unknown references never enter discrimination or calibration as negatives.

The observation-head vocabulary is stored in the prepared data. Its auxiliary loss balances source contributions after patient weighting; the canonical task and inference gates do not use the patient's source annotations. Preserve the same source calibration and policy records when appending external test records for frozen transfer. Semantic eligibility remains external policy metadata throughout.

## Patient/cell analysis

Run `python -m selectiveecg.analysis --draws 2000` after evaluation to compute grouped bootstrap intervals and the supported subgroup or repeatability summaries. The input/output paths can be set with `--input` and `--output`.
