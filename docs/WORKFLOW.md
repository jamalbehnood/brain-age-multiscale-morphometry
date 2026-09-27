# Analysis workflow

## 1. Validate the release

1. Create and activate a clean Python environment.
2. Install `requirements.txt`.
3. Run `python tests/smoke_test.py`.
4. Confirm that XGBoost and SHAP import successfully before the manuscript run.

## 2. Prepare the authorised workbook

1. Use one row per participant and unique IDs.
2. Match the exact field names in `data/expected_columns.csv`.
3. Retain the five morphometric families: volume, area, thickness, curvature, and FD.
4. Do not place cognition, education, normative Z scores, centiles, derivatives, or turning points into the model feature sets.

## 3. Configure paths

Use the repository defaults or set `BRAINAGE_DATA_PATH`, `BRAINAGE_SHEET_NAME`, and `BRAINAGE_OUTPUT_DIR`. No source edit is required.

## 4. Execute

Run `python src/brain_age_multiscale.py`. The pipeline validates the manifest, generates identical outer splits across feature sets, performs inner model selection, predicts untouched outer folds, and exports the audit trail.

## 5. Verify a completed run

- Fifteen feature sets are present.
- Each feature set has 25 outer evaluations.
- Every participant has exactly one validation prediction in each repeat and feature set.
- Primary accuracy is taken from `RepeatPerf_Summary`, not `Performance`.
- Scale, phenotype, and FD comparisons use participant-level paired differences.
- Raw and corrected accuracy outputs are not interchanged.
- SHAP outputs are labelled descriptive.

## 6. Preserve the release

Record package versions, commit the source and documentation only, and tag the exact commit used for a manuscript revision. Do not commit participant-level data or result files.
