# Input data contract

The study dataset is not distributed with this repository. Place an authorised copy at `data/BrainNormR_grouped_structural_dataset.xlsx` or set `BRAINAGE_DATA_PATH` to another location.

## Workbook requirements

- Worksheet: `ModelingData` by default
- One row per participant
- `ID`: nonmissing and unique
- `Age`: numeric and nonmissing for analysed rows
- Structural predictors: names must match `expected_columns.csv`
- `sex`: required for the sex-bias audit and nuisance sensitivity
- `eTIV`: required for the eTIV nuisance sensitivity

Missing predictor values may remain; median imputation is fitted inside each training partition. Predictors with fewer than 50% observed values are removed from every affected feature set before modelling, with an explicit warning. A workbook containing only headers or nonnumeric structural values is rejected.

## Aggregation assumptions

The pipeline expects precomputed whole-cortex, hemispheric, and bilateral-lobar summaries. Volume and surface area are summed. Thickness, mean curvature, and FD are surface-area-weighted means. Aggregated FD is a weighted composite of regional FD, not a new fractal estimate from a merged cortical surface.

## Privacy

Never upload the participant workbook, direct identifiers, linkage keys, or participant-level generated results to a public repository. Review staged files manually before every commit.
