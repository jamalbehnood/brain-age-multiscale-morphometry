# Output catalogue

The consolidated Excel workbook is the primary index of numerical outputs.

| Output | Status | Intended use |
|---|---|---|
| `RepeatPerf_Summary` | Primary | Mean, SD, minimum, and maximum accuracy across complete raw OOF repeats |
| `RepeatPerf_Detail` | Primary support | Per-repeat MAE, RMSE, R², and error summaries |
| `Performance` | Secondary | Metrics from participant-averaged OOF predictions |
| `NestedCV_Folds` | Audit | Selected model, hyperparameters, selected k, correction parameters, and fold metrics |
| `Subject_OOF_Predictions` | Secondary | Participant-averaged raw and corrected predictions for diagnostic plots |
| `Paired_Scales` | Formal comparison | Adjacent anatomical-scale contrasts |
| `Paired_Phenotypes` | Formal comparison | ROI All5 versus single-phenotype models |
| `FD_Incremental` | Formal comparison | All5 versus NoFD at each anatomical scale |
| `Feature_Stability` | Diagnostic | Selection frequency across outer fits |
| `OOF_PermutationImportance` | Primary explainability | Held-out MAE dependence on each feature |
| `Nuisance_*` | Sensitivity | Morphology plus sex, and morphology plus sex and eTIV |
| `FixedEN_*` | Sensitivity | FD incremental value with Elastic Net fixed |
| `LearningCurve_*` | Supplementary | Nested sample-size diagnostics |
| `SexBias_*` | Supplementary audit | Group metrics and age-adjusted sex-related terms |
| `Final_Descriptive_*` | Descriptive only | Full-data model, coefficients, and SHAP outputs |

Participant-level CSV files, trained models, and any output containing IDs must remain outside the public repository unless explicit data-governance approval permits release.
