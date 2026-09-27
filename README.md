# Multiscale Cortical Brain-Age Pipeline

This repository contains the analysis code used to compare cortical morphometric representations for brain-age prediction and to test the incremental predictive value of fractal dimension (FD). It implements leakage-controlled repeated nested cross-validation across whole-cortex, hemispheric, bilateral-lobar, Desikan-Killiany 68-region, and combined multiscale representations.

The code accompanies the manuscript **“Multiscale Cortical Morphometry for Brain-Age Prediction: Anatomical Scale and the Incremental Value of Fractal Dimension.”** Participant-level data and generated participant-level outputs are not included.

## Analysis design

- Target: chronological age
- Morphometric phenotypes: cortical volume, surface area, thickness, mean curvature, and FD
- Algorithms: Elastic Net, random forest, and XGBoost
- Validation: five repeats of nested five-fold cross-validation
- Primary accuracy: mean and standard deviation of MAE, RMSE, and R² across five complete raw out-of-fold repeats
- Formal comparisons: participant-paired bootstrap confidence intervals, two-sided sign-flip tests, and Benjamini-Hochberg FDR correction within prespecified families
- Explainability: outer-validation permutation importance and feature-selection stability; full-data SHAP is descriptive only

## Repository structure

```text
src/brain_age_multiscale.py   Main analysis pipeline
data/README.md                Data-access and preparation instructions
data/expected_columns.csv     Input data contract
docs/CODE_ARCHITECTURE.*      Architecture and safe-extension guide
docs/WORKFLOW.md              Execution and validation workflow
docs/OUTPUT_FILES.md          Output catalogue and interpretation
scripts/run_windows.bat       Windows launcher
scripts/run_linux_mac.sh      Linux/macOS launcher
tests/smoke_test.py           Non-data structural checks
results/                      Local output directory; contents are ignored
```

## Installation

Python 3.11 or 3.12 is recommended for broad binary-package compatibility.

```bash
python -m venv .venv
```

Activate the environment, then install dependencies:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Input data

The default workbook location is:

```text
data/BrainNormR_grouped_structural_dataset.xlsx
```

The default worksheet is `ModelingData`. The repository does not contain the study dataset. Use only data for which you have the required ethical and governance permissions. See `data/README.md` and `data/expected_columns.csv` for the exact schema.

## Run the analysis

With the workbook at the default location:

```bash
python src/brain_age_multiscale.py
```

Alternatively, set paths without editing the source code.

Windows PowerShell:

```powershell
$env:BRAINAGE_DATA_PATH = "D:\data\BrainNormR_grouped_structural_dataset.xlsx"
$env:BRAINAGE_SHEET_NAME = "ModelingData"
$env:BRAINAGE_OUTPUT_DIR = "D:\results\BrainAge_Multiscale_Q1"
python src\brain_age_multiscale.py
```

Linux or macOS:

```bash
export BRAINAGE_DATA_PATH="/path/to/BrainNormR_grouped_structural_dataset.xlsx"
export BRAINAGE_SHEET_NAME="ModelingData"
export BRAINAGE_OUTPUT_DIR="/path/to/results"
python src/brain_age_multiscale.py
```

The full run is computationally intensive. `TEST_MODE` in the source is for syntax and export debugging only; its outputs must not be reported as study results.

## Structural test

Run the non-data smoke test before a full analysis:

```bash
python tests/smoke_test.py
```

This verifies the prespecified feature-set counts, absence of duplicate multiscale predictors, FDR helper behaviour, and release settings. It does not validate scientific results or replace a full run on the authorised dataset.

## Reproducibility boundary

Imputation, zero-variance filtering, scaling, feature selection, algorithm selection, hyperparameter tuning, and age-bias correction are fitted within the relevant training data. The outer-validation partition is used only for evaluation and held-out permutation importance. Do not move sample-fitted operations outside the cross-validation loops.

## Documentation

The detailed architecture guide is provided in both Word and PDF formats under `docs/`. Read it before modifying feature families, anatomical scales, cross-validation logic, inference families, or explainability outputs.

## Data and privacy

Do not commit participant-level workbooks, identifiers, out-of-fold prediction files, fitted models trained on restricted data, or generated results containing IDs. The supplied `.gitignore` blocks the common forms of these files but does not replace manual review before each commit.

## Citation

Repository citation metadata are provided in `CITATION.cff`. If the accompanying manuscript receives a DOI, update that file and the manuscript citation before creating a release.

## License

The source code is released under the MIT License. Dataset access and third-party software remain subject to their own terms.
