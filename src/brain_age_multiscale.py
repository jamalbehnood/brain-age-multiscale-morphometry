# ============================================================
# BrainAge Multiscale Morphometry Pipeline — Q1-grade revision
# ============================================================
#
# Scientific question
# -------------------
# Which cortical morphometric phenotype and which spatial scale carry the
# most stable out-of-sample information for estimating individual Brain Age?
#
# Primary novelty
# ---------------
# 1) Direct spatial-scale benchmarking:
#       WholeBrain vs Hemisphere vs Bilateral Lobe vs DK68 ROI
#       vs Multiscale.
# 2) Phenotype ablation:
#       Volume, Area, Thickness, Curvature, FD.
# 3) Explicit incremental-value test for FD:
#       conventional morphometry (Volume+Area+Thickness+Curvature)
#       versus all five phenotypes (+FD).
# 4) Leakage-safe repeated nested CV in which preprocessing, feature
#       selection, model selection, and hyperparameter tuning occur inside
#       the training data of each outer fold.
# 5) Fold-specific Brain-Age bias correction estimated from OOF predictions
#       within each outer-training set, then applied to the untouched
#       outer-validation fold.
# 6) Out-of-sample feature-selection stability and optional OOF permutation
#       importance; SHAP is descriptive and generated only after performance
#       estimation is complete.
#
# IMPORTANT SEPARATION FROM BrainNormR PAPERS
# -------------------------------------------
# This script intentionally DOES NOT use:
#   - Norm_Z / CF_Norm_Z
#   - GAMLSS centiles / derivatives / turning points
#   - cognitive variables
#   - education as a predictor
# The target is chronological Age, and the prediction is interpreted as
# estimated Brain Age. Brain-Age Gap = predicted brain age - chronological age.
#
# DATASET EXPECTED
# ----------------
# Sheet: ModelingData
# Required columns include:
#   ID, Age, sex, edu, Hand_encoded, eTIV
#   DK68 ROI x 5 measures
#   WholeBrain x 5 measures
#   bilateral lobe x 5 measures
#   hemisphere x 5 measures
#
# The exact feature names below are matched to:
#   BrainNormR_grouped_structural_dataset.xlsx
#
# ============================================================

from __future__ import annotations

import os
import json
import math
import warnings
from pathlib import Path
from collections import Counter, defaultdict

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

from sklearn.base import clone
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_regression
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import ElasticNet
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import (
    StratifiedKFold,
    GridSearchCV,
    cross_val_predict,
)
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.inspection import permutation_importance

import joblib

__version__ = "1.0.0"

# Optional XGBoost. For the final paper, installing xgboost is recommended.
try:
    from xgboost import XGBRegressor
    HAS_XGBOOST = True
except Exception:
    HAS_XGBOOST = False

# Optional SHAP. The main performance analysis does NOT depend on SHAP.
try:
    import shap
    HAS_SHAP = True
except Exception:
    HAS_SHAP = False


# ============================================================
# 0. USER SETTINGS
# ============================================================

SEED = 1234

# Paths are configured without editing the source file. Environment variables
# override the repository-relative defaults; see README.md and .env.example.
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = os.environ.get(
    "BRAINAGE_DATA_PATH",
    str(REPOSITORY_ROOT / "data" / "BrainNormR_grouped_structural_dataset.xlsx"),
)
SHEET_NAME = os.environ.get("BRAINAGE_SHEET_NAME", "ModelingData")
OUTPUT_DIR = os.environ.get(
    "BRAINAGE_OUTPUT_DIR",
    str(REPOSITORY_ROOT / "results"),
)

# Repeated nested CV
OUTER_FOLDS = 5
N_REPEATS = 5          # Prespecified manuscript analysis.
INNER_FOLDS = 5

# Bootstrap/permutation for final paired comparisons
N_BOOTSTRAP = 2000
N_PAIRED_PERM = 5000

# OOF permutation importance is computationally expensive.
# These two are the most informative a-priori feature sets.
PERM_IMPORTANCE_FEATURESETS = {
    "ROI_All5",
    "Multiscale_All5",
}
PERM_IMPORTANCE_REPEATS = 5

# Final descriptive SHAP model.
RUN_FINAL_SHAP = True
SHAP_MAX_DISPLAY = 25

# Test mode for debugging only.
TEST_MODE = False
if TEST_MODE:
    N_REPEATS = 1
    N_BOOTSTRAP = 200
    N_PAIRED_PERM = 500
    PERM_IMPORTANCE_REPEATS = 2

# Pure cortical-morphometry Brain Age is the primary analysis.
# If True, sex and eTIV are added to EVERY feature set as nuisance predictors.
# Keep False for the primary paper and use True only as a sensitivity analysis.
ADD_NUISANCE_PREDICTORS = False
NUISANCE_COLUMNS = ["sex", "eTIV"]

# Require XGBoost in the three-model comparison?
# If unavailable, HistGradientBoosting is NOT silently substituted because
# that would change the prespecified analysis. Instead, ENet + RF are run and
# the log explicitly reports that XGBoost was unavailable.
USE_XGBOOST_IF_AVAILABLE = True
# For the final manuscript the advertised three-algorithm comparison must not
# silently degrade to two algorithms because a package is missing.
REQUIRE_XGBOOST_FOR_FINAL = True
REQUIRE_SHAP_FOR_FINAL = True

# Main feature-set plan.
RUN_SCALE_BENCHMARK = True
RUN_ROI_PHENOTYPE_ABLATION = True
RUN_FD_INCREMENTAL_TEST = True

# Supplementary sample-size diagnostic: leakage-safe nested learning curves.
# Each fraction is drawn only from the relevant outer-training partition;
# the outer-validation subjects remain untouched. Model family, feature count,
# and hyperparameters are reselected by inner CV at every curve point.
RUN_LEARNING_CURVES = True
LEARNING_CURVE_FEATURESETS = [
    "Lobe_All5",
    "ROI_All5",
    "Multiscale_All5",
]
LEARNING_CURVE_FRACTIONS = [0.30, 0.45, 0.60, 0.75, 0.90, 1.00]

if TEST_MODE:
    LEARNING_CURVE_FEATURESETS = ["Lobe_All5"]
    LEARNING_CURVE_FRACTIONS = [0.50, 1.00]

# Supplementary sex-bias/fairness audit on repeated OOF predictions.
# The primary dataset coding is 0=Male and 1=Female. Change this mapping only
# if the source workbook uses a different documented coding scheme.
RUN_SEX_BIAS_ANALYSIS = True
SEX_COLUMN = "sex"
SEX_LABEL_MAP = {0: "Male", 1: "Female"}
SEX_BIAS_FEATURESETS = ["ROI_All5", "Multiscale_All5"]

# Extended reviewer-facing robustness analyses.
RUN_REPEAT_LEVEL_PERFORMANCE = True
RUN_PAIRED_SCALE_COMPARISONS = True
RUN_PAIRED_PHENOTYPE_COMPARISONS = True
RUN_NUISANCE_SENSITIVITY = True
RUN_FIXED_ENET_FD_SENSITIVITY = True

# Nuisance sensitivity is intentionally restricted to the five scale models
# and their NoFD counterparts. It is saved separately from the primary run.
NUISANCE_SENSITIVITY_MODES = {
    "Morphology_plus_Sex": ["sex"],
    "Morphology_plus_Sex_eTIV": ["sex", "eTIV"],
}

if TEST_MODE:
    RUN_PAIRED_SCALE_COMPARISONS = True
    RUN_PAIRED_PHENOTYPE_COMPARISONS = True
    # These are computationally intensive; syntax/smoke testing does not need
    # to rerun hundreds of nested searches.
    RUN_NUISANCE_SENSITIVITY = False
    RUN_FIXED_ENET_FD_SENSITIVITY = False

# Reproducibility / output formatting
np.random.seed(SEED)
warnings.filterwarnings("ignore", category=FutureWarning)
sns.set_theme(style="whitegrid")

Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)


# ============================================================
# 1. EXACT DATA DICTIONARY FOR THE UPLOADED DATASET
# ============================================================

MEASURES = ["volume", "area", "thickness", "curvature", "FD"]

DK34 = [
    "bankssts",
    "caudalanteriorcingulate",
    "caudalmiddlefrontal",
    "cuneus",
    "entorhinal",
    "fusiform",
    "inferiorparietal",
    "inferiortemporal",
    "isthmuscingulate",
    "lateraloccipital",
    "lateralorbitofrontal",
    "lingual",
    "medialorbitofrontal",
    "middletemporal",
    "parahippocampal",
    "paracentral",
    "parsopercularis",
    "parsorbitalis",
    "parstriangularis",
    "pericalcarine",
    "postcentral",
    "posteriorcingulate",
    "precentral",
    "precuneus",
    "rostralanteriorcingulate",
    "rostralmiddlefrontal",
    "superiorfrontal",
    "superiorparietal",
    "superiortemporal",
    "supramarginal",
    "frontalpole",
    "temporalpole",
    "transversetemporal",
    "insula",
]

LOBES = [
    "Frontal",
    "Parietal",
    "Temporal",
    "Occipital",
    "Cingulate",
    "Insula",
]


def roi_columns(measures=MEASURES):
    cols = []
    # Keep the exact column naming used by the uploaded workbook.
    # FD columns exist for both hemispheres, followed by the other measures.
    for measure in measures:
        for hemi in ["lh", "rh"]:
            for roi in DK34:
                cols.append(f"{hemi}_{roi}_{measure}")
    return cols


def wholebrain_columns(measures=MEASURES):
    return [f"wholebrain_{m}" for m in measures]


def hemisphere_columns(measures=MEASURES):
    return [
        f"{hemi}_hemisphere_{m}"
        for hemi in ["lh", "rh"]
        for m in measures
    ]


def bilateral_lobe_columns(measures=MEASURES):
    return [
        f"bilateral_{lobe}_{m}"
        for lobe in LOBES
        for m in measures
    ]



def measure_only(columns, measure):
    suffix = f"_{measure}"
    return [c for c in columns if c.endswith(suffix)]


def add_nuisance(cols):
    if not ADD_NUISANCE_PREDICTORS:
        return list(cols)
    return list(dict.fromkeys(list(cols) + NUISANCE_COLUMNS))


# ============================================================
# 2. FEATURE-SET MANIFEST
# ============================================================

CONVENTIONAL_MEASURES = ["volume", "area", "thickness", "curvature"]

SCALE_COLUMNS = {
    "WholeBrain": wholebrain_columns(MEASURES),
    "Hemisphere": hemisphere_columns(MEASURES),
    "Lobe": bilateral_lobe_columns(MEASURES),
    "ROI": roi_columns(MEASURES),
}

# Primary multiscale representation is deliberately ANATOMICAL only.
# Yeo7 functional-network aggregates are excluded from the main ML analysis.
SCALE_COLUMNS["Multiscale"] = list(dict.fromkeys(
    SCALE_COLUMNS["WholeBrain"]
    + SCALE_COLUMNS["Hemisphere"]
    + SCALE_COLUMNS["Lobe"]
    + SCALE_COLUMNS["ROI"]
))


def build_feature_sets():
    feature_sets = {}

    # A. Primary spatial-scale benchmark: all 5 morphometric phenotypes.
    if RUN_SCALE_BENCHMARK:
        for scale, cols in SCALE_COLUMNS.items():
            feature_sets[f"{scale}_All5"] = add_nuisance(cols)

    # B. No-FD versions for direct incremental-value testing.
    if RUN_FD_INCREMENTAL_TEST:
        for scale in ["WholeBrain", "Hemisphere", "Lobe", "ROI", "Multiscale"]:
            cols = SCALE_COLUMNS[scale]
            no_fd = [c for c in cols if not c.endswith("_FD")]
            feature_sets[f"{scale}_NoFD"] = add_nuisance(no_fd)

    # C. Phenotype ablation at the DK68 ROI scale.
    # This is prespecified rather than selected post hoc.
    if RUN_ROI_PHENOTYPE_ABLATION:
        roi_all = SCALE_COLUMNS["ROI"]
        for measure in MEASURES:
            feature_sets[f"ROI_{measure}"] = add_nuisance(
                measure_only(roi_all, measure)
            )

    return feature_sets


FEATURE_SETS = build_feature_sets()


# ============================================================
# 3. LOAD + STRICT DATA VALIDATION
# ============================================================

def load_dataset(path, sheet_name):
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"\nDataset not found:\n{path}\n\n"
            "Place the authorised workbook in data/ or set the "
            "BRAINAGE_DATA_PATH environment variable."
        )

    df = pd.read_excel(path, sheet_name=sheet_name)

    required_basic = ["ID", "Age"]
    missing_basic = [c for c in required_basic if c not in df.columns]
    if missing_basic:
        raise ValueError(f"Missing required columns: {missing_basic}")

    # Age must be numeric and nonmissing for supervised learning.
    df["Age"] = pd.to_numeric(df["Age"], errors="coerce")
    before = len(df)
    df = df.loc[df["Age"].notna()].copy()
    after = len(df)

    if after < before:
        print(f"Dropped {before-after} rows with missing/non-numeric Age.")

    if after < 50:
        raise ValueError(
            f"Only {after} rows have valid Age. This is insufficient for the "
            "prespecified repeated nested-CV analysis."
        )

    # Duplicate IDs are not allowed because repeated-CV predictions are
    # aggregated by subject.
    if df["ID"].duplicated().any():
        dup = df.loc[df["ID"].duplicated(), "ID"].astype(str).tolist()[:10]
        raise ValueError(
            "Duplicate subject IDs detected. Example(s): "
            + ", ".join(dup)
        )

    return df


def validate_feature_manifest(df, feature_sets):
    all_needed = sorted(set(
        c for cols in feature_sets.values() for c in cols
    ))

    missing = [c for c in all_needed if c not in df.columns]
    if missing:
        msg = "\n".join(missing[:50])
        raise ValueError(
            f"{len(missing)} prespecified feature columns are missing.\n"
            f"First missing columns:\n{msg}"
        )

    # Convert intended model features to numeric without touching the original
    # non-model columns (e.g., cognition).
    for c in all_needed:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # Detect an accidentally empty workbook (headers + IDs but no data).
    nonmissing = df[all_needed].notna().sum()
    usable_feature_count = int((nonmissing > 0).sum())

    if usable_feature_count == 0:
        raise ValueError(
            "\nMODEL FEATURES ARE EMPTY.\n"
            "The workbook contains the expected column names, but none of the "
            "prespecified structural features contain numeric values.\n\n"
            "This script is correctly mapped to the variables, but the dataset "
            "must contain the 293 subjects' structural measurements before ML "
            "can be run."
        )

    # Features with extremely sparse observations are excluded within each
    # feature set. Missing values that remain are imputed INSIDE CV folds.
    sparse = nonmissing[nonmissing < max(20, int(0.50 * len(df)))].index.tolist()
    if sparse:
        print(
            f"Warning: {len(sparse)} feature(s) have <50% observed data and "
            "will be removed from all feature sets."
        )
        sparse_set = set(sparse)
        for k in list(feature_sets.keys()):
            feature_sets[k] = [
                c for c in feature_sets[k] if c not in sparse_set
            ]

    empty_sets = [k for k, v in feature_sets.items() if len(v) == 0]
    if empty_sets:
        raise ValueError(
            "These feature sets became empty after data checks: "
            + ", ".join(empty_sets)
        )

    return df, feature_sets


# ============================================================
# 4. AGE-STRATIFIED CV HELPERS
# ============================================================

def make_age_bins(y, n_bins=5):
    y = pd.Series(np.asarray(y), index=np.arange(len(y)))
    q = min(n_bins, max(2, y.nunique()))
    bins = pd.qcut(y, q=q, labels=False, duplicates="drop")

    # If any resulting bin is too small for stratified K-fold, reduce q.
    while bins.value_counts().min() < OUTER_FOLDS and q > 2:
        q -= 1
        bins = pd.qcut(y, q=q, labels=False, duplicates="drop")

    return np.asarray(bins)


def make_inner_cv(y_train, random_state):
    y_train = np.asarray(y_train)
    q = min(5, len(np.unique(y_train)))
    bins = pd.qcut(
        pd.Series(y_train),
        q=q,
        labels=False,
        duplicates="drop",
    )
    bins = np.asarray(bins)

    splitter = StratifiedKFold(
        n_splits=INNER_FOLDS,
        shuffle=True,
        random_state=random_state,
    )
    # GridSearchCV accepts an explicit list of (train, validation) indices.
    return list(splitter.split(np.zeros(len(y_train)), bins))


def repeated_outer_splits(y):
    y = np.asarray(y)
    bins = make_age_bins(y, n_bins=5)

    for repeat in range(N_REPEATS):
        splitter = StratifiedKFold(
            n_splits=OUTER_FOLDS,
            shuffle=True,
            random_state=SEED + repeat,
        )
        for fold, (tr, va) in enumerate(
            splitter.split(np.zeros(len(y)), bins),
            start=1,
        ):
            yield repeat + 1, fold, tr, va


# ============================================================
# 5. MODEL SEARCH SPACE
# ============================================================

def candidate_k_values(p):
    if p <= 8:
        return ["all"]
    if p <= 15:
        return [5, "all"]
    if p <= 40:
        return [10, 20, "all"]

    vals = [20, 40, 80]
    if p >= 120:
        vals.append(120)
    vals.append("all")

    # k must never exceed the number of available features.
    return [k for k in vals if k == "all" or k < p] + (
        [] if "all" in vals else ["all"]
    )


def make_base_pipeline():
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("variance", VarianceThreshold(threshold=0.0)),
        ("scaler", StandardScaler()),
        ("selector", SelectKBest(score_func=f_regression, k="all")),
        ("model", ElasticNet(
            max_iter=100000,
            tol=1e-3,
            selection="cyclic",
            random_state=SEED,
        )),
    ])


def make_param_grid(p):
    k_vals = candidate_k_values(p)

    grids = [
        {
            "selector__k": k_vals,
            "model": [
                ElasticNet(
                    max_iter=100000,
                    tol=1e-3,
                    selection="cyclic",
                    random_state=SEED,
                )
            ],
            "model__alpha": [0.05, 0.1, 0.3, 1.0, 3.0, 10.0],
            "model__l1_ratio": [0.1, 0.3, 0.5, 0.7, 0.9],
        },
        {
            "selector__k": k_vals,
            "model": [
                RandomForestRegressor(
                    random_state=SEED,
                    n_jobs=1,
                )
            ],
            "model__n_estimators": [300],
            "model__max_features": ["sqrt", 0.5],
            "model__max_depth": [None, 8, 16],
            "model__min_samples_leaf": [1, 3],
        },
    ]

    if USE_XGBOOST_IF_AVAILABLE and HAS_XGBOOST:
        grids.append({
            "selector__k": k_vals,
            "model": [
                XGBRegressor(
                    objective="reg:squarederror",
                    random_state=SEED,
                    n_jobs=1,
                    tree_method="hist",
                    verbosity=0,
                )
            ],
            "model__n_estimators": [200, 500],
            "model__max_depth": [2, 4],
            "model__learning_rate": [0.03, 0.08],
            "model__subsample": [0.8],
            "model__colsample_bytree": [0.8],
            "model__reg_lambda": [1.0],
        })

    return grids


def make_elasticnet_param_grid(p):
    """Prespecified fixed-algorithm grid for FD sensitivity analysis."""
    return [{
        "selector__k": candidate_k_values(p),
        "model": [ElasticNet(
            max_iter=100000,
            tol=1e-3,
            selection="cyclic",
            random_state=SEED,
        )],
        "model__alpha": [0.05, 0.1, 0.3, 1.0, 3.0, 10.0],
        "model__l1_ratio": [0.1, 0.3, 0.5, 0.7, 0.9],
    }]


def model_name_from_estimator(estimator):
    name = estimator.__class__.__name__
    if name == "ElasticNet":
        return "ElasticNet"
    if name == "RandomForestRegressor":
        return "RandomForest"
    if name == "XGBRegressor":
        return "XGBoost"
    return name


# ============================================================
# 6. METRICS + BIAS CORRECTION
# ============================================================

def regression_metrics(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    return {
        "R2": r2_score(y_true, y_pred),
        "RMSE": math.sqrt(mean_squared_error(y_true, y_pred)),
        "MAE": mean_absolute_error(y_true, y_pred),
        "MeanError": float(np.mean(y_pred - y_true)),
        "SD_Error": float(np.std(y_pred - y_true, ddof=1)),
    }


def fit_brainage_bias_correction(y_train, oof_pred_train):
    """
    Fit:
        predicted_age = intercept + slope * chronological_age
    using OOF predictions from OUTER-TRAINING data only.

    Then:
        corrected_age = (raw_predicted_age - intercept) / slope

    This avoids fitting the correction on the outer-validation subjects.
    """
    y_train = np.asarray(y_train, dtype=float)
    oof_pred_train = np.asarray(oof_pred_train, dtype=float)

    slope, intercept = np.polyfit(y_train, oof_pred_train, 1)

    if (not np.isfinite(slope)) or abs(slope) < 1e-6:
        raise RuntimeError(
            f"Invalid bias-correction slope: {slope}"
        )

    return float(slope), float(intercept)


def apply_brainage_bias_correction(pred, slope, intercept):
    return (np.asarray(pred, dtype=float) - intercept) / slope


# ============================================================
# 7. FEATURE METADATA + SELECTED FEATURE EXTRACTION
# ============================================================

def parse_feature_metadata(feature):
    if feature in NUISANCE_COLUMNS:
        return {
            "Feature": feature,
            "Scale": "Nuisance",
            "Phenotype": "Nuisance",
        }

    phenotype = "Unknown"
    for m in MEASURES:
        if feature.endswith(f"_{m}"):
            phenotype = m
            break

    if feature.startswith("wholebrain_"):
        scale = "WholeBrain"
    elif feature.startswith("lh_hemisphere_") or feature.startswith("rh_hemisphere_"):
        scale = "Hemisphere"
    elif feature.startswith("bilateral_"):
        scale = "Lobe"
    elif feature.startswith("lh_") or feature.startswith("rh_"):
        scale = "ROI"
    else:
        scale = "Unknown"

    return {
        "Feature": feature,
        "Scale": scale,
        "Phenotype": phenotype,
    }


def selected_feature_names(best_estimator, input_columns):
    names = np.asarray(input_columns, dtype=object)

    variance_step = best_estimator.named_steps["variance"]
    names = names[variance_step.get_support()]

    selector_step = best_estimator.named_steps["selector"]
    names = names[selector_step.get_support()]

    return list(names)


# ============================================================
# 8. ONE FEATURE SET: REPEATED NESTED CV
# ============================================================

def run_feature_set_nested_cv(
    df,
    feature_set_name,
    feature_cols,
    param_grid_factory=make_param_grid,
    compute_importance=True,
    analysis_label="Primary",
):
    print("\n" + "=" * 78)
    print(f"FEATURE SET: {feature_set_name}")
    print(f"N features: {len(feature_cols)}")
    print("=" * 78)

    X = df[feature_cols].copy()
    y = df["Age"].to_numpy(dtype=float)
    ids = df["ID"].astype(str).to_numpy()

    fold_rows = []
    pred_rows = []
    selected_rows = []
    importance_rows = []

    for repeat, fold, tr_idx, va_idx in repeated_outer_splits(y):
        print(
            f"[{feature_set_name}] repeat {repeat}/{N_REPEATS} "
            f"| fold {fold}/{OUTER_FOLDS}"
        )

        X_tr = X.iloc[tr_idx].copy()
        X_va = X.iloc[va_idx].copy()
        y_tr = y[tr_idx]
        y_va = y[va_idx]

        inner_cv = make_inner_cv(
            y_tr,
            random_state=SEED + repeat * 100 + fold,
        )

        pipeline = make_base_pipeline()
        param_grid = param_grid_factory(len(feature_cols))

        search = GridSearchCV(
            estimator=pipeline,
            param_grid=param_grid,
            scoring="neg_mean_absolute_error",
            cv=inner_cv,
            n_jobs=-1,
            refit=True,
            return_train_score=False,
            error_score="raise",
        )

        search.fit(X_tr, y_tr)
        best = search.best_estimator_

        # Raw outer-validation Brain Age.
        pred_raw = best.predict(X_va)

        # --------------------------------------------------------
        # Leakage-safe fold-specific age-bias correction.
        #
        # OOF predictions are generated only within outer training.
        # Hyperparameters are already selected using outer-training
        # data only; outer-validation data are never used here.
        # --------------------------------------------------------
        correction_estimator = clone(best)

        oof_train_pred = cross_val_predict(
            correction_estimator,
            X_tr,
            y_tr,
            cv=inner_cv,
            n_jobs=-1,
            method="predict",
        )

        slope, intercept = fit_brainage_bias_correction(
            y_tr,
            oof_train_pred,
        )

        pred_corr = apply_brainage_bias_correction(
            pred_raw,
            slope,
            intercept,
        )

        raw_m = regression_metrics(y_va, pred_raw)
        cor_m = regression_metrics(y_va, pred_corr)

        chosen_model = model_name_from_estimator(
            best.named_steps["model"]
        )

        selected = selected_feature_names(
            best,
            feature_cols,
        )

        fold_rows.append({
            "Analysis": analysis_label,
            "FeatureSet": feature_set_name,
            "Repeat": repeat,
            "Fold": fold,
            "N_Train": len(tr_idx),
            "N_Validation": len(va_idx),
            "Model": chosen_model,
            "InnerBest_MAE": -float(search.best_score_),
            "N_Selected": len(selected),
            "BiasSlope": slope,
            "BiasIntercept": intercept,
            "Raw_R2": raw_m["R2"],
            "Raw_RMSE": raw_m["RMSE"],
            "Raw_MAE": raw_m["MAE"],
            "Corrected_R2": cor_m["R2"],
            "Corrected_RMSE": cor_m["RMSE"],
            "Corrected_MAE": cor_m["MAE"],
            "Corrected_MeanError": cor_m["MeanError"],
            "BestParamsJSON": json.dumps(
                {
                    k: (
                        model_name_from_estimator(v)
                        if k == "model"
                        else v
                    )
                    for k, v in search.best_params_.items()
                },
                default=str,
                sort_keys=True,
            ),
        })

        for subject_id, actual, raw, corr in zip(
            ids[va_idx],
            y_va,
            pred_raw,
            pred_corr,
        ):
            pred_rows.append({
                "Analysis": analysis_label,
                "FeatureSet": feature_set_name,
                "Repeat": repeat,
                "Fold": fold,
                "ID": subject_id,
                "ActualAge": actual,
                "RawBrainAge": raw,
                "CorrectedBrainAge": corr,
                "RawBAG": raw - actual,
                "CorrectedBAG": corr - actual,
                "Model": chosen_model,
            })

        selected_set = set(selected)
        for c in feature_cols:
            meta = parse_feature_metadata(c)
            selected_rows.append({
                "Analysis": analysis_label,
                "FeatureSet": feature_set_name,
                "Repeat": repeat,
                "Fold": fold,
                "Model": chosen_model,
                "Feature": c,
                "Selected": int(c in selected_set),
                "Scale": meta["Scale"],
                "Phenotype": meta["Phenotype"],
            })

        # --------------------------------------------------------
        # Out-of-sample permutation importance on untouched outer
        # validation data. This is primary explainability evidence.
        # --------------------------------------------------------
        if compute_importance and feature_set_name in PERM_IMPORTANCE_FEATURESETS:
            perm = permutation_importance(
                best,
                X_va,
                y_va,
                scoring="neg_mean_absolute_error",
                n_repeats=PERM_IMPORTANCE_REPEATS,
                random_state=SEED + repeat * 1000 + fold,
                n_jobs=-1,
            )

            for c, imp_mean, imp_sd in zip(
                feature_cols,
                perm.importances_mean,
                perm.importances_std,
            ):
                meta = parse_feature_metadata(c)
                importance_rows.append({
                    "Analysis": analysis_label,
                    "FeatureSet": feature_set_name,
                    "Repeat": repeat,
                    "Fold": fold,
                    "Model": chosen_model,
                    "Feature": c,
                    "PermutationImportance_MAE": float(imp_mean),
                    "PermutationImportance_SD": float(imp_sd),
                    "Scale": meta["Scale"],
                    "Phenotype": meta["Phenotype"],
                })

    return (
        pd.DataFrame(fold_rows),
        pd.DataFrame(pred_rows),
        pd.DataFrame(selected_rows),
        pd.DataFrame(importance_rows),
    )


# ============================================================
# 9. AGGREGATE REPEATED OOF PREDICTIONS
# ============================================================

def aggregate_subject_predictions(pred_df):
    # Each subject is outer-validation exactly once per repeat.
    agg = (
        pred_df
        .groupby(["FeatureSet", "ID"], as_index=False)
        .agg(
            ActualAge=("ActualAge", "first"),
            RawBrainAge=("RawBrainAge", "mean"),
            CorrectedBrainAge=("CorrectedBrainAge", "mean"),
            RawBrainAge_SD=("RawBrainAge", "std"),
            CorrectedBrainAge_SD=("CorrectedBrainAge", "std"),
        )
    )

    agg["RawBAG"] = agg["RawBrainAge"] - agg["ActualAge"]
    agg["CorrectedBAG"] = (
        agg["CorrectedBrainAge"] - agg["ActualAge"]
    )

    return agg


def bootstrap_metrics_subject_level(
    actual,
    predicted,
    n_boot=N_BOOTSTRAP,
    seed=SEED,
):
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)

    rng = np.random.default_rng(seed)
    n = len(actual)

    rows = []
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        yt = actual[idx]
        yp = predicted[idx]

        # R2 is undefined in the extremely unlikely event of zero variance.
        if np.std(yt) == 0:
            continue

        m = regression_metrics(yt, yp)
        rows.append([m["R2"], m["RMSE"], m["MAE"]])

    arr = np.asarray(rows, dtype=float)

    return {
        "R2_L95": np.percentile(arr[:, 0], 2.5),
        "R2_U95": np.percentile(arr[:, 0], 97.5),
        "RMSE_L95": np.percentile(arr[:, 1], 2.5),
        "RMSE_U95": np.percentile(arr[:, 1], 97.5),
        "MAE_L95": np.percentile(arr[:, 2], 2.5),
        "MAE_U95": np.percentile(arr[:, 2], 97.5),
        "BootstrapSuccessN": len(arr),
    }


def performance_summary_from_aggregated(agg_df):
    """
    PRIMARY predictive performance = RAW repeated-OOF Brain Age.
    Bias-corrected Brain Age/BAG is retained as a SECONDARY age-bias analysis.
    """
    rows = []

    for fs, g in agg_df.groupby("FeatureSet"):
        raw = regression_metrics(g["ActualAge"], g["RawBrainAge"])
        corr = regression_metrics(g["ActualAge"], g["CorrectedBrainAge"])

        raw_ci = bootstrap_metrics_subject_level(
            g["ActualAge"].to_numpy(),
            g["RawBrainAge"].to_numpy(),
            seed=SEED + sum(ord(ch) for ch in fs + "raw") % 100000,
        )
        corr_ci = bootstrap_metrics_subject_level(
            g["ActualAge"].to_numpy(),
            g["CorrectedBrainAge"].to_numpy(),
            seed=SEED + sum(ord(ch) for ch in fs + "corr") % 100000,
        )

        raw_corr_bag_age = np.corrcoef(g["ActualAge"], g["RawBAG"])[0, 1]
        corrected_corr_bag_age = np.corrcoef(g["ActualAge"], g["CorrectedBAG"])[0, 1]

        rows.append({
            "FeatureSet": fs,
            "N": len(g),
            "N_Features": len(FEATURE_SETS[fs]),
            # PRIMARY performance
            "Raw_R2": raw["R2"],
            "Raw_RMSE": raw["RMSE"],
            "Raw_MAE": raw["MAE"],
            "Raw_R2_L95": raw_ci["R2_L95"],
            "Raw_R2_U95": raw_ci["R2_U95"],
            "Raw_RMSE_L95": raw_ci["RMSE_L95"],
            "Raw_RMSE_U95": raw_ci["RMSE_U95"],
            "Raw_MAE_L95": raw_ci["MAE_L95"],
            "Raw_MAE_U95": raw_ci["MAE_U95"],
            # SECONDARY age-bias corrected outputs
            "Corrected_R2": corr["R2"],
            "Corrected_RMSE": corr["RMSE"],
            "Corrected_MAE": corr["MAE"],
            "Corrected_R2_L95": corr_ci["R2_L95"],
            "Corrected_R2_U95": corr_ci["R2_U95"],
            "Corrected_RMSE_L95": corr_ci["RMSE_L95"],
            "Corrected_RMSE_U95": corr_ci["RMSE_U95"],
            "Corrected_MAE_L95": corr_ci["MAE_L95"],
            "Corrected_MAE_U95": corr_ci["MAE_U95"],
            "Raw_MeanBAG": float(g["RawBAG"].mean()),
            "Corrected_MeanBAG": float(g["CorrectedBAG"].mean()),
            "Raw_corr_BAG_Age": raw_corr_bag_age,
            "Corrected_corr_BAG_Age": corrected_corr_bag_age,
        })

    return pd.DataFrame(rows).sort_values(
        ["Raw_MAE", "Raw_RMSE", "Raw_R2"],
        ascending=[True, True, False],
    )


def repeat_level_performance(predictions_long):
    """One complete OOF performance estimate per repeat and its summary."""
    detail_rows = []
    for (analysis, fs, repeat), g in predictions_long.groupby(
        ["Analysis", "FeatureSet", "Repeat"]
    ):
        raw = regression_metrics(g["ActualAge"], g["RawBrainAge"])
        corr = regression_metrics(g["ActualAge"], g["CorrectedBrainAge"])
        detail_rows.append({
            "Analysis": analysis, "FeatureSet": fs, "Repeat": repeat, "N": len(g),
            "Raw_MAE": raw["MAE"], "Raw_RMSE": raw["RMSE"], "Raw_R2": raw["R2"],
            "Raw_CalibrationSlope": float(np.polyfit(g["ActualAge"], g["RawBrainAge"], 1)[0]),
            "Raw_corr_BAG_Age": float(np.corrcoef(g["RawBAG"], g["ActualAge"])[0, 1]),
            "Corrected_MAE": corr["MAE"], "Corrected_RMSE": corr["RMSE"],
            "Corrected_R2": corr["R2"],
            "Corrected_CalibrationSlope": float(np.polyfit(g["ActualAge"], g["CorrectedBrainAge"], 1)[0]),
            "Corrected_corr_BAG_Age": float(np.corrcoef(g["CorrectedBAG"], g["ActualAge"])[0, 1]),
        })
    detail = pd.DataFrame(detail_rows)
    if detail.empty:
        return detail, pd.DataFrame()
    metric_cols = [c for c in detail.columns if c not in {"Analysis", "FeatureSet", "Repeat", "N"}]
    summary_rows = []
    for (analysis, fs), g in detail.groupby(["Analysis", "FeatureSet"]):
        row = {
            "Analysis": analysis, "FeatureSet": fs, "N_Repeats": len(g),
            "PrimaryReportingNote": (
                "Mean and SD across complete OOF repeats; participant-averaged "
                "OOF metrics are secondary ensemble-style summaries."
            ),
        }
        for c in metric_cols:
            row[f"{c}_MeanAcrossRepeats"] = float(g[c].mean())
            row[f"{c}_SDAcrossRepeats"] = float(g[c].std(ddof=1)) if len(g) > 1 else np.nan
            row[f"{c}_Min"] = float(g[c].min())
            row[f"{c}_Max"] = float(g[c].max())
        summary_rows.append(row)
    return detail, pd.DataFrame(summary_rows)


# ============================================================
# 10. PAIRED INCREMENTAL-VALUE TEST FOR FD
# ============================================================

def bh_fdr(pvals):
    pvals = np.asarray(pvals, dtype=float)
    n = len(pvals)
    order = np.argsort(pvals)
    ranked = pvals[order]

    q = ranked * n / np.arange(1, n + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    q = np.clip(q, 0, 1)

    out = np.empty(n)
    out[order] = q
    return out


def _paired_delta_from_repeated_oof(merged, pred_a, pred_b, rng):
    """Two-sided paired test; positive delta means model A has lower MAE."""
    err_a = np.abs(merged[pred_a] - merged["ActualAge"])
    err_b = np.abs(merged[pred_b] - merged["ActualAge"])
    repeated_d = np.asarray(err_b - err_a, dtype=float)
    d = (
        pd.DataFrame({"ID": merged["ID"].astype(str), "d": repeated_d})
        .groupby("ID", as_index=False)["d"].mean()["d"].to_numpy()
    )
    delta = float(np.mean(d))
    n = len(d)

    boot = np.empty(N_BOOTSTRAP, dtype=float)
    for b in range(N_BOOTSTRAP):
        idx = rng.integers(0, n, n)
        boot[b] = np.mean(d[idx])
    ci_l, ci_u = np.percentile(boot, [2.5, 97.5])

    perm_stats = np.empty(N_PAIRED_PERM, dtype=float)
    for i in range(N_PAIRED_PERM):
        signs = rng.choice([-1.0, 1.0], size=n)
        perm_stats[i] = np.mean(d * signs)
    p = (1 + np.sum(np.abs(perm_stats) >= abs(delta))) / (N_PAIRED_PERM + 1)

    return {
        "N": n,
        "MAE_A": float(np.mean(err_a)),
        "MAE_B": float(np.mean(err_b)),
        "DeltaMAE_B_minus_A": delta,
        "DeltaMAE_L95": float(ci_l),
        "DeltaMAE_U95": float(ci_u),
        "p_two_sided": float(p),
    }


def paired_workflow_tests(predictions_long, pairs, family_label):
    """Generic paired workflow comparisons with subject as inference unit."""
    rows = []
    available = set(predictions_long["FeatureSet"].unique())
    rng = np.random.default_rng(SEED + sum(ord(x) for x in family_label))
    for label, fs_a, fs_b in pairs:
        if fs_a not in available or fs_b not in available:
            continue
        a = predictions_long.loc[predictions_long["FeatureSet"] == fs_a,
            ["ID", "Repeat", "ActualAge", "RawBrainAge", "CorrectedBrainAge"]
        ].rename(columns={"RawBrainAge": "Raw_A", "CorrectedBrainAge": "Corr_A"})
        b = predictions_long.loc[predictions_long["FeatureSet"] == fs_b,
            ["ID", "Repeat", "RawBrainAge", "CorrectedBrainAge"]
        ].rename(columns={"RawBrainAge": "Raw_B", "CorrectedBrainAge": "Corr_B"})
        m = a.merge(b, on=["ID", "Repeat"], how="inner", validate="one_to_one")
        raw = _paired_delta_from_repeated_oof(m, "Raw_A", "Raw_B", rng)
        corr = _paired_delta_from_repeated_oof(m, "Corr_A", "Corr_B", rng)
        rows.append({
            "Family": family_label, "Comparison": label,
            "Model_A": fs_a, "Model_B": fs_b,
            "N_Subjects": raw["N"], "N_RepeatedPairs": len(m),
            "Raw_MAE_A": raw["MAE_A"], "Raw_MAE_B": raw["MAE_B"],
            "Raw_DeltaMAE_B_minus_A": raw["DeltaMAE_B_minus_A"],
            "Raw_DeltaMAE_L95": raw["DeltaMAE_L95"],
            "Raw_DeltaMAE_U95": raw["DeltaMAE_U95"],
            "Raw_Permutation_p_two_sided": raw["p_two_sided"],
            "Corrected_MAE_A": corr["MAE_A"], "Corrected_MAE_B": corr["MAE_B"],
            "Corrected_DeltaMAE_B_minus_A": corr["DeltaMAE_B_minus_A"],
            "Corrected_DeltaMAE_L95": corr["DeltaMAE_L95"],
            "Corrected_DeltaMAE_U95": corr["DeltaMAE_U95"],
            "Corrected_Permutation_p_two_sided": corr["p_two_sided"],
            "InferenceUnit": "Participant mean paired error difference across repeats",
        })
    out = pd.DataFrame(rows)
    if not out.empty:
        out["Raw_Permutation_q_BH"] = bh_fdr(
            out["Raw_Permutation_p_two_sided"].to_numpy()
        )
        out["Corrected_Permutation_q_BH"] = bh_fdr(
            out["Corrected_Permutation_p_two_sided"].to_numpy()
        )
    return out


def paired_fd_incremental_tests(predictions_long):
    scales = ["WholeBrain", "Hemisphere", "Lobe", "ROI", "Multiscale"]
    pairs = [(s, f"{s}_All5", f"{s}_NoFD") for s in scales]
    out = paired_workflow_tests(predictions_long, pairs, "FD_incremental")
    if not out.empty:
        out = out.rename(columns={
            "Comparison": "Scale",
            "Raw_DeltaMAE_B_minus_A": "Raw_DeltaMAE_NoFD_minus_All5",
            "Corrected_DeltaMAE_B_minus_A": "Corrected_DeltaMAE_NoFD_minus_All5",
        })
    return out


# ============================================================
# 11. SELECTION STABILITY + IMPORTANCE SUMMARIES
# ============================================================

def selection_stability_summary(selected_df):
    if selected_df.empty:
        return pd.DataFrame()

    out = (
        selected_df
        .groupby(
            [
                "FeatureSet",
                "Feature",
                "Scale",
                "Phenotype",
            ],
            as_index=False,
        )
        .agg(
            SelectionFrequency=("Selected", "mean"),
            NOuterFits=("Selected", "size"),
        )
    )

    out["SelectionFrequencyPct"] = (
        100 * out["SelectionFrequency"]
    )

    return out.sort_values(
        ["FeatureSet", "SelectionFrequencyPct"],
        ascending=[True, False],
    )


def permutation_importance_summary(importance_df):
    if importance_df.empty:
        return pd.DataFrame(), pd.DataFrame()

    feature_summary = (
        importance_df
        .groupby(
            [
                "FeatureSet",
                "Feature",
                "Scale",
                "Phenotype",
            ],
            as_index=False,
        )
        .agg(
            MeanPermutationImportance_MAE=(
                "PermutationImportance_MAE",
                "mean",
            ),
            SDPermutationImportance_MAE=(
                "PermutationImportance_MAE",
                "std",
            ),
            NOuterFits=("PermutationImportance_MAE", "size"),
        )
        .sort_values(
            ["FeatureSet", "MeanPermutationImportance_MAE"],
            ascending=[True, False],
        )
    )

    phenotype_scale_summary = (
        feature_summary
        .groupby(
            ["FeatureSet", "Scale", "Phenotype"],
            as_index=False,
        )
        .agg(
            TotalMeanPermutationImportance_MAE=(
                "MeanPermutationImportance_MAE",
                "sum",
            ),
            MeanFeatureImportance_MAE=(
                "MeanPermutationImportance_MAE",
                "mean",
            ),
            NFeatures=("Feature", "size"),
        )
    )

    return feature_summary, phenotype_scale_summary


# ============================================================
# 12. FINAL DESCRIPTIVE MODEL + SHAP
# ============================================================

def transform_for_model(best_pipeline, X, feature_cols):
    z = best_pipeline.named_steps["imputer"].transform(X)
    z = best_pipeline.named_steps["variance"].transform(z)
    z = best_pipeline.named_steps["scaler"].transform(z)
    z = best_pipeline.named_steps["selector"].transform(z)

    names = selected_feature_names(
        best_pipeline,
        feature_cols,
    )
    return z, names


def fit_final_descriptive_model(df, feature_set_name, feature_cols):
    """
    This model is fitted on the full dataset ONLY AFTER repeated nested-CV
    performance estimation is complete. It is for interpretation/deployment,
    not for unbiased performance estimation.
    """
    X = df[feature_cols].copy()
    y = df["Age"].to_numpy(dtype=float)

    inner_cv = make_inner_cv(
        y,
        random_state=SEED + 99999,
    )

    search = GridSearchCV(
        estimator=make_base_pipeline(),
        param_grid=make_param_grid(len(feature_cols)),
        scoring="neg_mean_absolute_error",
        cv=inner_cv,
        n_jobs=-1,
        refit=True,
        error_score="raise",
    )

    search.fit(X, y)
    best = search.best_estimator_

    joblib.dump(
        {
            "feature_set": feature_set_name,
            "feature_columns": feature_cols,
            "best_estimator": best,
            "best_params": search.best_params_,
        },
        os.path.join(
            OUTPUT_DIR,
            "Final_Descriptive_BrainAge_Model.joblib",
        ),
    )

    return best, search


def make_final_shap_outputs(
    df,
    feature_set_name,
    feature_cols,
    best_pipeline,
    subject_oof_predictions=None,
):
    """
    Descriptive explainability for the final full-data model.

    IMPORTANT:
    - This function is NOT used for unbiased performance estimation.
    - Primary inferential explainability remains outer-validation permutation
      importance and outer-fold feature-selection stability.
    - For ElasticNet, standardized coefficients + Linear SHAP are exported.
    - For tree models, Tree SHAP is used.
    - Optional individual waterfall plots are supplementary only.
    """
    model = best_pipeline.named_steps["model"]
    model_name = model_name_from_estimator(model)

    z, names = transform_for_model(
        best_pipeline,
        df[feature_cols],
        feature_cols,
    )
    z_df = pd.DataFrame(z, columns=names, index=df.index)

    # --------------------------------------------------------
    # A) Standardized coefficients for ElasticNet
    # --------------------------------------------------------
    coefficient_table = pd.DataFrame()
    if model_name == "ElasticNet":
        coef = np.asarray(model.coef_, dtype=float)
        coefficient_table = pd.DataFrame({
            "Feature": names,
            "Coefficient": coef,
            "AbsCoefficient": np.abs(coef),
        })
        meta = pd.DataFrame([parse_feature_metadata(c) for c in names])
        coefficient_table = coefficient_table.merge(meta, on="Feature", how="left")
        coefficient_table = coefficient_table.sort_values(
            "AbsCoefficient", ascending=False
        )

        coefficient_table.to_excel(
            os.path.join(
                OUTPUT_DIR,
                "Final_ElasticNet_Standardized_Coefficients.xlsx",
            ),
            index=False,
        )

        top = coefficient_table.head(SHAP_MAX_DISPLAY).sort_values("Coefficient")
        plt.figure(figsize=(9, 8))
        plt.barh(top["Feature"], top["Coefficient"])
        plt.axvline(0, color="black", linewidth=0.8)
        plt.xlabel("Standardized Elastic Net coefficient")
        plt.ylabel("Feature")
        plt.title(
            f"Largest standardized Elastic Net coefficients — {feature_set_name}\n"
            "Full-data descriptive model; not used for OOF performance inference"
        )
        plt.tight_layout()
        plt.savefig(
            os.path.join(
                OUTPUT_DIR,
                "Figure_Final_ElasticNet_TopCoefficients.png",
            ),
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()

    if not RUN_FINAL_SHAP:
        return coefficient_table

    if not HAS_SHAP:
        print(
            "SHAP is not installed. Coefficients were exported when applicable; "
            "SHAP plots were skipped."
        )
        return coefficient_table

    # --------------------------------------------------------
    # B) SHAP values
    # --------------------------------------------------------
    try:
        if model_name == "ElasticNet":
            print(
                "Final selected model is ElasticNet; using LinearExplainer "
                "for descriptive SHAP outputs."
            )
            explainer = shap.LinearExplainer(model, z_df)
            shap_values = explainer(z_df)
        elif model_name in {"RandomForest", "XGBoost"}:
            explainer = shap.TreeExplainer(model)
            shap_values = explainer(z_df)
        else:
            print(
                f"SHAP skipped for unsupported final model: {model_name}."
            )
            return coefficient_table
    except Exception as e:
        print(f"SHAP explainer failed safely: {type(e).__name__}: {e}")
        return coefficient_table

    values = np.asarray(shap_values.values)
    if values.ndim != 2:
        print(
            f"Unexpected SHAP value shape {values.shape}; SHAP figures skipped safely."
        )
        return coefficient_table

    mean_abs = np.abs(values).mean(axis=0)
    shap_summary = pd.DataFrame({
        "Feature": names,
        "MeanAbsSHAP": mean_abs,
    })
    meta = pd.DataFrame([parse_feature_metadata(c) for c in names])
    shap_summary = shap_summary.merge(meta, on="Feature", how="left")
    shap_summary = shap_summary.sort_values("MeanAbsSHAP", ascending=False)

    shap_summary.to_excel(
        os.path.join(
            OUTPUT_DIR,
            "Final_Descriptive_SHAP_Importance.xlsx",
        ),
        index=False,
    )

    # SHAP bar
    try:
        plt.figure(figsize=(9, 8))
        shap.plots.bar(
            shap_values,
            max_display=SHAP_MAX_DISPLAY,
            show=False,
        )
        plt.title(
            f"Descriptive SHAP importance — {feature_set_name}\n"
            "Full-data final model; not used for performance inference"
        )
        plt.tight_layout()
        plt.savefig(
            os.path.join(
                OUTPUT_DIR,
                "Figure_Final_SHAP_Bar.png",
            ),
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()
    except Exception as e:
        print(f"SHAP bar plot failed safely: {type(e).__name__}: {e}")
        plt.close("all")

    # SHAP beeswarm
    try:
        plt.figure(figsize=(10, 9))
        shap.plots.beeswarm(
            shap_values,
            max_display=SHAP_MAX_DISPLAY,
            show=False,
        )
        plt.title(
            f"Descriptive SHAP beeswarm — {feature_set_name}\n"
            "Full-data final model; standardized model inputs"
        )
        plt.tight_layout()
        plt.savefig(
            os.path.join(
                OUTPUT_DIR,
                "Figure_Final_SHAP_Beeswarm.png",
            ),
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()
    except Exception as e:
        print(f"SHAP beeswarm failed safely: {type(e).__name__}: {e}")
        plt.close("all")

    # --------------------------------------------------------
    # C) Supplementary waterfall plots for 3 pre-defined OOF BAG profiles
    #    Selection uses OOF BAG, whereas explanations use the final full-data
    #    descriptive model. These plots are illustrative only.
    # --------------------------------------------------------
    if subject_oof_predictions is not None and "ID" in df.columns:
        try:
            oof = subject_oof_predictions.loc[
                subject_oof_predictions["FeatureSet"] == feature_set_name
            ].copy()
            if not oof.empty:
                # One near-zero BAG, one positive BAG, one negative BAG.
                near0_id = oof.loc[oof["RawBAG"].abs().idxmin(), "ID"]
                pos_id = oof.loc[oof["RawBAG"].idxmax(), "ID"]
                neg_id = oof.loc[oof["RawBAG"].idxmin(), "ID"]
                reps = [
                    ("NearZeroBAG", near0_id),
                    ("PositiveBAG", pos_id),
                    ("NegativeBAG", neg_id),
                ]

                id_to_pos = {
                    str(v): i for i, v in enumerate(df["ID"].astype(str).tolist())
                }
                for label, sid in reps:
                    pos = id_to_pos.get(str(sid))
                    if pos is None:
                        continue
                    plt.figure(figsize=(10, 7))
                    shap.plots.waterfall(
                        shap_values[pos],
                        max_display=12,
                        show=False,
                    )
                    bag_value = float(
                        oof.loc[oof["ID"].astype(str) == str(sid), "RawBAG"].iloc[0]
                    )
                    plt.title(
                        f"Supplementary descriptive SHAP waterfall — {label}\n"
                        f"ID={sid}; OOF raw BAG={bag_value:.2f} years"
                    )
                    plt.tight_layout()
                    plt.savefig(
                        os.path.join(
                            OUTPUT_DIR,
                            f"Supplementary_SHAP_Waterfall_{label}.png",
                        ),
                        dpi=300,
                        bbox_inches="tight",
                    )
                    plt.close()
        except Exception as e:
            print(f"Supplementary SHAP waterfall failed safely: {type(e).__name__}: {e}")
            plt.close("all")

    # Return a unified explainability table when possible.
    if model_name == "ElasticNet" and not coefficient_table.empty:
        merged = coefficient_table.merge(
            shap_summary[["Feature", "MeanAbsSHAP"]],
            on="Feature",
            how="left",
        )
        return merged.sort_values("AbsCoefficient", ascending=False)

    return shap_summary


# ============================================================
# 13. FIGURES
# ============================================================

def plot_scale_benchmark(perf):
    order = [
        "WholeBrain_All5",
        "Hemisphere_All5",
        "Lobe_All5",
        "ROI_All5",
        "Multiscale_All5",
    ]

    z = perf.loc[perf["FeatureSet"].isin(order)].copy()
    if z.empty:
        return

    z["FeatureSet"] = pd.Categorical(
        z["FeatureSet"], categories=order, ordered=True
    )
    z = z.sort_values("FeatureSet")

    # MAIN: dot-whisker plot with subject-level bootstrap 95% CI.
    y = np.arange(len(z))
    x = z["Raw_MAE"].to_numpy(dtype=float)
    lo = z["Raw_MAE_L95"].to_numpy(dtype=float)
    hi = z["Raw_MAE_U95"].to_numpy(dtype=float)
    xerr = np.vstack([x - lo, hi - x])

    plt.figure(figsize=(9, 5.8))
    plt.errorbar(x, y, xerr=xerr, fmt="o", capsize=4)
    plt.yticks(y, z["FeatureSet"].astype(str))
    plt.xlabel("Raw repeated-OOF Brain-Age MAE (years)")
    plt.ylabel("Anatomical representation")
    plt.title("Brain-Age prediction across anatomical scales\n95% subject-bootstrap CI")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(
        os.path.join(OUTPUT_DIR, "Figure_SpatialScale_Benchmark_MAE.png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

    # Supplementary familiar bar version.
    plt.figure(figsize=(10, 6))
    plt.bar(z["FeatureSet"].astype(str), z["Raw_MAE"])
    plt.ylabel("Raw repeated-OOF Brain-Age MAE (years)")
    plt.xlabel("Spatial scale")
    plt.xticks(rotation=30, ha="right")
    plt.title("Supplementary spatial-scale benchmark")
    plt.tight_layout()
    plt.savefig(
        os.path.join(OUTPUT_DIR, "Supplementary_SpatialScale_Benchmark_Bar.png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()


def plot_roi_phenotype(perf):
    order = [
        "ROI_volume",
        "ROI_area",
        "ROI_thickness",
        "ROI_curvature",
        "ROI_FD",
        "ROI_NoFD",
        "ROI_All5",
    ]

    z = perf.loc[perf["FeatureSet"].isin(order)].copy()
    if z.empty:
        return

    z["FeatureSet"] = pd.Categorical(
        z["FeatureSet"], categories=order, ordered=True
    )
    z = z.sort_values("FeatureSet")

    y = np.arange(len(z))
    x = z["Raw_MAE"].to_numpy(dtype=float)
    lo = z["Raw_MAE_L95"].to_numpy(dtype=float)
    hi = z["Raw_MAE_U95"].to_numpy(dtype=float)
    xerr = np.vstack([x - lo, hi - x])

    plt.figure(figsize=(9, 6.2))
    plt.errorbar(x, y, xerr=xerr, fmt="o", capsize=4)
    plt.yticks(y, z["FeatureSet"].astype(str))
    plt.xlabel("Raw repeated-OOF Brain-Age MAE (years)")
    plt.ylabel("ROI morphometric feature set")
    plt.title("ROI phenotype ablation\n95% subject-bootstrap CI")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(
        os.path.join(OUTPUT_DIR, "Figure_ROI_Phenotype_Ablation_MAE.png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()


def plot_fd_incremental(fd_test):
    if fd_test.empty:
        return

    order = ["WholeBrain", "Hemisphere", "Lobe", "ROI", "Multiscale"]
    z = fd_test.copy()
    z["Scale"] = pd.Categorical(z["Scale"], categories=order, ordered=True)
    z = z.sort_values("Scale")

    x = z["Raw_DeltaMAE_NoFD_minus_All5"].to_numpy(dtype=float)
    lo = z["Raw_DeltaMAE_L95"].to_numpy(dtype=float)
    hi = z["Raw_DeltaMAE_U95"].to_numpy(dtype=float)
    xerr = np.vstack([x - lo, hi - x])
    y = np.arange(len(z))

    # MAIN: forest-style CI plot. This preserves the key distinction between
    # FDR significance and whether the bootstrap CI crosses zero.
    plt.figure(figsize=(9, 5.8))
    plt.errorbar(x, y, xerr=xerr, fmt="o", capsize=4)
    plt.axvline(0, color="black", linestyle="--", linewidth=1)
    plt.yticks(y, z["Scale"].astype(str))
    plt.xlabel("Raw ΔMAE = No-FD MAE − All-5 MAE (years)")
    plt.ylabel("Anatomical scale")
    plt.title(
        "Incremental predictive value of fractal dimension\n"
        "Positive values favor adding FD; bars are 95% paired-bootstrap CI"
    )
    plt.gca().invert_yaxis()

    # q-value labels, without turning them into visual effect-size claims.
    xmax = max(hi.max(), 0)
    span = max(hi.max() - min(lo.min(), 0), 0.5)
    for yi, (_, row) in enumerate(z.iterrows()):
        q = float(row.get("Raw_Permutation_q_BH", np.nan))
        label = f"q={q:.3f}" if np.isfinite(q) else "q=NA"
        plt.text(hi[yi] + 0.03 * span, yi, label, va="center", fontsize=9)

    plt.tight_layout()
    plt.savefig(
        os.path.join(OUTPUT_DIR, "Figure_FD_Incremental_Value.png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.savefig(
        os.path.join(OUTPUT_DIR, "Figure_FD_Incremental_Value_Forest.png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

    # Supplementary bar version retained for completeness, but not preferred
    # for the manuscript because it hides CI overlap with zero.
    plt.figure(figsize=(9, 5.5))
    plt.bar(z["Scale"].astype(str), z["Raw_DeltaMAE_NoFD_minus_All5"])
    plt.axhline(0, color="black", linewidth=0.8)
    plt.ylabel("Raw ΔMAE (years)")
    plt.xlabel("Spatial scale")
    plt.title("Supplementary incremental predictive value of FD")
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    plt.savefig(
        os.path.join(OUTPUT_DIR, "Supplementary_FD_Incremental_Value_Bar.png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()


def plot_best_brainage(agg_df, best_fs):
    g = agg_df.loc[agg_df["FeatureSet"] == best_fs].copy()
    if g.empty:
        return

    # 1) PRIMARY raw OOF scatter
    m_raw = regression_metrics(g["ActualAge"], g["RawBrainAge"])
    plt.figure(figsize=(7, 6))
    sns.scatterplot(data=g, x="ActualAge", y="RawBrainAge", alpha=0.75)
    lo = min(g["ActualAge"].min(), g["RawBrainAge"].min())
    hi = max(g["ActualAge"].max(), g["RawBrainAge"].max())
    plt.plot([lo, hi], [lo, hi], "--", color="black")
    plt.title(
        f"Raw repeated nested-CV Brain Age — {best_fs}\n"
        f"R²={m_raw['R2']:.3f}, MAE={m_raw['MAE']:.2f}, RMSE={m_raw['RMSE']:.2f}"
    )
    plt.xlabel("Chronological age (years)")
    plt.ylabel("Raw predicted Brain Age (years)")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "Figure_BestModel_Raw_BrainAge_Scatter.png"), dpi=300, bbox_inches="tight")
    # Also overwrite the legacy filename so an older corrected/stale figure
    # cannot be mistaken for the current primary figure.
    plt.savefig(os.path.join(OUTPUT_DIR, "Figure_BestModel_BrainAge_Scatter.png"), dpi=300, bbox_inches="tight")
    plt.close()

    # 2) Corrected scatter — secondary
    m_cor = regression_metrics(g["ActualAge"], g["CorrectedBrainAge"])
    plt.figure(figsize=(7, 6))
    sns.scatterplot(data=g, x="ActualAge", y="CorrectedBrainAge", alpha=0.75)
    lo = min(g["ActualAge"].min(), g["CorrectedBrainAge"].min())
    hi = max(g["ActualAge"].max(), g["CorrectedBrainAge"].max())
    plt.plot([lo, hi], [lo, hi], "--", color="black")
    plt.title(
        f"Bias-corrected Brain Age — {best_fs}\n"
        f"R²={m_cor['R2']:.3f}, MAE={m_cor['MAE']:.2f}"
    )
    plt.xlabel("Chronological age (years)")
    plt.ylabel("Corrected Brain Age (years)")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "Figure_BestModel_Corrected_BrainAge_Scatter.png"), dpi=300, bbox_inches="tight")
    plt.close()

    # 3) Raw BAG vs age
    plt.figure(figsize=(7, 5.5))
    sns.regplot(data=g, x="ActualAge", y="RawBAG", scatter_kws={"alpha": 0.65}, line_kws={"color": "black"})
    plt.axhline(0, linestyle="--", color="gray")
    r = np.corrcoef(g["ActualAge"], g["RawBAG"])[0, 1]
    plt.title(f"Raw Brain-Age Gap vs age — {best_fs}\nr(BAG, age)={r:.3f}")
    plt.xlabel("Chronological age (years)")
    plt.ylabel("Raw Brain-Age Gap (years)")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "Figure_BestModel_RawBAG_vs_Age.png"), dpi=300, bbox_inches="tight")
    plt.close()

    # 4) Corrected BAG vs age
    plt.figure(figsize=(7, 5.5))
    sns.regplot(data=g, x="ActualAge", y="CorrectedBAG", scatter_kws={"alpha": 0.65}, line_kws={"color": "black"})
    plt.axhline(0, linestyle="--", color="gray")
    r = np.corrcoef(g["ActualAge"], g["CorrectedBAG"])[0, 1]
    plt.title(f"Residual age bias after correction — {best_fs}\nr(BAG, age)={r:.3f}")
    plt.xlabel("Chronological age (years)")
    plt.ylabel("Corrected Brain-Age Gap (years)")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "Figure_BestModel_CorrectedBAG_vs_Age.png"), dpi=300, bbox_inches="tight")
    plt.close()

    # 5) Residual plot (raw prediction error)
    plt.figure(figsize=(7, 5.5))
    sns.regplot(data=g, x="ActualAge", y="RawBAG", scatter_kws={"alpha": 0.65}, line_kws={"color": "black"})
    plt.axhline(0, linestyle="--", color="gray")
    plt.xlabel("Chronological age (years)")
    plt.ylabel("Raw prediction error (Predicted − Actual, years)")
    plt.title(f"Raw OOF residual diagnostic — {best_fs}")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "Figure_BestModel_Raw_Residuals.png"), dpi=300, bbox_inches="tight")
    plt.close()

    # 6) Calibration plot
    plt.figure(figsize=(7, 5.5))
    sns.regplot(data=g, x="ActualAge", y="RawBrainAge", scatter_kws={"alpha": 0.65}, line_kws={"color": "black"})
    lo, hi = g["ActualAge"].min(), g["ActualAge"].max()
    plt.plot([lo, hi], [lo, hi], "--", color="gray")
    plt.xlabel("Chronological age (years)")
    plt.ylabel("Raw predicted Brain Age (years)")
    plt.title(f"Raw OOF calibration — {best_fs}")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "Figure_BestModel_Raw_Calibration.png"), dpi=300, bbox_inches="tight")
    plt.close()

    # 7) Bland–Altman diagnostic
    mean_age = (g["ActualAge"] + g["RawBrainAge"]) / 2
    diff = g["RawBrainAge"] - g["ActualAge"]
    md = float(diff.mean())
    sd = float(diff.std(ddof=1))
    plt.figure(figsize=(7, 5.5))
    sns.scatterplot(x=mean_age, y=diff, alpha=0.7)
    plt.axhline(md, linestyle="--", color="black")
    plt.axhline(md + 1.96 * sd, linestyle="--", color="gray")
    plt.axhline(md - 1.96 * sd, linestyle="--", color="gray")
    plt.xlabel("Mean of chronological and predicted age (years)")
    plt.ylabel("Prediction error (years)")
    plt.title(f"Bland–Altman diagnostic — {best_fs}")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "Figure_BestModel_BlandAltman.png"), dpi=300, bbox_inches="tight")
    plt.close()

    # 8) Prediction-error distribution
    plt.figure(figsize=(7, 5.5))
    sns.histplot(g["RawBAG"], bins=20, kde=True)
    plt.axvline(0, linestyle="--", color="black")
    plt.xlabel("Raw prediction error (years)")
    plt.ylabel("Count")
    plt.title(f"Distribution of raw OOF prediction errors — {best_fs}")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "Figure_BestModel_Raw_ErrorDistribution.png"), dpi=300, bbox_inches="tight")
    plt.close()


def plot_model_selection_frequency(model_frequency):
    if model_frequency.empty:
        return
    plt.figure(figsize=(10, 6))
    sns.barplot(data=model_frequency, x="FeatureSet", y="SelectionPercent", hue="Model")
    plt.ylabel("Selection frequency across outer folds (%)")
    plt.xlabel("Feature set")
    plt.xticks(rotation=40, ha="right")
    plt.title("Model-selection frequency in repeated nested CV")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "Figure_Model_Selection_Frequency.png"), dpi=300, bbox_inches="tight")
    plt.close()


def plot_feature_stability(selection_stability, best_fs, top_n=25):
    if selection_stability.empty:
        return
    z = selection_stability.loc[selection_stability["FeatureSet"] == best_fs].nlargest(top_n, "SelectionFrequencyPct").copy()
    if z.empty:
        return
    plt.figure(figsize=(9, 8))
    sns.barplot(data=z, x="SelectionFrequencyPct", y="Feature")
    plt.xlabel("Selection frequency across outer folds (%)")
    plt.ylabel("Feature")
    plt.title(f"Top-{top_n} feature-selection stability — {best_fs}")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "Figure_Feature_Selection_Stability.png"), dpi=300, bbox_inches="tight")
    plt.close()


def plot_top_permutation_importance(importance_summary, best_fs, top_n=25):
    if importance_summary.empty:
        return
    z = importance_summary.loc[importance_summary["FeatureSet"] == best_fs].nlargest(top_n, "MeanPermutationImportance_MAE").copy()
    if z.empty:
        return
    plt.figure(figsize=(9, 8))
    sns.barplot(data=z, x="MeanPermutationImportance_MAE", y="Feature")
    plt.xlabel("Mean OOF permutation importance (increase in MAE)")
    plt.ylabel("Feature")
    plt.title(f"Top-{top_n} out-of-sample permutation importances — {best_fs}")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "Figure_Top_OOF_Permutation_Importance.png"), dpi=300, bbox_inches="tight")
    plt.close()


def plot_importance_heatmap(importance_grouped, feature_set="Multiscale_All5"):
    if importance_grouped.empty:
        return
    z = importance_grouped.loc[importance_grouped["FeatureSet"] == feature_set].copy()
    if z.empty:
        return
    order_scale = ["WholeBrain", "Hemisphere", "Lobe", "ROI"]
    order_measure = ["volume", "area", "thickness", "curvature", "FD"]
    pivot = z.pivot(index="Phenotype", columns="Scale", values="TotalMeanPermutationImportance_MAE")
    pivot = pivot.reindex(index=order_measure, columns=[c for c in order_scale if c in pivot.columns])
    plt.figure(figsize=(8, 5.5))
    sns.heatmap(pivot, annot=True, fmt=".3f", cmap="viridis")
    plt.xlabel("Anatomical scale")
    plt.ylabel("Morphometric phenotype")
    plt.title(f"Permutation importance by phenotype × anatomical scale\n{feature_set}")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "Figure_Phenotype_x_Scale_Importance_Heatmap.png"), dpi=300, bbox_inches="tight")
    plt.close()


def best_model_diagnostics_table(agg_df, best_fs):
    """Compact manuscript-ready diagnostics for the best raw-MAE scale."""
    g = agg_df.loc[agg_df["FeatureSet"] == best_fs].copy()
    if g.empty:
        return pd.DataFrame()

    raw = regression_metrics(g["ActualAge"], g["RawBrainAge"])
    corr = regression_metrics(g["ActualAge"], g["CorrectedBrainAge"])

    # Calibration: predicted = intercept + slope * actual.
    raw_slope, raw_intercept = np.polyfit(g["ActualAge"], g["RawBrainAge"], 1)
    cor_slope, cor_intercept = np.polyfit(g["ActualAge"], g["CorrectedBrainAge"], 1)

    diff = (g["RawBrainAge"] - g["ActualAge"]).to_numpy(dtype=float)
    md = float(np.mean(diff))
    sd = float(np.std(diff, ddof=1))

    return pd.DataFrame([{
        "FeatureSet": best_fs,
        "N": len(g),
        "Raw_R2": raw["R2"],
        "Raw_RMSE": raw["RMSE"],
        "Raw_MAE": raw["MAE"],
        "Raw_MeanError": raw["MeanError"],
        "Raw_CalibrationSlope": float(raw_slope),
        "Raw_CalibrationIntercept": float(raw_intercept),
        "Raw_corr_BAG_Age": float(np.corrcoef(g["ActualAge"], g["RawBAG"])[0, 1]),
        "Corrected_R2": corr["R2"],
        "Corrected_RMSE": corr["RMSE"],
        "Corrected_MAE": corr["MAE"],
        "Corrected_CalibrationSlope": float(cor_slope),
        "Corrected_CalibrationIntercept": float(cor_intercept),
        "Corrected_corr_BAG_Age": float(np.corrcoef(g["ActualAge"], g["CorrectedBAG"])[0, 1]),
        "BlandAltman_MeanBias": md,
        "BlandAltman_Lower95LoA": md - 1.96 * sd,
        "BlandAltman_Upper95LoA": md + 1.96 * sd,
        "PrimaryPerformanceNote": "Use RAW OOF metrics for predictive accuracy; corrected BAG is secondary age-bias analysis.",
    }])


def run_targeted_robustness_models(
    df,
    feature_sets,
    mode_to_extra_columns,
    param_grid_factory,
    label_prefix,
):
    """Run scale+FD robustness models without overwriting primary results."""
    scale_order = ["WholeBrain", "Hemisphere", "Lobe", "ROI", "Multiscale"]
    targets = [f"{s}_{suffix}" for s in scale_order for suffix in ["All5", "NoFD"]]
    all_folds, all_predictions = [], []

    for mode, extras in mode_to_extra_columns.items():
        missing = [c for c in extras if c not in df.columns]
        if missing:
            raise ValueError(f"Robustness mode {mode} is missing columns: {missing}")
        for c in extras:
            df[c] = pd.to_numeric(df[c], errors="coerce")
            if df[c].notna().sum() < max(20, int(0.50 * len(df))):
                raise ValueError(f"Robustness predictor {c} has insufficient numeric data.")

        analysis_label = f"{label_prefix}:{mode}"
        for fs in targets:
            if fs not in feature_sets:
                continue
            cols = list(dict.fromkeys(list(feature_sets[fs]) + list(extras)))
            fold_df, pred_df, _, _ = run_feature_set_nested_cv(
                df,
                fs,
                cols,
                param_grid_factory=param_grid_factory,
                compute_importance=False,
                analysis_label=analysis_label,
            )
            fold_df["RobustnessMode"] = mode
            fold_df["ExtraPredictors"] = ",".join(extras) if extras else "None"
            pred_df["RobustnessMode"] = mode
            all_folds.append(fold_df)
            all_predictions.append(pred_df)

    if not all_predictions:
        empty = pd.DataFrame()
        return empty, empty, empty, empty, empty
    folds = pd.concat(all_folds, ignore_index=True)
    predictions = pd.concat(all_predictions, ignore_index=True)
    repeat_detail, repeat_summary = repeat_level_performance(predictions)
    fd_rows = []
    for analysis, g in predictions.groupby("Analysis"):
        out = paired_fd_incremental_tests(g)
        if not out.empty:
            out.insert(0, "Analysis", analysis)
            fd_rows.append(out)
    fd_tests = pd.concat(fd_rows, ignore_index=True) if fd_rows else pd.DataFrame()
    return folds, predictions, repeat_detail, repeat_summary, fd_tests


def run_nuisance_sensitivity(df, feature_sets):
    if not RUN_NUISANCE_SENSITIVITY:
        empty = pd.DataFrame()
        return empty, empty, empty, empty, empty
    return run_targeted_robustness_models(
        df, feature_sets, NUISANCE_SENSITIVITY_MODES,
        make_param_grid, "NuisanceSensitivity"
    )


def run_fixed_enet_fd_sensitivity(df, feature_sets):
    if not RUN_FIXED_ENET_FD_SENSITIVITY:
        empty = pd.DataFrame()
        return empty, empty, empty, empty, empty
    return run_targeted_robustness_models(
        df, feature_sets, {"Morphology_only": []},
        make_elasticnet_param_grid, "FixedElasticNetFD"
    )


# ============================================================
# 14. SEX-BIAS / FAIRNESS AUDIT ON REPEATED OOF PREDICTIONS
# ============================================================

def normalize_sex_labels(series):
    """Normalize documented binary sex coding without silently guessing."""
    s = series.copy()
    numeric = pd.to_numeric(s, errors="coerce")
    numeric_nonmissing = numeric.dropna()

    if len(numeric_nonmissing) == s.notna().sum():
        unknown = sorted(set(numeric_nonmissing.unique()) - set(SEX_LABEL_MAP))
        if unknown:
            raise ValueError(
                f"Unexpected numeric values in {SEX_COLUMN}: {unknown}. "
                "Update SEX_LABEL_MAP to match the documented workbook coding."
            )
        return numeric.map(SEX_LABEL_MAP)

    text_values = s.astype("string").str.strip().str.lower()
    text_map = {
        "male": "Male", "m": "Male", "man": "Male",
        "female": "Female", "f": "Female", "woman": "Female",
    }
    labels = text_values.map(text_map)
    unresolved = sorted(text_values[s.notna() & labels.isna()].dropna().unique())
    if unresolved:
        raise ValueError(
            f"Unrecognized text values in {SEX_COLUMN}: {unresolved}. "
            "Use documented Male/Female labels or update normalize_sex_labels()."
        )
    return labels


def _bootstrap_group_metric(actual, predicted, metric, rng):
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    n = len(actual)
    values = np.empty(N_BOOTSTRAP, dtype=float)

    for b in range(N_BOOTSTRAP):
        idx = rng.integers(0, n, n)
        if metric == "MAE":
            values[b] = np.mean(np.abs(predicted[idx] - actual[idx]))
        elif metric == "MeanBAG":
            values[b] = np.mean(predicted[idx] - actual[idx])
        else:
            raise ValueError(f"Unsupported group metric: {metric}")
    return tuple(np.percentile(values, [2.5, 97.5]))


def _two_group_difference_test(values, labels, rng):
    """Female-minus-Male difference with stratified bootstrap and permutation."""
    values = np.asarray(values, dtype=float)
    labels = np.asarray(labels, dtype=object)
    female = values[labels == "Female"]
    male = values[labels == "Male"]
    if len(female) < 2 or len(male) < 2:
        raise ValueError("Both sex groups require at least two observations.")

    observed = float(np.mean(female) - np.mean(male))
    boot = np.empty(N_BOOTSTRAP, dtype=float)
    for b in range(N_BOOTSTRAP):
        f = female[rng.integers(0, len(female), len(female))]
        m = male[rng.integers(0, len(male), len(male))]
        boot[b] = np.mean(f) - np.mean(m)
    ci_l, ci_u = np.percentile(boot, [2.5, 97.5])

    perm = np.empty(N_PAIRED_PERM, dtype=float)
    n_female = len(female)
    for i in range(N_PAIRED_PERM):
        shuffled = rng.permutation(values)
        perm[i] = np.mean(shuffled[:n_female]) - np.mean(shuffled[n_female:])
    p_two_sided = (
        1 + np.sum(np.abs(perm) >= abs(observed))
    ) / (N_PAIRED_PERM + 1)
    return observed, float(ci_l), float(ci_u), float(p_two_sided)


def _hc3_age_sex_interaction(y, age, female_indicator):
    """OLS y ~ Age(centered) + Female + Age×Female with HC3 robust SEs."""
    y = np.asarray(y, dtype=float)
    age = np.asarray(age, dtype=float)
    female = np.asarray(female_indicator, dtype=float)
    age_mean = float(np.mean(age))
    age_c = age - age_mean
    X = np.column_stack([
        np.ones(len(y)),
        age_c,
        female,
        age_c * female,
    ])
    names = ["Intercept", "Age_centered", "Female", "Age_x_Female"]

    xtx_inv = np.linalg.pinv(X.T @ X)
    beta = xtx_inv @ X.T @ y
    residual = y - X @ beta
    leverage = np.sum((X @ xtx_inv) * X, axis=1)
    denom = np.maximum(1.0 - leverage, 1e-8)
    adjusted_sq = (residual / denom) ** 2
    meat = X.T @ (X * adjusted_sq[:, None])
    cov_hc3 = xtx_inv @ meat @ xtx_inv
    se = np.sqrt(np.maximum(np.diag(cov_hc3), 0.0))
    t_value = np.divide(
        beta, se, out=np.full_like(beta, np.nan), where=se > 0
    )
    df_resid = max(len(y) - X.shape[1], 1)
    p_value = 2 * stats.t.sf(np.abs(t_value), df=df_resid)
    critical = stats.t.ppf(0.975, df=df_resid)

    return pd.DataFrame({
        "Term": names,
        "Estimate": beta,
        "HC3_SE": se,
        "t": t_value,
        "p_two_sided": p_value,
        "CI_L95": beta - critical * se,
        "CI_U95": beta + critical * se,
        "N": len(y),
        "AgeCenter": age_mean,
    })


def run_sex_bias_analysis(df, subject_predictions):
    """
    Audit sex-stratified accuracy, BAG, and calibration using repeated OOF
    predictions only. This does not retrain sex-specific models.
    """
    empty = (pd.DataFrame(), pd.DataFrame(), pd.DataFrame())
    if not RUN_SEX_BIAS_ANALYSIS:
        return empty
    if SEX_COLUMN not in df.columns:
        print(f"\nSex-bias analysis skipped: missing column '{SEX_COLUMN}'.")
        return empty

    sex_frame = df[["ID", "Age", SEX_COLUMN]].copy()
    sex_frame["ID"] = sex_frame["ID"].astype(str)
    sex_frame["Sex"] = normalize_sex_labels(sex_frame[SEX_COLUMN])
    if sex_frame["Sex"].isna().any():
        n_missing = int(sex_frame["Sex"].isna().sum())
        print(f"Warning: excluding {n_missing} subject(s) with missing sex.")
        sex_frame = sex_frame.loc[sex_frame["Sex"].notna()].copy()

    available = set(subject_predictions["FeatureSet"].unique())
    requested = [fs for fs in SEX_BIAS_FEATURESETS if fs in available]
    if not requested:
        print("\nSex-bias analysis skipped: requested feature sets unavailable.")
        return empty

    group_rows = []
    difference_rows = []
    regression_rows = []
    rng = np.random.default_rng(SEED + 880000)

    for fs in requested:
        g = subject_predictions.loc[
            subject_predictions["FeatureSet"] == fs
        ].merge(
            sex_frame[["ID", "Sex"]], on="ID", how="inner", validate="one_to_one"
        )
        if set(g["Sex"].unique()) != {"Female", "Male"}:
            raise ValueError(
                f"Sex-bias analysis for {fs} requires both Female and Male groups."
            )

        for sex, sg in g.groupby("Sex"):
            raw = regression_metrics(sg["ActualAge"], sg["RawBrainAge"])
            corrected = regression_metrics(
                sg["ActualAge"], sg["CorrectedBrainAge"]
            )
            raw_mae_ci = _bootstrap_group_metric(
                sg["ActualAge"], sg["RawBrainAge"], "MAE", rng
            )
            corrected_mae_ci = _bootstrap_group_metric(
                sg["ActualAge"], sg["CorrectedBrainAge"], "MAE", rng
            )
            raw_bag_ci = _bootstrap_group_metric(
                sg["ActualAge"], sg["RawBrainAge"], "MeanBAG", rng
            )
            corrected_bag_ci = _bootstrap_group_metric(
                sg["ActualAge"], sg["CorrectedBrainAge"], "MeanBAG", rng
            )
            raw_slope, raw_intercept = np.polyfit(
                sg["ActualAge"], sg["RawBrainAge"], 1
            )
            corr_slope, corr_intercept = np.polyfit(
                sg["ActualAge"], sg["CorrectedBrainAge"], 1
            )

            group_rows.append({
                "FeatureSet": fs,
                "Sex": sex,
                "N": len(sg),
                "AgeMean": float(sg["ActualAge"].mean()),
                "AgeSD": float(sg["ActualAge"].std(ddof=1)),
                "Raw_MAE": raw["MAE"],
                "Raw_MAE_L95": raw_mae_ci[0],
                "Raw_MAE_U95": raw_mae_ci[1],
                "Raw_RMSE": raw["RMSE"],
                "Raw_R2": raw["R2"],
                "Raw_MeanBAG": raw["MeanError"],
                "Raw_MeanBAG_L95": raw_bag_ci[0],
                "Raw_MeanBAG_U95": raw_bag_ci[1],
                "Raw_CalibrationSlope": float(raw_slope),
                "Raw_CalibrationIntercept": float(raw_intercept),
                "Raw_corr_BAG_Age": float(np.corrcoef(
                    sg["RawBAG"], sg["ActualAge"]
                )[0, 1]),
                "Corrected_MAE": corrected["MAE"],
                "Corrected_MAE_L95": corrected_mae_ci[0],
                "Corrected_MAE_U95": corrected_mae_ci[1],
                "Corrected_RMSE": corrected["RMSE"],
                "Corrected_R2": corrected["R2"],
                "Corrected_MeanBAG": corrected["MeanError"],
                "Corrected_MeanBAG_L95": corrected_bag_ci[0],
                "Corrected_MeanBAG_U95": corrected_bag_ci[1],
                "Corrected_CalibrationSlope": float(corr_slope),
                "Corrected_CalibrationIntercept": float(corr_intercept),
                "Corrected_corr_BAG_Age": float(np.corrcoef(
                    sg["CorrectedBAG"], sg["ActualAge"]
                )[0, 1]),
            })

        comparison_values = {
            "RawAbsoluteError": np.abs(g["RawBrainAge"] - g["ActualAge"]),
            "RawBAG": g["RawBAG"],
            "CorrectedBAG": g["CorrectedBAG"],
        }
        for outcome, values in comparison_values.items():
            estimate, ci_l, ci_u, p = _two_group_difference_test(
                values, g["Sex"], rng
            )
            difference_rows.append({
                "FeatureSet": fs,
                "Outcome": outcome,
                "Contrast": "Female_minus_Male",
                "Difference": estimate,
                "CI_L95": ci_l,
                "CI_U95": ci_u,
                "Permutation_p_two_sided_unadjusted": p,
                "InterpretationNote": (
                    "Unadjusted descriptive group contrast; use age-adjusted "
                    "HC3 model for primary sex-bias inference."
                ),
            })

        female_indicator = (g["Sex"] == "Female").astype(float)
        regression_outcomes = {
            "RawBAG": g["RawBAG"],
            "CorrectedBAG": g["CorrectedBAG"],
            "RawAbsoluteError": np.abs(g["RawBrainAge"] - g["ActualAge"]),
            "RawPredictedAge_Calibration": g["RawBrainAge"],
            "CorrectedPredictedAge_Calibration": g["CorrectedBrainAge"],
        }
        for outcome, values in regression_outcomes.items():
            fit = _hc3_age_sex_interaction(
                values, g["ActualAge"], female_indicator
            )
            fit.insert(0, "Outcome", outcome)
            fit.insert(0, "FeatureSet", fs)
            fit["Model"] = (
                "Outcome ~ Age_centered + Female + Age_centered×Female; HC3 SE"
            )
            regression_rows.append(fit)

    group_metrics = pd.DataFrame(group_rows)
    group_differences = pd.DataFrame(difference_rows)
    age_adjusted = pd.concat(regression_rows, ignore_index=True)

    if not group_differences.empty:
        group_differences["Permutation_q_BH"] = bh_fdr(
            group_differences["Permutation_p_two_sided_unadjusted"].to_numpy()
        )
    sex_terms = age_adjusted["Term"].isin(["Female", "Age_x_Female"])
    age_adjusted["SexTerms_q_BH"] = np.nan
    age_adjusted.loc[sex_terms, "SexTerms_q_BH"] = bh_fdr(
        age_adjusted.loc[sex_terms, "p_two_sided"].to_numpy()
    )
    age_adjusted["InferenceNote"] = (
        "Female is the adjusted difference at mean age; Age_x_Female is the "
        "difference in age slope. BH-FDR covers all sex-related terms in this audit."
    )
    return group_metrics, group_differences, age_adjusted


def plot_sex_bias_summary(group_metrics):
    if group_metrics.empty:
        return

    sets = list(group_metrics["FeatureSet"].drop_duplicates())
    sexes = ["Female", "Male"]
    colors = {"Female": "#D95F02", "Male": "#2C7BB6"}
    x = np.arange(len(sets), dtype=float)
    width = 0.34
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.0))

    for j, sex in enumerate(sexes):
        sg = (
            group_metrics.loc[group_metrics["Sex"] == sex]
            .set_index("FeatureSet")
            .reindex(sets)
        )
        offset = (j - 0.5) * width
        axes[0].bar(
            x + offset, sg["Raw_MAE"], width=width,
            color=colors[sex], alpha=0.85, label=sex,
            yerr=np.vstack([
                sg["Raw_MAE"] - sg["Raw_MAE_L95"],
                sg["Raw_MAE_U95"] - sg["Raw_MAE"],
            ]), capsize=4,
        )
        axes[1].bar(
            x + offset, sg["Corrected_MeanBAG"], width=width,
            color=colors[sex], alpha=0.85, label=sex,
            yerr=np.vstack([
                sg["Corrected_MeanBAG"] - sg["Corrected_MeanBAG_L95"],
                sg["Corrected_MeanBAG_U95"] - sg["Corrected_MeanBAG"],
            ]), capsize=4,
        )

    axes[0].set_ylabel("Raw OOF MAE (years)")
    axes[0].set_title("Prediction error by sex")
    axes[1].axhline(0, color="black", linewidth=1)
    axes[1].set_ylabel("Corrected mean BAG (years)")
    axes[1].set_title("Residual mean bias by sex")
    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels([s.replace("_", " ") for s in sets], rotation=12)
        ax.grid(axis="y", alpha=0.25)
    axes[0].legend(frameon=False)
    fig.suptitle("Supplementary sex-bias audit on repeated OOF predictions")
    fig.tight_layout()
    fig.savefig(
        os.path.join(OUTPUT_DIR, "Supplementary_Sex_Bias_OOF.png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)


# ============================================================
# 15. LEAKAGE-SAFE NESTED LEARNING-CURVE ANALYSIS
# ============================================================

def nested_stratified_training_subsets(y_outer_train, fractions, random_state):
    """Create age-balanced nested subsets: every smaller set is a true subset."""
    y_outer_train = np.asarray(y_outer_train, dtype=float)
    n = len(y_outer_train)
    bins = make_age_bins(y_outer_train, n_bins=5)
    rng = np.random.default_rng(random_state)
    queues = {}
    for b in sorted(np.unique(bins)):
        idx = np.flatnonzero(bins == b)
        queues[b] = list(rng.permutation(idx))

    order = []
    while len(order) < n:
        active = [b for b, q in queues.items() if q]
        if not active:
            break
        for b in rng.permutation(active):
            if queues[b]:
                order.append(queues[b].pop())
    order = np.asarray(order, dtype=int)

    out = {}
    for fraction in sorted(set(float(x) for x in fractions)):
        if fraction >= 1.0:
            n_target = n
        else:
            n_target = max(int(round(fraction * n)), INNER_FOLDS * 10)
            n_target = min(n_target, n - 1)
        out[fraction] = np.sort(order[:n_target])
    return out


def run_nested_learning_curves(df, feature_sets):
    """
    Estimate learning curves without exposing outer-validation observations.

    At every feature set, outer split, and training fraction, the complete
    imputation -> scaling -> feature-selection -> model-selection pipeline is
    retuned by inner CV using only the selected outer-training subset.
    """
    if not RUN_LEARNING_CURVES:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    requested = [
        fs for fs in LEARNING_CURVE_FEATURESETS
        if fs in feature_sets
    ]
    if not requested:
        print("\nLearning curves skipped: requested feature sets unavailable.")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    y = df["Age"].to_numpy(dtype=float)
    ids = df["ID"].astype(str).to_numpy()
    fold_rows = []
    pred_rows = []

    print("\n" + "=" * 78)
    print("NESTED LEARNING-CURVE ANALYSIS")
    print("=" * 78)

    for fs_index, fs in enumerate(requested):
        cols = feature_sets[fs]
        X = df[cols].copy()

        for repeat, fold, tr_idx, va_idx in repeated_outer_splits(y):
            y_outer_train = y[tr_idx]
            X_va = X.iloc[va_idx].copy()
            y_va = y[va_idx]

            nested_subsets = nested_stratified_training_subsets(
                y_outer_train,
                LEARNING_CURVE_FRACTIONS,
                random_state=SEED + repeat * 1000 + fold * 10,
            )

            for fraction in LEARNING_CURVE_FRACTIONS:
                subset_local = nested_subsets[float(fraction)]
                subset_global = tr_idx[subset_local]
                X_sub = X.iloc[subset_global].copy()
                y_sub = y[subset_global]

                inner_cv = make_inner_cv(
                    y_sub,
                    random_state=(
                        SEED
                        + fs_index * 100000
                        + repeat * 1000
                        + fold * 10
                        + int(round(fraction * 100))
                    ),
                )
                search = GridSearchCV(
                    estimator=make_base_pipeline(),
                    param_grid=make_param_grid(len(cols)),
                    scoring="neg_mean_absolute_error",
                    cv=inner_cv,
                    n_jobs=-1,
                    refit=True,
                    return_train_score=False,
                    error_score="raise",
                )
                search.fit(X_sub, y_sub)
                best = search.best_estimator_

                pred_train = best.predict(X_sub)
                pred_validation = best.predict(X_va)
                train_m = regression_metrics(y_sub, pred_train)
                validation_m = regression_metrics(y_va, pred_validation)
                chosen_model = model_name_from_estimator(
                    best.named_steps["model"]
                )

                print(
                    f"[LearningCurve:{fs}] repeat {repeat}/{N_REPEATS} "
                    f"| fold {fold}/{OUTER_FOLDS} | fraction={fraction:.2f} "
                    f"| n={len(y_sub)} | val MAE={validation_m['MAE']:.3f}"
                )

                fold_rows.append({
                    "FeatureSet": fs,
                    "Fraction": float(fraction),
                    "Repeat": repeat,
                    "Fold": fold,
                    "N_OuterTrainAvailable": len(tr_idx),
                    "N_TrainingSubset": len(y_sub),
                    "N_Validation": len(va_idx),
                    "Model": chosen_model,
                    "N_Selected": len(selected_feature_names(best, cols)),
                    "InnerBest_MAE": -float(search.best_score_),
                    "Training_R2": train_m["R2"],
                    "Training_RMSE": train_m["RMSE"],
                    "Training_MAE": train_m["MAE"],
                    "Validation_R2": validation_m["R2"],
                    "Validation_RMSE": validation_m["RMSE"],
                    "Validation_MAE": validation_m["MAE"],
                    "BestParamsJSON": json.dumps(
                        {
                            k: (
                                model_name_from_estimator(v)
                                if k == "model" else v
                            )
                            for k, v in search.best_params_.items()
                        },
                        default=str,
                        sort_keys=True,
                    ),
                })

                for subject_id, actual, predicted in zip(
                    ids[va_idx], y_va, pred_validation
                ):
                    pred_rows.append({
                        "FeatureSet": fs,
                        "Fraction": float(fraction),
                        "Repeat": repeat,
                        "Fold": fold,
                        "ID": subject_id,
                        "ActualAge": actual,
                        "PredictedAge": predicted,
                    })

    folds = pd.DataFrame(fold_rows)
    predictions = pd.DataFrame(pred_rows)
    summary_rows = []

    for group_index, ((fs, fraction), g) in enumerate(
        predictions.groupby(["FeatureSet", "Fraction"], sort=False)
    ):
        subject = (
            g.groupby("ID", as_index=False)
            .agg(
                ActualAge=("ActualAge", "first"),
                PredictedAge=("PredictedAge", "mean"),
                Prediction_SD=("PredictedAge", "std"),
            )
        )
        metrics = regression_metrics(
            subject["ActualAge"], subject["PredictedAge"]
        )
        ci = bootstrap_metrics_subject_level(
            subject["ActualAge"].to_numpy(),
            subject["PredictedAge"].to_numpy(),
            seed=SEED + 700000 + group_index,
        )
        fg = folds.loc[
            (folds["FeatureSet"] == fs)
            & np.isclose(folds["Fraction"], fraction)
        ]
        train_mae = float(np.average(
            fg["Training_MAE"], weights=fg["N_TrainingSubset"]
        ))

        summary_rows.append({
            "FeatureSet": fs,
            "Fraction": float(fraction),
            "Mean_N_Training": float(fg["N_TrainingSubset"].mean()),
            "Min_N_Training": int(fg["N_TrainingSubset"].min()),
            "Max_N_Training": int(fg["N_TrainingSubset"].max()),
            "N_Subjects_Validation": len(subject),
            "N_OuterEvaluations": len(fg),
            "Training_MAE": train_mae,
            "Training_MAE_SD_AcrossOuterFits": float(
                fg["Training_MAE"].std(ddof=1)
            ),
            "Validation_MAE": metrics["MAE"],
            "Validation_MAE_L95": ci["MAE_L95"],
            "Validation_MAE_U95": ci["MAE_U95"],
            "Validation_RMSE": metrics["RMSE"],
            "Validation_RMSE_L95": ci["RMSE_L95"],
            "Validation_RMSE_U95": ci["RMSE_U95"],
            "Validation_R2": metrics["R2"],
            "Validation_R2_L95": ci["R2_L95"],
            "Validation_R2_U95": ci["R2_U95"],
            "GeneralizationGap_MAE": metrics["MAE"] - train_mae,
            "UncertaintyNote": (
                "Participant-bootstrap CI conditional on averaged repeated-OOF "
                "predictions; it does not refit the full nested workflow."
            ),
        })

    summary = pd.DataFrame(summary_rows).sort_values(
        ["FeatureSet", "Fraction"]
    )
    return folds, predictions, summary


def learning_curve_diagnostics(summary):
    """Descriptive plateau diagnostics; not a formal adequacy test."""
    rows = []
    if summary.empty:
        return pd.DataFrame()

    for fs, g in summary.groupby("FeatureSet"):
        g = g.sort_values("Mean_N_Training")
        last = g.iloc[-1]
        previous = g.iloc[-2] if len(g) >= 2 else g.iloc[-1]
        tail = g.tail(min(3, len(g)))

        if len(tail) >= 2 and tail["Mean_N_Training"].nunique() > 1:
            slope_per_100 = float(
                np.polyfit(
                    tail["Mean_N_Training"],
                    tail["Validation_MAE"],
                    1,
                )[0] * 100
            )
        else:
            slope_per_100 = np.nan

        last_step_improvement = float(
            previous["Validation_MAE"] - last["Validation_MAE"]
        )
        approximate_plateau = bool(
            np.isfinite(slope_per_100)
            and abs(slope_per_100) < 0.25
            and abs(last_step_improvement) < 0.10
        )
        rows.append({
            "FeatureSet": fs,
            "Largest_Mean_N_Training": last["Mean_N_Training"],
            "Validation_MAE_at_Largest_N": last["Validation_MAE"],
            "LastStep_DeltaMAE_Improvement": last_step_improvement,
            "TailSlope_MAE_per_100_AdditionalSubjects": slope_per_100,
            "ApproximatePlateau_Flag": approximate_plateau,
            "Interpretation": (
                "Approximate plateau within evaluated range"
                if approximate_plateau else
                "No clear plateau; additional data may improve or stabilize performance"
            ),
            "Caution": (
                "Descriptive diagnostic only; does not prove sample-size "
                "adequacy and does not replace external validation."
            ),
        })
    return pd.DataFrame(rows)


def plot_nested_learning_curves(summary):
    if summary.empty:
        return

    preferred_order = ["Lobe_All5", "ROI_All5", "Multiscale_All5"]
    available = list(summary["FeatureSet"].drop_duplicates())
    feature_sets = [fs for fs in preferred_order if fs in available]
    feature_sets += [fs for fs in available if fs not in feature_sets]
    panel_titles = {
        "Lobe_All5": "Lobe — all five phenotypes",
        "ROI_All5": "ROI — all five phenotypes",
        "Multiscale_All5": "Multiscale — all five phenotypes",
    }
    fig, axes = plt.subplots(
        1,
        len(feature_sets),
        figsize=(6.2 * len(feature_sets), 5.2),
        sharey=True,
    )
    if len(feature_sets) == 1:
        axes = [axes]

    for ax, fs in zip(axes, feature_sets):
        g = summary.loc[summary["FeatureSet"] == fs].sort_values(
            "Mean_N_Training"
        )
        x = g["Mean_N_Training"].to_numpy(dtype=float)
        train = g["Training_MAE"].to_numpy(dtype=float)
        train_sd = g["Training_MAE_SD_AcrossOuterFits"].to_numpy(dtype=float)
        val = g["Validation_MAE"].to_numpy(dtype=float)
        val_l = g["Validation_MAE_L95"].to_numpy(dtype=float)
        val_u = g["Validation_MAE_U95"].to_numpy(dtype=float)

        # MAE is bounded below by zero. Clipping only affects the visual SD/CI
        # envelope; point estimates and exported numerical results are unchanged.
        train_lower = np.maximum(train - train_sd, 0.0)
        train_upper = train + train_sd
        val_lower = np.maximum(val_l, 0.0)
        val_upper = val_u

        ax.plot(
            x, train, marker="o", color="#4472C4",
            label="In-sample training MAE"
        )
        ax.fill_between(
            x, train_lower, train_upper,
            color="#4472C4", alpha=0.15,
            label="Training variability (±1 outer-fit SD)"
        )
        ax.plot(
            x, val, marker="o", color="#C00000",
            label="Outer-validation MAE"
        )
        ax.fill_between(
            x, val_lower, val_upper,
            color="#C00000", alpha=0.16,
            label="Validation 95% participant-bootstrap CI"
        )
        ax.set_title(panel_titles.get(fs, fs.replace("_", " ")))
        ax.set_xlabel("Mean number of training participants")
        ax.grid(alpha=0.25)
        ax.set_ylim(bottom=0)

    axes[0].set_ylabel("Mean absolute error (years)")
    handles, labels = axes[-1].get_legend_handles_labels()
    fig.legend(
        handles, labels, loc="lower center", ncol=2,
        bbox_to_anchor=(0.5, -0.04), frameon=False
    )
    fig.suptitle(
        "Leakage-safe nested learning curves",
        fontsize=14,
        y=1.02,
    )
    fig.tight_layout(rect=[0, 0.11, 1, 1])
    fig.savefig(
        os.path.join(OUTPUT_DIR, "Supplementary_Nested_Learning_Curves.png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)


# ============================================================
# 16. MAIN
# ============================================================

def main():
    print("\nBrainAge Multiscale Morphometry Pipeline")
    print("========================================")

    if USE_XGBOOST_IF_AVAILABLE:
        print(
            "XGBoost status:",
            "available" if HAS_XGBOOST else "NOT installed — ENet + RF only",
        )
    if REQUIRE_XGBOOST_FOR_FINAL and not HAS_XGBOOST:
        raise ImportError(
            "Final analysis requires xgboost, but it is not installed. "
            "Install xgboost or explicitly set REQUIRE_XGBOOST_FOR_FINAL=False "
            "only for debugging—not for manuscript results."
        )
    if RUN_FINAL_SHAP and REQUIRE_SHAP_FOR_FINAL and not HAS_SHAP:
        raise ImportError(
            "Final analysis requests SHAP outputs, but shap is not installed. "
            "Install shap or disable both RUN_FINAL_SHAP and "
            "REQUIRE_SHAP_FOR_FINAL only for debugging."
        )

    print("\nLoading dataset...")
    df = load_dataset(DATA_PATH, SHEET_NAME)

    # Use a copy because validation converts model variables to numeric.
    feature_sets = {
        k: list(v)
        for k, v in FEATURE_SETS.items()
    }

    df, feature_sets = validate_feature_manifest(
        df,
        feature_sets,
    )

    # Update global manifest after sparse-feature filtering.
    FEATURE_SETS.clear()
    FEATURE_SETS.update(feature_sets)

    print(f"N subjects: {len(df)}")
    print(
        f"Age range: {df['Age'].min():.1f}–{df['Age'].max():.1f}"
    )
    print(f"Feature sets: {len(FEATURE_SETS)}")

    # Save feature manifest before modeling.
    manifest_rows = []
    for fs, cols in FEATURE_SETS.items():
        for c in cols:
            meta = parse_feature_metadata(c)
            manifest_rows.append({
                "FeatureSet": fs,
                "Feature": c,
                "Scale": meta["Scale"],
                "Phenotype": meta["Phenotype"],
            })

    manifest_df = pd.DataFrame(manifest_rows)
    manifest_df.to_excel(
        os.path.join(
            OUTPUT_DIR,
            "Feature_Set_Manifest.xlsx",
        ),
        index=False,
    )

    all_fold = []
    all_pred = []
    all_selected = []
    all_importance = []

    for fs, cols in FEATURE_SETS.items():
        fold_df, pred_df, sel_df, imp_df = (
            run_feature_set_nested_cv(
                df,
                fs,
                cols,
            )
        )

        all_fold.append(fold_df)
        all_pred.append(pred_df)
        all_selected.append(sel_df)

        if not imp_df.empty:
            all_importance.append(imp_df)

    fold_results = pd.concat(
        all_fold,
        ignore_index=True,
    )
    predictions_long = pd.concat(
        all_pred,
        ignore_index=True,
    )
    selected_long = pd.concat(
        all_selected,
        ignore_index=True,
    )

    if all_importance:
        importance_long = pd.concat(
            all_importance,
            ignore_index=True,
        )
    else:
        importance_long = pd.DataFrame()

    subject_predictions = aggregate_subject_predictions(
        predictions_long
    )

    performance = performance_summary_from_aggregated(
        subject_predictions
    )

    selection_stability = selection_stability_summary(
        selected_long
    )

    (
        importance_summary,
        importance_grouped,
    ) = permutation_importance_summary(
        importance_long
    )

    repeat_performance_detail, repeat_performance_summary = (
        repeat_level_performance(predictions_long)
    )

    fd_incremental = paired_fd_incremental_tests(predictions_long)

    scale_comparison_pairs = [
        ("Hemisphere_vs_WholeBrain", "Hemisphere_All5", "WholeBrain_All5"),
        ("Lobe_vs_Hemisphere", "Lobe_All5", "Hemisphere_All5"),
        ("ROI_vs_Lobe", "ROI_All5", "Lobe_All5"),
        ("Multiscale_vs_ROI", "Multiscale_All5", "ROI_All5"),
    ]
    scale_pairwise = (
        paired_workflow_tests(
            predictions_long, scale_comparison_pairs, "Adjacent_scale_comparisons"
        ) if RUN_PAIRED_SCALE_COMPARISONS else pd.DataFrame()
    )

    phenotype_comparison_pairs = [
        (f"All5_vs_{p}", "ROI_All5", f"ROI_{p}")
        for p in ["volume", "area", "thickness", "curvature", "FD"]
    ]
    phenotype_pairwise = (
        paired_workflow_tests(
            predictions_long, phenotype_comparison_pairs,
            "ROI_All5_vs_single_phenotypes"
        ) if RUN_PAIRED_PHENOTYPE_COMPARISONS else pd.DataFrame()
    )

    # Model-selection frequency across outer folds.
    model_frequency = (
        fold_results
        .groupby(
            ["FeatureSet", "Model"],
            as_index=False,
        )
        .size()
        .rename(columns={"size": "SelectionCount"})
    )
    model_frequency["SelectionPercent"] = (
        100
        * model_frequency["SelectionCount"]
        / (
            OUTER_FOLDS
            * N_REPEATS
        )
    )

    # --------------------------------------------------------
    # Pick best SCALE benchmark by PRIMARY raw repeated-OOF MAE.
    # This determines only the final descriptive model/plots.
    # It does NOT alter the already-completed nested-CV estimates.
    # --------------------------------------------------------
    scale_sets = [
        "WholeBrain_All5",
        "Hemisphere_All5",
        "Lobe_All5",
        "ROI_All5",
        "Multiscale_All5",
    ]

    scale_perf = repeat_performance_summary.loc[
        repeat_performance_summary["FeatureSet"].isin(scale_sets)
    ].copy()

    if scale_perf.empty:
        best_fs = performance.iloc[0]["FeatureSet"]
    else:
        best_fs = (
            scale_perf
            .sort_values("Raw_MAE_MeanAcrossRepeats")
            .iloc[0]["FeatureSet"]
        )

    print(f"\nBest all-5 anatomical scale by mean repeat-level OOF MAE: {best_fs}")

    best_diagnostics = best_model_diagnostics_table(subject_predictions, best_fs)

    # --------------------------------------------------------
    # Supplementary sex-bias/fairness audit on repeated OOF predictions.
    # --------------------------------------------------------
    (
        sex_bias_group_metrics,
        sex_bias_group_differences,
        sex_bias_age_adjusted,
    ) = run_sex_bias_analysis(df, subject_predictions)

    # --------------------------------------------------------
    # Supplementary nested learning curves.
    # This is a sample-size diagnostic, not a correction for small N.
    # --------------------------------------------------------
    (
        learning_curve_folds,
        learning_curve_predictions,
        learning_curve_summary,
    ) = run_nested_learning_curves(df, FEATURE_SETS)
    learning_curve_diagnostic = learning_curve_diagnostics(
        learning_curve_summary
    )

    # Reviewer-facing robustness analyses. These add substantial runtime but
    # are kept fully separate from the primary morphology-only results.
    (
        nuisance_folds,
        nuisance_predictions,
        nuisance_repeat_detail,
        nuisance_repeat_summary,
        nuisance_fd_tests,
    ) = run_nuisance_sensitivity(df, FEATURE_SETS)

    (
        fixed_enet_folds,
        fixed_enet_predictions,
        fixed_enet_repeat_detail,
        fixed_enet_repeat_summary,
        fixed_enet_fd_tests,
    ) = run_fixed_enet_fd_sensitivity(df, FEATURE_SETS)

    analysis_config = pd.DataFrame([
        ["SEED", SEED], ["OUTER_FOLDS", OUTER_FOLDS],
        ["N_REPEATS", N_REPEATS], ["INNER_FOLDS", INNER_FOLDS],
        ["N_BOOTSTRAP", N_BOOTSTRAP], ["N_PAIRED_PERM", N_PAIRED_PERM],
        ["ADD_NUISANCE_PREDICTORS_PRIMARY", ADD_NUISANCE_PREDICTORS],
        ["RUN_LEARNING_CURVES", RUN_LEARNING_CURVES],
        ["RUN_SEX_BIAS_ANALYSIS", RUN_SEX_BIAS_ANALYSIS],
        ["RUN_NUISANCE_SENSITIVITY", RUN_NUISANCE_SENSITIVITY],
        ["RUN_FIXED_ENET_FD_SENSITIVITY", RUN_FIXED_ENET_FD_SENSITIVITY],
        ["FD_TEST_SIDEDNESS", "Two-sided"],
        ["PRIMARY_PERFORMANCE", "Mean across complete repeated OOF runs"],
        ["XGBOOST_AVAILABLE", HAS_XGBOOST], ["SHAP_AVAILABLE", HAS_SHAP],
    ], columns=["Setting", "Value"])

    # --------------------------------------------------------
    # Save one consolidated workbook
    # --------------------------------------------------------
    result_xlsx = os.path.join(
        OUTPUT_DIR,
        "BrainAge_AnatomicalMultiscale_Q1_Results.xlsx",
    )

    with pd.ExcelWriter(
        result_xlsx,
        engine="openpyxl",
    ) as writer:
        performance.to_excel(
            writer,
            sheet_name="Performance",
            index=False,
        )
        fold_results.to_excel(
            writer,
            sheet_name="NestedCV_Folds",
            index=False,
        )
        subject_predictions.to_excel(
            writer,
            sheet_name="Subject_OOF_Predictions",
            index=False,
        )
        model_frequency.to_excel(
            writer,
            sheet_name="Model_Selection",
            index=False,
        )
        selection_stability.to_excel(
            writer,
            sheet_name="Feature_Stability",
            index=False,
        )
        fd_incremental.to_excel(
            writer,
            sheet_name="FD_Incremental",
            index=False,
        )
        analysis_config.to_excel(
            writer, sheet_name="Analysis_Config", index=False
        )
        repeat_performance_detail.to_excel(
            writer, sheet_name="RepeatPerf_Detail", index=False
        )
        repeat_performance_summary.to_excel(
            writer, sheet_name="RepeatPerf_Summary", index=False
        )
        if not scale_pairwise.empty:
            scale_pairwise.to_excel(
                writer, sheet_name="Paired_Scales", index=False
            )
        if not phenotype_pairwise.empty:
            phenotype_pairwise.to_excel(
                writer, sheet_name="Paired_Phenotypes", index=False
            )
        manifest_df.to_excel(
            writer,
            sheet_name="Feature_Manifest",
            index=False,
        )
        if not best_diagnostics.empty:
            best_diagnostics.to_excel(
                writer,
                sheet_name="BestModel_Diagnostics",
                index=False,
            )
        if not learning_curve_summary.empty:
            learning_curve_summary.to_excel(
                writer,
                sheet_name="LearningCurve_Summary",
                index=False,
            )
            learning_curve_diagnostic.to_excel(
                writer,
                sheet_name="LearningCurve_Diagnostic",
                index=False,
            )
            learning_curve_folds.to_excel(
                writer,
                sheet_name="LearningCurve_Folds",
                index=False,
            )
        if not sex_bias_group_metrics.empty:
            sex_bias_group_metrics.to_excel(
                writer,
                sheet_name="SexBias_GroupMetrics",
                index=False,
            )
            sex_bias_group_differences.to_excel(
                writer,
                sheet_name="SexBias_GroupDiff",
                index=False,
            )
            sex_bias_age_adjusted.to_excel(
                writer,
                sheet_name="SexBias_AgeAdjusted",
                index=False,
            )
        if not nuisance_repeat_summary.empty:
            nuisance_repeat_summary.to_excel(
                writer, sheet_name="Nuisance_RepeatSummary", index=False
            )
            nuisance_fd_tests.to_excel(
                writer, sheet_name="Nuisance_FD_Tests", index=False
            )
            nuisance_folds.to_excel(
                writer, sheet_name="Nuisance_Folds", index=False
            )
        if not fixed_enet_repeat_summary.empty:
            fixed_enet_repeat_summary.to_excel(
                writer, sheet_name="FixedEN_RepeatSummary", index=False
            )
            fixed_enet_fd_tests.to_excel(
                writer, sheet_name="FixedEN_FD_Tests", index=False
            )
            fixed_enet_folds.to_excel(
                writer, sheet_name="FixedEN_Folds", index=False
            )

        if not importance_summary.empty:
            importance_summary.to_excel(
                writer,
                sheet_name="OOF_PermutationImportance",
                index=False,
            )
            importance_grouped.to_excel(
                writer,
                sheet_name="Importance_ByScaleMeasure",
                index=False,
            )

    predictions_long.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "All_Repeated_OOF_Predictions.csv",
        ),
        index=False,
    )

    selected_long.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "All_OuterFold_FeatureSelections.csv",
        ),
        index=False,
    )

    if not importance_long.empty:
        importance_long.to_csv(
            os.path.join(
                OUTPUT_DIR,
                "All_OOF_PermutationImportance.csv",
            ),
            index=False,
        )

    if not learning_curve_predictions.empty:
        learning_curve_predictions.to_csv(
            os.path.join(
                OUTPUT_DIR,
                "Nested_LearningCurve_OOF_Predictions.csv",
            ),
            index=False,
        )
    if not nuisance_predictions.empty:
        nuisance_predictions.to_csv(
            os.path.join(OUTPUT_DIR, "Nuisance_Sensitivity_OOF_Predictions.csv"),
            index=False,
        )
    if not fixed_enet_predictions.empty:
        fixed_enet_predictions.to_csv(
            os.path.join(OUTPUT_DIR, "Fixed_ElasticNet_FD_OOF_Predictions.csv"),
            index=False,
        )

    # --------------------------------------------------------
    # Figures
    # --------------------------------------------------------
    plot_scale_benchmark(performance)
    plot_roi_phenotype(performance)
    plot_fd_incremental(fd_incremental)
    plot_best_brainage(subject_predictions, best_fs)
    plot_model_selection_frequency(model_frequency)
    plot_feature_stability(selection_stability, best_fs)
    plot_top_permutation_importance(importance_summary, best_fs)
    plot_importance_heatmap(importance_grouped, feature_set="Multiscale_All5")
    plot_nested_learning_curves(learning_curve_summary)
    plot_sex_bias_summary(sex_bias_group_metrics)

    # --------------------------------------------------------
    # Final descriptive model + SHAP.
    # Performance is NOT taken from this full-data model.
    # --------------------------------------------------------
    final_best, final_search = fit_final_descriptive_model(
        df,
        best_fs,
        FEATURE_SETS[best_fs],
    )

    final_model_info = pd.DataFrame([{
        "FeatureSet": best_fs,
        "SelectedModel": model_name_from_estimator(
            final_best.named_steps["model"]
        ),
        "CV_BestMAE": -float(final_search.best_score_),
        "N_Selected": len(
            selected_feature_names(
                final_best,
                FEATURE_SETS[best_fs],
            )
        ),
        "BestParamsJSON": json.dumps(
            {
                k: (
                    model_name_from_estimator(v)
                    if k == "model"
                    else v
                )
                for k, v in final_search.best_params_.items()
            },
            default=str,
            sort_keys=True,
        ),
        "InterpretationNote": (
            "Full-data model for description/deployment only; "
            "unbiased performance comes from repeated nested CV."
        ),
    }])

    final_model_info.to_excel(
        os.path.join(
            OUTPUT_DIR,
            "Final_Descriptive_Model_Info.xlsx",
        ),
        index=False,
    )

    final_explain = make_final_shap_outputs(
        df,
        best_fs,
        FEATURE_SETS[best_fs],
        final_best,
        subject_oof_predictions=subject_predictions,
    )

    if not final_explain.empty:
        final_explain.to_excel(
            os.path.join(
                OUTPUT_DIR,
                "Final_Descriptive_Explainability.xlsx",
            ),
            index=False,
        )

    # --------------------------------------------------------
    # Plain-text analysis summary
    # --------------------------------------------------------
    summary_txt = os.path.join(
        OUTPUT_DIR,
        "RUN_SUMMARY.txt",
    )

    with open(summary_txt, "w", encoding="utf-8") as f:
        f.write("BrainAge Multiscale Morphometry Pipeline\n")
        f.write("========================================\n\n")
        f.write(f"N = {len(df)}\n")
        f.write(
            f"Age range = {df['Age'].min():.1f}–"
            f"{df['Age'].max():.1f}\n"
        )
        f.write(
            f"Repeated nested CV = {OUTER_FOLDS} folds x "
            f"{N_REPEATS} repeats; inner={INNER_FOLDS}\n"
        )
        f.write(
            f"XGBoost available = {HAS_XGBOOST}\n"
        )
        f.write(
            f"Primary nuisance predictors added = "
            f"{ADD_NUISANCE_PREDICTORS}\n\n"
        )
        f.write(
            "Excluded by design: cognition, education, hand, "
            "BrainNormR Norm_Z/CF_Norm_Z, centiles, derivatives.\n\n"
        )
        f.write(f"Best all-5 spatial scale = {best_fs}\n\n")
        f.write("Primary repeat-level performance summary:\n")
        f.write(repeat_performance_summary.to_string(index=False))
        f.write("\n\nSecondary participant-averaged OOF performance rows:\n")
        f.write(
            performance.head(10).to_string(index=False)
        )
        f.write("\n\nFD incremental test:\n")
        f.write(
            fd_incremental.to_string(index=False)
            if not fd_incremental.empty
            else "Not run."
        )
        f.write("\n\nAdjacent-scale paired comparisons:\n")
        f.write(scale_pairwise.to_string(index=False) if not scale_pairwise.empty else "Not run.")
        f.write("\n\nAll5 versus single-phenotype comparisons:\n")
        f.write(phenotype_pairwise.to_string(index=False) if not phenotype_pairwise.empty else "Not run.")
        f.write("\n\nNuisance sensitivity FD tests:\n")
        f.write(nuisance_fd_tests.to_string(index=False) if not nuisance_fd_tests.empty else "Not run.")
        f.write("\n\nFixed-ElasticNet FD sensitivity tests:\n")
        f.write(fixed_enet_fd_tests.to_string(index=False) if not fixed_enet_fd_tests.empty else "Not run.")
        f.write("\n\nLearning-curve diagnostic:\n")
        f.write(
            learning_curve_diagnostic.to_string(index=False)
            if not learning_curve_diagnostic.empty
            else "Not run."
        )
        f.write("\n\nSex-bias group metrics:\n")
        f.write(
            sex_bias_group_metrics.to_string(index=False)
            if not sex_bias_group_metrics.empty
            else "Not run."
        )
        f.write("\n\nAge-adjusted sex-related terms:\n")
        if not sex_bias_age_adjusted.empty:
            sex_terms = sex_bias_age_adjusted.loc[
                sex_bias_age_adjusted["Term"].isin(
                    ["Female", "Age_x_Female"]
                )
            ]
            f.write(sex_terms.to_string(index=False))
        else:
            f.write("Not run.")

    figures_manifest = pd.DataFrame([
        ["Main", "Figure_SpatialScale_Benchmark_MAE.png", "Raw OOF MAE across anatomical scales with 95% bootstrap CI"],
        ["Main", "Figure_ROI_Phenotype_Ablation_MAE.png", "ROI phenotype ablation with 95% bootstrap CI"],
        ["Main", "Figure_FD_Incremental_Value.png", "Paired FD incremental value with 95% CI and BH-FDR q-values"],
        ["Main", "Figure_BestModel_BrainAge_Scatter.png", "Primary raw OOF actual-vs-predicted Brain Age for best scale"],
        ["Main/Supp", "Figure_BestModel_CorrectedBAG_vs_Age.png", "Residual age bias after correction"],
        ["Main/Supp", "Figure_Top_OOF_Permutation_Importance.png", "Primary OOF predictive importance"],
        ["Main/Supp", "Figure_Feature_Selection_Stability.png", "Outer-fold feature-selection stability"],
        ["Main/Supp", "Figure_Phenotype_x_Scale_Importance_Heatmap.png", "Phenotype by anatomical-scale decomposition"],
        ["Supplementary", "Figure_Final_ElasticNet_TopCoefficients.png", "Largest standardized final Elastic Net coefficients"],
        ["Supplementary", "Figure_Final_SHAP_Beeswarm.png", "Descriptive Linear SHAP beeswarm when ElasticNet wins"],
        ["Supplementary", "Figure_Final_SHAP_Bar.png", "Descriptive SHAP global importance"],
        ["Supplementary", "Supplementary_SHAP_Waterfall_*.png", "Illustrative individual SHAP explanations"],
        ["Supplementary", "Figure_BestModel_Raw_Calibration.png", "Calibration diagnostic"],
        ["Supplementary", "Figure_BestModel_BlandAltman.png", "Bland-Altman diagnostic"],
        ["Supplementary", "Figure_BestModel_Raw_ErrorDistribution.png", "Raw OOF error distribution"],
        ["Supplementary", "Figure_Model_Selection_Frequency.png", "Algorithm selection frequency across outer folds"],
        ["Supplementary", "Supplementary_Nested_Learning_Curves.png", "Nested-subsample learning curves: in-sample training MAE ±1 outer-fit SD and outer-validation MAE with conditional participant-bootstrap 95% CI"],
        ["Supplementary", "Supplementary_Sex_Bias_OOF.png", "Sex-stratified raw OOF MAE and corrected mean BAG with 95% bootstrap CIs"],
    ], columns=["Role", "File", "Purpose"])
    figures_manifest.to_excel(
        os.path.join(OUTPUT_DIR, "Figure_Manifest.xlsx"),
        index=False,
    )

    print("\n" + "=" * 78)
    print("DONE")
    print("=" * 78)
    print("Main results:")
    print(result_xlsx)
    print("Best scale:", best_fs)
    print("Output directory:", OUTPUT_DIR)
    print("\nIMPORTANT:")
    print(
        "Interpret Brain Age as age estimated from brain morphology. "
        "Do NOT interpret cross-sectional BAG as an individual's true "
        "rate of biological aging."
    )


if __name__ == "__main__":
    main()
