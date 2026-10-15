"""
Insurance Claims Triage — Final end-to-end analysis
===================================================
Run from the project root:
    python src/final_insurance_claims_pipeline.py

The script is deliberately defensive:
- Finds the project root from this file, or from the current working directory.
- Uses existing train.csv/test.csv when both are present; otherwise splits claims_clean.csv.
- Does not overwrite source data.
- Produces EDA tables, professional plots, a 10-model comparison, imbalance diagnostics,
  leakage/feature exclusion audits, a final Markdown report, and a JSON summary.
- Handles class imbalance only when the target is meaningfully imbalanced, using
  class_weight="balanced" where supported (no blind SMOTE).
- Trains on a stratified cap if the training set is very large; evaluates on the full test set.
- Keeps identifier and payment-derived fields out of the model by default.
"""

from __future__ import annotations

import json
import math
import os
import sys
import time
import warnings
from pathlib import Path
from typing import Any

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# Keep numerical libraries from creating too many threads on modest Windows machines.
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

try:
    import numpy as np
    import pandas as pd
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    from sklearn.base import clone
    from sklearn.compose import ColumnTransformer
    from sklearn.dummy import DummyClassifier
    from sklearn.ensemble import (
        AdaBoostClassifier,
        ExtraTreesClassifier,
        GradientBoostingClassifier,
        HistGradientBoostingClassifier,
        RandomForestClassifier,
        RandomForestRegressor,
        VotingClassifier,
    )
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression, SGDClassifier, RidgeClassifier, Perceptron
    from sklearn.metrics import (
        accuracy_score, balanced_accuracy_score, classification_report,
        confusion_matrix, f1_score, precision_score, recall_score,
        roc_auc_score, average_precision_score, matthews_corrcoef,
    )
    from sklearn.model_selection import train_test_split
    from sklearn.naive_bayes import GaussianNB
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler, OrdinalEncoder
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.svm import LinearSVC
except Exception as exc:
    print("\nMISSING OR BROKEN PYTHON PACKAGE")
    print("Run: python -m pip install -r requirements.txt")
    print("If that fails, run: python -m pip install pandas numpy matplotlib seaborn scikit-learn scipy")
    raise SystemExit(f"Import error: {exc}")


PROJECT_NAME = "Insurance Claims Triage"
TARGET_CANDIDATES = ["is_inpatient", "target", "TARGET", "label", "class"]
ID_COLUMNS = ["DESYNPUF_ID", "CLM_ID", "claim_id", "beneficiary_id", "id"]
PAYMENT_COLUMNS = [
    "MEDREIMB_IP", "BENRES_IP", "PPPYMT_IP",
    "MEDREIMB_OP", "BENRES_OP", "PPPYMT_OP",
    "MEDREIMB_CAR", "BENRES_CAR", "PPPYMT_CAR",
    "CLM_PMT_AMT", "NCH_PRMRY_PYR_CLM_PD_AMT",
    "NCH_BENE_BLOOD_DDCTBL_LBLTY_AM",
]
# Fields that can encode a result rather than information available at the decision point.
# claim_duration_days is excluded by default because it may be unavailable before claim
# classification and may nearly reveal the inpatient label.
SUSPICIOUS_POST_OUTCOME_COLUMNS = ["claim_duration_days"]
MAX_TRAIN_ROWS = 100_000
RANDOM_STATE = 42
MIN_IMBALANCE_RATIO = 1.5
PLOT_DPI = 180


def project_root() -> Path:
    here = Path(__file__).resolve()
    candidates = [here.parent, here.parent.parent, Path.cwd()]
    for candidate in candidates:
        if (candidate / "data").is_dir() and (candidate / "reports").exists():
            return candidate
    for candidate in candidates:
        if (candidate / "data").is_dir():
            return candidate
    return Path.cwd()


ROOT = project_root()
DATA_DIR = ROOT / "data"
REPORTS_DIR = ROOT / "reports"
PLOTS_DIR = REPORTS_DIR / "final_plots"
MODELS_DIR = REPORTS_DIR / "final_models"
TABLES_DIR = REPORTS_DIR / "final_tables"
for folder in (REPORTS_DIR, PLOTS_DIR, MODELS_DIR, TABLES_DIR):
    folder.mkdir(parents=True, exist_ok=True)


def log(message: str) -> None:
    print(f"[Insurance Claims Triage] {message}", flush=True)


def safe_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): safe_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [safe_json(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, Path):
        return str(value)
    if pd.isna(value) if not isinstance(value, (dict, list, tuple, np.ndarray)) else False:
        return None
    return value


def save_csv(frame: pd.DataFrame, path: Path) -> None:
    frame.to_csv(path, index=False, encoding="utf-8-sig")


def read_csv_robust(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Required data file not found: {path}")
    # The dataset is CSV and may be hundreds of MB. Low-memory mode avoids dtype warnings.
    try:
        return pd.read_csv(path, low_memory=False)
    except UnicodeDecodeError:
        return pd.read_csv(path, low_memory=False, encoding="latin-1")


def find_target(df: pd.DataFrame) -> str:
    for col in TARGET_CANDIDATES:
        if col in df.columns:
            return col
    raise ValueError(
        "Could not find target column. Expected one of: "
        + ", ".join(TARGET_CANDIDATES)
        + f". Available columns: {list(df.columns)}"
    )


def normalise_target(y: pd.Series) -> pd.Series:
    y = y.copy()
    if y.isna().any():
        raise ValueError(f"Target column has {int(y.isna().sum())} missing rows. Fix target labels first.")
    # Convert common text labels to stable numeric classes.
    if not pd.api.types.is_numeric_dtype(y):
        vals = y.astype(str).str.strip()
        unique = list(pd.unique(vals))
        if len(unique) != 2:
            raise ValueError(f"This script expects a binary classification target; found {len(unique)} classes: {unique[:20]}")
        mapping = {value: idx for idx, value in enumerate(sorted(unique))}
        return vals.map(mapping).astype(int)
    unique = sorted(pd.unique(y))
    if len(unique) != 2:
        raise ValueError(f"This script expects a binary target; found {len(unique)} classes: {unique[:20]}")
    # Keep 0/1 labels as-is; otherwise map sorted classes to 0/1.
    if set(unique) == {0, 1}:
        return y.astype(int)
    mapping = {value: idx for idx, value in enumerate(unique)}
    return y.map(mapping).astype(int)


def describe_dataset(df: pd.DataFrame, name: str) -> dict[str, Any]:
    rows, cols = df.shape
    missing_cells = int(df.isna().sum().sum())
    duplicated_rows = int(df.duplicated().sum())
    return {
        "dataset": name,
        "rows": int(rows),
        "columns": int(cols),
        "duplicate_rows": duplicated_rows,
        "missing_cells": missing_cells,
        "missing_cell_percent": round(100 * missing_cells / max(rows * cols, 1), 4),
        "memory_mb_approx": round(float(df.memory_usage(deep=True).sum()) / (1024 ** 2), 2),
    }


def create_eda(df: pd.DataFrame, target: str) -> dict[str, Any]:
    log("Generating data inventory, data quality checks, descriptive statistics and EDA plots...")
    # Inventory / data quality
    inventory = pd.DataFrame({
        "feature": df.columns,
        "dtype": [str(df[c].dtype) for c in df.columns],
        "non_null_count": [int(df[c].notna().sum()) for c in df.columns],
        "missing_count": [int(df[c].isna().sum()) for c in df.columns],
        "missing_percent": [round(100 * float(df[c].isna().mean()), 4) for c in df.columns],
        "unique_count": [int(df[c].nunique(dropna=True)) for c in df.columns],
        "duplicate_values_possible": [bool(df[c].nunique(dropna=True) < len(df)) for c in df.columns],
    })
    save_csv(inventory, TABLES_DIR / "data_inventory.csv")

    missing = inventory[["feature", "missing_count", "missing_percent"]].sort_values(
        "missing_percent", ascending=False
    )
    save_csv(missing, TABLES_DIR / "missingness_report.csv")

    numeric = df.select_dtypes(include=[np.number]).copy()
    if not numeric.empty:
        desc = numeric.describe(percentiles=[.01, .05, .25, .5, .75, .95, .99]).T
        desc["skewness"] = numeric.skew(numeric_only=True)
        desc["kurtosis_excess"] = numeric.kurtosis(numeric_only=True)
        desc["zero_count"] = (numeric == 0).sum()
        desc["negative_count"] = (numeric < 0).sum()
        desc["infinite_count"] = np.isinf(numeric).sum()
        save_csv(desc.rename_axis("feature").reset_index(), TABLES_DIR / "numeric_descriptive_statistics_skew_kurtosis.csv")

        skew_kurt = pd.DataFrame({
            "feature": numeric.columns,
            "skewness": numeric.skew(numeric_only=True).values,
            "kurtosis_excess": numeric.kurtosis(numeric_only=True).values,
            "mean": numeric.mean(numeric_only=True).values,
            "median": numeric.median(numeric_only=True).values,
            "std": numeric.std(numeric_only=True).values,
        })
        save_csv(skew_kurt.sort_values("skewness", key=lambda s: s.abs(), ascending=False),
                 TABLES_DIR / "skewness_kurtosis_all_numeric_features.csv")

        # Outliers using IQR fence; counts only, no automatic removal.
        out_rows = []
        for col in numeric.columns:
            series = numeric[col].replace([np.inf, -np.inf], np.nan).dropna()
            if series.empty:
                continue
            q1, q3 = series.quantile([.25, .75])
            iqr = q3 - q1
            low, high = q1 - 1.5 * iqr, q3 + 1.5 * iqr
            out_rows.append({
                "feature": col, "q1": q1, "q3": q3, "iqr": iqr,
                "lower_fence": low, "upper_fence": high,
                "outlier_count": int(((series < low) | (series > high)).sum()),
                "outlier_percent": round(100 * float(((series < low) | (series > high)).mean()), 4),
            })
        save_csv(pd.DataFrame(out_rows), TABLES_DIR / "iqr_outlier_audit.csv")

        # Correlation matrix and high-correlation pairs.
        corr = numeric.corr()
        corr.to_csv(TABLES_DIR / "numeric_correlation_matrix.csv")
        pairs = []
        cols = list(corr.columns)
        for i in range(len(cols)):
            for j in range(i + 1, len(cols)):
                val = corr.iloc[i, j]
                if pd.notna(val) and abs(val) >= 0.85:
                    pairs.append({"feature_1": cols[i], "feature_2": cols[j], "correlation": float(val)})
        save_csv(pd.DataFrame(pairs, columns=["feature_1", "feature_2", "correlation"]),
                 TABLES_DIR / "high_correlation_pairs_abs_0_85.csv")

    # Professional, consistently formatted plots.
    sns.set_theme(style="whitegrid", context="notebook", font_scale=0.9)

    # 1: Target balance
    plt.figure(figsize=(8, 5))
    counts = df[target].value_counts(dropna=False).sort_index()
    ax = sns.barplot(x=counts.index.astype(str), y=counts.values, color="#2878B5")
    ax.set(title=f"Target Class Distribution — {target}", xlabel="Class", ylabel="Number of records")
    for container in ax.containers:
        ax.bar_label(container, fmt="%.0f")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "01_target_class_distribution.png", dpi=PLOT_DPI)
    plt.close()

    # 2: missingness
    miss_plot = inventory[inventory["missing_percent"] > 0].sort_values("missing_percent", ascending=False).head(25)
    if not miss_plot.empty:
        plt.figure(figsize=(10, max(5, 0.32 * len(miss_plot))))
        ax = sns.barplot(data=miss_plot, y="feature", x="missing_percent", color="#2878B5")
        ax.set(title="Top Features by Missingness", xlabel="Missing values (%)", ylabel="")
        plt.tight_layout()
        plt.savefig(PLOTS_DIR / "02_top_missingness.png", dpi=PLOT_DPI, bbox_inches="tight")
        plt.close()

    # 3: numeric distributions, cap plots to top 12 numeric columns by variance.
    if len(numeric.columns):
        var_cols = numeric.var(numeric_only=True).sort_values(ascending=False).head(12).index.tolist()
        ncols = 3
        nrows = math.ceil(len(var_cols) / ncols)
        fig, axes = plt.subplots(nrows, ncols, figsize=(15, 4 * nrows))
        axes = np.array(axes).reshape(-1)
        for ax, col in zip(axes, var_cols):
            values = numeric[col].replace([np.inf, -np.inf], np.nan).dropna()
            if len(values) > 100_000:
                values = values.sample(100_000, random_state=RANDOM_STATE)
            sns.histplot(values, kde=(len(values) > 20 and values.nunique() > 10), ax=ax, bins=40, color="#2878B5")
            ax.set_title(f"Distribution: {col}")
        for ax in axes[len(var_cols):]:
            ax.axis("off")
        fig.suptitle("Numeric Feature Distributions (highest-variance features)", y=1.01, fontsize=14)
        fig.tight_layout()
        fig.savefig(PLOTS_DIR / "03_numeric_distributions.png", dpi=PLOT_DPI, bbox_inches="tight")
        plt.close(fig)

        # 4: skewness
        sk = numeric.skew(numeric_only=True).dropna().sort_values()
        if len(sk):
            show = pd.concat([sk.head(10), sk.tail(10)]).drop_duplicates()
            plt.figure(figsize=(10, max(5, len(show) * .28)))
            sns.barplot(x=show.values, y=show.index, color="#2878B5")
            plt.axvline(0, linewidth=1)
            plt.title("Skewness by Numeric Feature (most negative and positive)")
            plt.xlabel("Skewness")
            plt.ylabel("")
            plt.tight_layout()
            plt.savefig(PLOTS_DIR / "04_feature_skewness.png", dpi=PLOT_DPI, bbox_inches="tight")
            plt.close()

        # 5: correlation heatmap (top 25 numeric features by variance)
        heat_cols = numeric.var(numeric_only=True).sort_values(ascending=False).head(25).index.tolist()
        if len(heat_cols) >= 2:
            plt.figure(figsize=(14, 11))
            sns.heatmap(numeric[heat_cols].corr(), cmap="vlag", center=0, square=False)
            plt.title("Correlation Heatmap — Top Numeric Features")
            plt.tight_layout()
            plt.savefig(PLOTS_DIR / "05_correlation_heatmap.png", dpi=PLOT_DPI, bbox_inches="tight")
            plt.close()

    # 6: Target rate by selected low-cardinality categorical features.
    cat_cols = [c for c in df.columns if c != target and (df[c].dtype == "object" or str(df[c].dtype) == "category")
                and df[c].nunique(dropna=True) <= 20]
    if not cat_cols:
        cat_cols = [c for c in df.columns if c != target and df[c].nunique(dropna=True) <= 12][:6]
    for idx, col in enumerate(cat_cols[:6], start=6):
        tmp = df[[col, target]].copy()
        tmp[col] = tmp[col].astype("string").fillna("Missing")
        rate = tmp.groupby(col, observed=False)[target].agg(["mean", "count"]).reset_index()
        rate = rate[rate["count"] >= max(20, int(len(tmp) * 0.001))]
        if rate.empty:
            continue
        rate = rate.sort_values("mean", ascending=False).head(20)
        plt.figure(figsize=(10, max(4.5, len(rate) * .28)))
        ax = sns.barplot(data=rate, x="mean", y=col, color="#2878B5")
        ax.set(title=f"Target Rate by {col}", xlabel=f"Mean target value (class rate)", ylabel="")
        ax.set_xlim(0, 1)
        plt.tight_layout()
        plt.savefig(PLOTS_DIR / f"{idx:02d}_target_rate_by_{col[:40]}.png", dpi=PLOT_DPI, bbox_inches="tight")
        plt.close()

    # Target statistics
    target_counts = df[target].value_counts().sort_index()
    total = max(int(target_counts.sum()), 1)
    minority_count = int(target_counts.min())
    majority_count = int(target_counts.max())
    imbalance_ratio = float(majority_count / max(minority_count, 1))
    target_summary = {
        "target": target,
        "counts": {str(k): int(v) for k, v in target_counts.items()},
        "percentages": {str(k): round(100 * float(v) / total, 4) for k, v in target_counts.items()},
        "majority_to_minority_ratio": round(imbalance_ratio, 4),
        "imbalance_threshold_ratio": MIN_IMBALANCE_RATIO,
        "imbalance_handling_recommended": bool(imbalance_ratio >= MIN_IMBALANCE_RATIO),
    }
    with open(TABLES_DIR / "target_balance_summary.json", "w", encoding="utf-8") as f:
        json.dump(target_summary, f, indent=2)

    # Feature types and unique counts
    feature_summary = {
        "total_columns_in_eda_data": int(df.shape[1]),
        "target_column": target,
        "numeric_columns_including_target": int(len(df.select_dtypes(include=[np.number]).columns)),
        "categorical_columns": int(df.select_dtypes(exclude=[np.number]).shape[1]),
        "duplicate_rows": int(df.duplicated().sum()),
    }
    return {"target_balance": target_summary, "feature_summary": feature_summary}


def prepare_feature_matrices(
    X_train: pd.DataFrame, X_test: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, list[str], list[str]]:
    """Normalize feature dtypes safely, including pandas 3 StringDtype columns.

    Feature-type decisions are learned from training data only. Any text field (such as
    dx_chapter containing values like 'Musculoskeletal') is categorical, never sent to
    the median numeric imputer. Numeric-looking text columns are converted to numeric
    only when almost all non-missing training values parse successfully.
    """
    train = X_train.copy()
    test = X_test.copy()
    numeric_cols: list[str] = []
    categorical_cols: list[str] = []
    always_categorical = {
        "dx_chapter", "SP_STATE_CODE", "BENE_SEX_IDENT_CD", "BENE_RACE_CD", "BENE_ESRD_IND"
    }

    for col in train.columns:
        tr = train[col]
        te = test[col] if col in test.columns else pd.Series(np.nan, index=test.index)
        non_missing = tr.dropna()
        forced_cat = col.upper() in {x.upper() for x in always_categorical}
        is_bool = pd.api.types.is_bool_dtype(tr.dtype)
        native_numeric = pd.api.types.is_numeric_dtype(tr.dtype) and not is_bool

        numeric_parse_ratio = 0.0
        parsed_train = None
        if len(non_missing):
            parsed_non_missing = pd.to_numeric(non_missing, errors="coerce")
            numeric_parse_ratio = float(parsed_non_missing.notna().mean())

        # Coded demographics/conditions with low cardinality are categories, not measurements.
        low_cardinality_code = (
            col.upper().startswith(("BENE_", "SP_", "DX_", "STATE", "SEX", "RACE"))
            and 1 < tr.nunique(dropna=True) <= 12
        )
        treat_as_numeric = (
            not forced_cat and not low_cardinality_code and
            (native_numeric or (len(non_missing) > 0 and numeric_parse_ratio >= 0.98))
        )

        if treat_as_numeric:
            train[col] = pd.to_numeric(tr, errors="coerce")
            test[col] = pd.to_numeric(te, errors="coerce")
            numeric_cols.append(col)
        else:
            # Converting nonmissing category values to plain strings avoids mixed-type
            # category sorting issues. Missing values remain np.nan for SimpleImputer.
            train[col] = tr.map(lambda value: np.nan if pd.isna(value) else str(value)).astype(object)
            test[col] = te.map(lambda value: np.nan if pd.isna(value) else str(value)).astype(object)
            categorical_cols.append(col)

    return train, test, numeric_cols, categorical_cols


def make_preprocessor(
    numeric_cols: list[str], categorical_cols: list[str]
) -> ColumnTransformer:
    """Build preprocessing explicitly from the already-normalized column lists."""
    # OneHotEncoder API compatibility with older sklearn releases.
    try:
        onehot = OneHotEncoder(handle_unknown="ignore", min_frequency=10, sparse_output=True)
    except TypeError:
        onehot = OneHotEncoder(handle_unknown="ignore", sparse=True)

    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler(with_mean=False)),
    ])
    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", onehot),
    ])
    transformers = []
    if numeric_cols:
        transformers.append(("numeric", numeric_pipe, numeric_cols))
    if categorical_cols:
        transformers.append(("categorical", categorical_pipe, categorical_cols))
    if not transformers:
        raise ValueError("No usable numeric or categorical predictors were found.")
    return ColumnTransformer(transformers=transformers, remainder="drop", sparse_threshold=0.3)


def make_models(imbalance_needed: bool) -> list[tuple[str, Any]]:
    cw = "balanced" if imbalance_needed else None
    return [
        ("Dummy_Baseline", DummyClassifier(strategy="most_frequent")),
        ("Logistic_Regression", LogisticRegression(
            max_iter=700, class_weight=cw, solver="liblinear", random_state=RANDOM_STATE
        )),
        ("SGD_Classifier", SGDClassifier(
            loss="log_loss", max_iter=1000, tol=1e-3, class_weight=cw, random_state=RANDOM_STATE
        )),
        ("Decision_Tree", DecisionTreeClassifier(
            max_depth=18, min_samples_leaf=5, class_weight=cw, random_state=RANDOM_STATE
        )),
        ("Random_Forest", RandomForestClassifier(
            n_estimators=60, max_depth=22, min_samples_leaf=3, max_features="sqrt",
            class_weight=cw, n_jobs=2, random_state=RANDOM_STATE
        )),
        ("Extra_Trees", ExtraTreesClassifier(
            n_estimators=60, max_depth=22, min_samples_leaf=3, max_features="sqrt",
            class_weight=cw, n_jobs=2, random_state=RANDOM_STATE
        )),
        ("Ridge_Classifier", RidgeClassifier(class_weight=cw)),
        ("AdaBoost", AdaBoostClassifier(
            n_estimators=70, learning_rate=0.5, random_state=RANDOM_STATE
        )),
        ("Gaussian_NB", GaussianNB()),
        ("Linear_SVM", LinearSVC(
            class_weight=cw, random_state=RANDOM_STATE, max_iter=1200, tol=1e-2
        )),
    ]


def score_model(model: Any, X_test: Any, y_test: pd.Series) -> tuple[dict[str, Any], np.ndarray]:
    pred = model.predict(X_test)
    probs = None
    if hasattr(model, "predict_proba"):
        try:
            probs = model.predict_proba(X_test)[:, 1]
        except Exception:
            probs = None
    elif hasattr(model, "decision_function"):
        try:
            raw = model.decision_function(X_test)
            # Monotonic scaling for AUC/AP only; these are scores, not calibrated probabilities.
            probs = raw
        except Exception:
            probs = None

    metrics = {
        "accuracy": float(accuracy_score(y_test, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_test, pred)),
        "precision": float(precision_score(y_test, pred, zero_division=0)),
        "recall": float(recall_score(y_test, pred, zero_division=0)),
        "f1": float(f1_score(y_test, pred, zero_division=0)),
        "matthews_corrcoef": float(matthews_corrcoef(y_test, pred)),
        "roc_auc": float(roc_auc_score(y_test, probs)) if probs is not None and len(np.unique(y_test)) == 2 else float("nan"),
        "average_precision_pr_auc": float(average_precision_score(y_test, probs)) if probs is not None and len(np.unique(y_test)) == 2 else float("nan"),
    }
    return metrics, pred


def load_train_test() -> tuple[pd.DataFrame, pd.DataFrame, str, str]:
    train_path = DATA_DIR / "train.csv"
    test_path = DATA_DIR / "test.csv"
    clean_candidates = [
        DATA_DIR / "claims_clean.csv",
        DATA_DIR / "merged_claims_sample1.csv",
        ROOT / "claims_clean.csv",
    ]
    if train_path.exists() and test_path.exists():
        log("Using existing data/train.csv and data/test.csv.")
        train_df = read_csv_robust(train_path)
        test_df = read_csv_robust(test_path)
        target = find_target(train_df)
        if target not in test_df.columns:
            raise ValueError(f"Target {target!r} exists in train.csv but not in test.csv.")
        return train_df, test_df, target, "existing train.csv/test.csv"

    clean_path = next((p for p in clean_candidates if p.exists()), None)
    if clean_path is None:
        raise FileNotFoundError(
            "Could not find train.csv + test.csv or claims_clean.csv. "
            f"Expected files under {DATA_DIR}"
        )
    log(f"Splitting {clean_path.name} into train/test (80/20, stratified).")
    full = read_csv_robust(clean_path)
    target = find_target(full)
    y = normalise_target(full[target])
    full[target] = y
    train_df, test_df = train_test_split(
        full, test_size=0.2, random_state=RANDOM_STATE, stratify=full[target]
    )
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)
    return train_df.reset_index(drop=True), test_df.reset_index(drop=True), target, f"stratified 80/20 split from {clean_path.name}"


def build_feature_plan(train_df: pd.DataFrame, test_df: pd.DataFrame, target: str) -> tuple[pd.DataFrame, pd.DataFrame, list[str], pd.DataFrame]:
    # The target is never a predictor. Remove identifiers and direct claim payment fields to
    # reduce leakage and prevent the model from using downstream financial outcomes.
    all_columns = list(train_df.columns)
    excluded_reasons: dict[str, str] = {}
    for col in all_columns:
        if col == target:
            excluded_reasons[col] = "Target label; never a predictor"
        elif col in ID_COLUMNS or col.lower().endswith("_id"):
            excluded_reasons[col] = "Identifier; high-cardinality / memorisation risk"
        elif col in PAYMENT_COLUMNS:
            excluded_reasons[col] = "Claim payment/reimbursement; potential target leakage or post-decision information"
        elif col in SUSPICIOUS_POST_OUTCOME_COLUMNS:
            excluded_reasons[col] = "Potential post-outcome field; excluded to reduce temporal leakage"
    # Always intersect the same columns from train/test and exclude the target.
    feature_cols = [c for c in train_df.columns if c in test_df.columns and c not in excluded_reasons]
    # Drop all-null training columns and constants; record each decision.
    for col in list(feature_cols):
        if train_df[col].isna().all():
            excluded_reasons[col] = "All values missing in training data"
            feature_cols.remove(col)
        elif train_df[col].nunique(dropna=True) <= 1:
            excluded_reasons[col] = "Constant feature; adds no predictive information"
            feature_cols.remove(col)
    if not feature_cols:
        raise ValueError("No usable predictors remain after leakage/identifier exclusions.")

    X_train = train_df[feature_cols].copy()
    X_test = test_df[feature_cols].copy()
    y_train = normalise_target(train_df[target])
    y_test = normalise_target(test_df[target])
    if set(y_train.unique()) != set(y_test.unique()):
        raise ValueError("Train/test target classes do not match. Please verify the split.")

    exclusion_table = pd.DataFrame(
        [{"feature": col, "decision": "EXCLUDED", "reason": reason} for col, reason in excluded_reasons.items()]
        + [{"feature": col, "decision": "INCLUDED", "reason": "Available at prediction time (verify business timing)"} for col in feature_cols]
    )
    save_csv(exclusion_table, TABLES_DIR / "feature_inclusion_exclusion_audit.csv")
    save_csv(pd.DataFrame({"feature": feature_cols}), TABLES_DIR / "model_feature_list.csv")

    excluded_rows = exclusion_table[exclusion_table["decision"] == "EXCLUDED"]
    feature_counts = {
        "project_name": PROJECT_NAME,
        "total_input_columns": int(len(train_df.columns)),
        "target_column": target,
        "target_columns_excluded_from_predictors": int((excluded_rows["reason"] == "Target label; never a predictor").sum()),
        "identifier_columns_excluded": int(excluded_rows["reason"].str.startswith("Identifier").sum()),
        "payment_columns_excluded": int(excluded_rows["reason"].str.startswith("Claim payment").sum()),
        "post_outcome_columns_excluded": int(excluded_rows["reason"].str.startswith("Potential post-outcome").sum()),
        "empty_or_constant_columns_excluded": int(excluded_rows["reason"].str.startswith(("All values missing", "Constant feature")).sum()),
        "other_columns_excluded": int(len(excluded_rows) - (
            (excluded_rows["reason"] == "Target label; never a predictor").sum()
            + excluded_rows["reason"].str.startswith("Identifier").sum()
            + excluded_rows["reason"].str.startswith("Claim payment").sum()
            + excluded_rows["reason"].str.startswith("Potential post-outcome").sum()
            + excluded_rows["reason"].str.startswith(("All values missing", "Constant feature")).sum()
        )),
        "total_excluded_columns_including_target": int(len(excluded_rows)),
        "retained_predictor_count": int(len(feature_cols)),
        "retained_predictor_names": feature_cols,
        "excluded_column_names": excluded_rows["feature"].tolist(),
    }
    with open(TABLES_DIR / "feature_count_summary.json", "w", encoding="utf-8") as f:
        json.dump(safe_json(feature_counts), f, indent=2)
    save_csv(pd.DataFrame([{
        k: (", ".join(v) if isinstance(v, list) else v)
        for k, v in feature_counts.items()
    }]), TABLES_DIR / "feature_count_summary.csv")
    log(f"Feature audit: {len(train_df.columns)} input columns; {len(feature_cols)} retained predictors; {len(excluded_rows)} excluded columns including target.")
    return X_train.assign(**{target: y_train}), X_test.assign(**{target: y_test}), feature_cols, exclusion_table


def run_models(train_df: pd.DataFrame, test_df: pd.DataFrame, target: str, feature_cols: list[str], balance_summary: dict[str, Any]) -> dict[str, Any]:
    log("Preparing 10 classifiers. Imbalance handling is enabled only when the class ratio meets the configured threshold.")
    X_train = train_df[feature_cols].copy()
    y_train = normalise_target(train_df[target])
    X_test = test_df[feature_cols].copy()
    y_test = normalise_target(test_df[target])

    # Normalize pandas/Python dtypes before sampling or building the imputation pipeline.
    X_train, X_test, numeric_cols, categorical_cols = prepare_feature_matrices(X_train, X_test)
    log(f"Feature typing: {len(numeric_cols)} numeric, {len(categorical_cols)} categorical predictors.")

    # Avoid memory overload on very large training datasets. Stratified sample preserves target proportions.
    train_cap = min(MAX_TRAIN_ROWS, len(X_train))
    if len(X_train) > train_cap:
        X_train, _, y_train, _ = train_test_split(
            X_train, y_train, train_size=train_cap, random_state=RANDOM_STATE, stratify=y_train
        )
        log(f"Training cap applied: {train_cap:,} stratified training rows; full test set retained.")
    else:
        log(f"Training on all {len(X_train):,} training rows; testing on all {len(X_test):,} test rows.")

    counts = y_train.value_counts()
    imbalance_ratio = float(counts.max() / max(counts.min(), 1))
    imbalance_needed = imbalance_ratio >= MIN_IMBALANCE_RATIO
    preprocessor = make_preprocessor(numeric_cols, categorical_cols)
    save_csv(pd.DataFrame({
        "feature": numeric_cols + categorical_cols,
        "preprocessing_type": ["median_imputation_and_scaling"] * len(numeric_cols)
                              + ["mode_imputation_and_one_hot_encoding"] * len(categorical_cols),
    }), TABLES_DIR / "preprocessing_plan.csv")

    rows = []
    predictions = {}
    reports = {}
    fitted = {}
    for name, estimator in make_models(imbalance_needed):
        log(f"Training {name}...")
        start = time.time()
        try:
            # GaussianNB requires dense input. Use a dense transformed matrix only for this model,
            # with a bounded row cap to avoid exhausting RAM.
            if name == "Gaussian_NB":
                prep = clone(preprocessor)
                Xtr = prep.fit_transform(X_train)
                Xte = prep.transform(X_test)
                if hasattr(Xtr, "toarray"):
                    # Bound dense conversion to keep RAM predictable; if too wide, skip safely.
                    if Xtr.shape[0] * Xtr.shape[1] > 20_000_000:
                        raise MemoryError("GaussianNB dense conversion would exceed safe memory limit.")
                    Xtr = Xtr.toarray()
                    Xte = Xte.toarray()
                estimator.fit(Xtr, y_train)
                metrics, pred = score_model(estimator, Xte, y_test)
                fitted[name] = (estimator, prep, True)
            else:
                pipe = Pipeline([("preprocess", clone(preprocessor)), ("model", estimator)])
                pipe.fit(X_train, y_train)
                metrics, pred = score_model(pipe, X_test, y_test)
                fitted[name] = (pipe, None, False)

            elapsed = round(time.time() - start, 2)
            row = {"model": name, **metrics, "training_seconds": elapsed, "status": "OK"}
            rows.append(row)
            predictions[name] = pred
            reports[name] = classification_report(y_test, pred, output_dict=True, zero_division=0)
            with open(MODELS_DIR / f"{name.lower()}_classification_report.json", "w", encoding="utf-8") as f:
                json.dump(safe_json(reports[name]), f, indent=2)
            cm = confusion_matrix(y_test, pred, labels=[0, 1])
            save_csv(pd.DataFrame(cm, index=["actual_0", "actual_1"], columns=["predicted_0", "predicted_1"]).reset_index(names="actual"),
                     MODELS_DIR / f"{name.lower()}_confusion_matrix.csv")
        except Exception as exc:
            rows.append({
                "model": name, "accuracy": np.nan, "balanced_accuracy": np.nan,
                "precision": np.nan, "recall": np.nan, "f1": np.nan,
                "matthews_corrcoef": np.nan, "roc_auc": np.nan,
                "average_precision_pr_auc": np.nan, "training_seconds": np.nan,
                "status": f"FAILED SAFELY: {type(exc).__name__}: {exc}",
            })
            log(f"{name} could not complete; continuing to next model. Reason: {type(exc).__name__}: {exc}")
        # Persist every iteration so a long or interrupted run never loses completed model results.
        save_csv(pd.DataFrame(rows), REPORTS_DIR / "final_model_comparison_progress.csv")

    comparison = pd.DataFrame(rows)
    if comparison.empty or not (comparison["status"] == "OK").any():
        save_csv(comparison, REPORTS_DIR / "final_model_comparison.csv")
        raise RuntimeError("No models completed successfully. Check the error statuses in final_model_comparison.csv.")
    comparison = comparison.sort_values(["f1", "average_precision_pr_auc"], ascending=False, na_position="last")
    save_csv(comparison, REPORTS_DIR / "final_model_comparison.csv")
    save_csv(comparison, TABLES_DIR / "final_model_comparison.csv")

    # Professional metric comparison plots.
    ok = comparison[comparison["status"] == "OK"].copy()
    for metric in ["f1", "recall", "precision", "balanced_accuracy", "roc_auc", "average_precision_pr_auc"]:
        valid = ok.dropna(subset=[metric]).sort_values(metric, ascending=False)
        if valid.empty:
            continue
        plt.figure(figsize=(11, max(5, len(valid) * .42)))
        ax = sns.barplot(data=valid, x=metric, y="model", color="#2878B5")
        ax.set_title(f"Model Comparison — {metric.replace('_', ' ').title()}")
        ax.set_xlabel(metric.replace("_", " ").title())
        ax.set_ylabel("")
        if metric != "matthews_corrcoef":
            ax.set_xlim(0, max(1.0, float(valid[metric].max()) * 1.05))
        plt.tight_layout()
        plt.savefig(PLOTS_DIR / f"model_comparison_{metric}.png", dpi=PLOT_DPI, bbox_inches="tight")
        plt.close()

    best_row = ok.sort_values(["f1", "average_precision_pr_auc"], ascending=False, na_position="last").iloc[0]
    best_name = str(best_row["model"])
    best_pred = predictions[best_name]
    cm = confusion_matrix(y_test, best_pred, labels=[0, 1])
    plt.figure(figsize=(6.5, 5.5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Predicted 0", "Predicted 1"], yticklabels=["Actual 0", "Actual 1"])
    plt.title(f"Best Model Confusion Matrix — {best_name}")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "best_model_confusion_matrix.png", dpi=PLOT_DPI)
    plt.close()

    summary = {
        "target": target,
        "train_rows_used": int(len(X_train)),
        "test_rows_evaluated": int(len(X_test)),
        "predictor_count": int(len(feature_cols)),
        "imbalance_ratio_in_model_training_data": round(imbalance_ratio, 4),
        "imbalance_handling_used": bool(imbalance_needed),
        "imbalance_method": "class_weight='balanced' for supported models; no resampling" if imbalance_needed else "No class weighting; imbalance ratio below threshold",
        "best_model_by_f1_then_pr_auc": best_name,
        "best_model_metrics": {k: safe_json(best_row[k]) for k in ["accuracy", "balanced_accuracy", "precision", "recall", "f1", "matthews_corrcoef", "roc_auc", "average_precision_pr_auc"]},
        "models_attempted": int(len(rows)),
        "models_successful": int((comparison["status"] == "OK").sum()),
        "models": safe_json(comparison.to_dict(orient="records")),
    }
    with open(MODELS_DIR / "final_run_summary.json", "w", encoding="utf-8") as f:
        json.dump(safe_json(summary), f, indent=2)
    return summary


def check_beneficiary_overlap(train_df: pd.DataFrame, test_df: pd.DataFrame) -> dict[str, Any]:
    id_col = next((c for c in ID_COLUMNS if c in train_df.columns and c in test_df.columns), None)
    if id_col is None:
        return {"checked": False, "reason": "No beneficiary/identifier column available in both splits."}
    # DESYNPUF_ID is the beneficiary identifier. CLM_ID is claim-level and is not an overlap check.
    if id_col == "CLM_ID" and "DESYNPUF_ID" in train_df.columns and "DESYNPUF_ID" in test_df.columns:
        id_col = "DESYNPUF_ID"
    tr = set(train_df[id_col].dropna().astype(str).unique())
    te = set(test_df[id_col].dropna().astype(str).unique())
    overlap = len(tr.intersection(te))
    return {
        "checked": True,
        "identifier": id_col,
        "unique_train_identifiers": len(tr),
        "unique_test_identifiers": len(te),
        "overlapping_identifiers": int(overlap),
        "overlap_percent_of_test_unique_ids": round(100 * overlap / max(len(te), 1), 4),
        "warning": "Beneficiary overlap exists; consider a group split by beneficiary for a stricter generalisation estimate." if overlap else "No beneficiary overlap detected.",
    }


def write_final_report(
    train_df: pd.DataFrame, test_df: pd.DataFrame, target: str, feature_cols: list[str],
    exclusion_table: pd.DataFrame, eda_summary: dict[str, Any], model_summary: dict[str, Any],
    split_method: str, overlap: dict[str, Any]
) -> None:
    train_target = normalise_target(train_df[target])
    test_target = normalise_target(test_df[target])
    total_rows = len(train_df) + len(test_df)
    excluded_features = exclusion_table[exclusion_table["decision"] == "EXCLUDED"]
    included_features = exclusion_table[exclusion_table["decision"] == "INCLUDED"]
    best = model_summary["best_model_by_f1_then_pr_auc"]
    metrics = model_summary["best_model_metrics"]
    balance = eda_summary["target_balance"]
    top_models = sorted(
        [r for r in model_summary["models"] if r.get("status") == "OK"],
        key=lambda r: (r.get("f1") if r.get("f1") is not None else -1),
        reverse=True
    )[:5]

    report = f"""# {PROJECT_NAME} — Final Project Report

## 1. Executive summary
This project builds and evaluates a binary classification workflow for triaging insurance claims into the target classes in **`{target}`**. The workflow includes data quality assessment, exploratory data analysis (EDA), skewness and kurtosis statistics, feature inclusion/exclusion decisions, class-imbalance diagnostics, ten classifier attempts, model comparison, and a business-use proposal.

**Best completed model by F1, with PR-AUC as tie-breaker:** `{best}`.

| Metric | Result |
|---|---:|
| Accuracy | {metrics.get('accuracy', float('nan')):.4f} |
| Balanced accuracy | {metrics.get('balanced_accuracy', float('nan')):.4f} |
| Precision | {metrics.get('precision', float('nan')):.4f} |
| Recall | {metrics.get('recall', float('nan')):.4f} |
| F1 score | {metrics.get('f1', float('nan')):.4f} |
| ROC-AUC | {metrics.get('roc_auc', float('nan')):.4f} |
| PR-AUC / Average precision | {metrics.get('average_precision_pr_auc', float('nan')):.4f} |
| Matthews correlation coefficient | {metrics.get('matthews_corrcoef', float('nan')):.4f} |

These are test-set metrics from the files and split described below. They are not evidence of real-world financial savings, fraud detection, or deployment readiness.

## 2. Dataset and split
- Dataset rows across train/test files: **{total_rows:,}**
- Training rows available: **{len(train_df):,}**
- Test rows evaluated: **{len(test_df):,}**
- Columns in training file: **{train_df.shape[1]}**
- Target: **`{target}`**
- Predictors retained for modeling: **{len(feature_cols)}**
- Features/columns excluded from predictors: **{len(excluded_features)}** (including target, identifiers, payment fields, post-outcome or unusable columns)
- Feature arithmetic check: **{train_df.shape[1]} total columns = {len(feature_cols)} retained predictors + {len(excluded_features)} excluded columns**
- Split method: **{split_method}**

### Retained predictor names ({len(feature_cols)} total)
{', '.join(f'`{c}`' for c in feature_cols)}

### Excluded columns and reasons
{chr(10).join(f"- `{r['feature']}` — {r['reason']}" for _, r in excluded_features.iterrows())}

### Target distribution
- Counts: `{json.dumps(balance['counts'])}`
- Percentages: `{json.dumps(balance['percentages'])}`
- Majority/minority ratio: **{balance['majority_to_minority_ratio']:.2f}:1**
- Imbalance handling recommended by threshold ({MIN_IMBALANCE_RATIO}:1): **{balance['imbalance_handling_recommended']}**
- Handling actually used in model training: **{model_summary['imbalance_handling_used']}** — {model_summary['imbalance_method']}

## 3. Data quality and EDA outputs
The pipeline writes the following tables under `reports/final_tables/`:
- `data_inventory.csv`: type, non-null count, missingness, and unique-value count for every column.
- `missingness_report.csv`: missing counts and percentages.
- `numeric_descriptive_statistics_skew_kurtosis.csv`: count, mean, standard deviation, quantiles, skewness, kurtosis, zeros, negatives and infinities.
- `skewness_kurtosis_all_numeric_features.csv`: full numeric-feature skewness/kurtosis ranking.
- `iqr_outlier_audit.csv`: IQR outlier counts. Outliers are flagged, not blindly deleted.
- `numeric_correlation_matrix.csv` and `high_correlation_pairs_abs_0_85.csv`.
- `feature_inclusion_exclusion_audit.csv`: each column's inclusion/exclusion decision and reason.
- `preprocessing_plan.csv`: imputation, scaling and encoding plan.

The EDA plots under `reports/final_plots/` include target balance, missingness, numeric distributions, skewness, correlation heatmap, target-rate plots, metric comparisons, and the best-model confusion matrix. The script does not automatically remove outliers or transform every skewed variable: those actions require domain justification and should be tested against held-out performance.

## 4. Feature selection and leakage prevention
The target is excluded from predictors. Claim/beneficiary identifiers are excluded to reduce memorisation. Payment/reimbursement columns are excluded because they may encode downstream decisions or leak information unavailable at triage time. `claim_duration_days` is excluded by default because it can be unavailable at initial triage and may act as a proxy for inpatient status. Constant and all-missing training columns are also excluded.

**Important:** the inclusion/exclusion list is a conservative starting point, not a substitute for a data dictionary and timestamp review. Before deployment, the insurer must confirm that every retained predictor is genuinely available at the moment a triage decision is made.

## 5. Model comparison
The complete model comparison is in `reports/final_model_comparison.csv`. The pipeline attempts ten models: majority-class dummy baseline, logistic regression, SGD classifier, decision tree, random forest, extra trees, Ridge classifier, AdaBoost, Gaussian Naive Bayes, and linear SVM. Individual model reports and confusion matrices are saved under `reports/final_models/`.

Top completed models by F1:
{chr(10).join(f"- **{r['model']}** — F1 {float(r['f1']):.4f}; recall {float(r['recall']):.4f}; precision {float(r['precision']):.4f}; PR-AUC {float(r['average_precision_pr_auc']):.4f}" for r in top_models)}

F1 is used as the primary ranking metric to balance precision and recall; PR-AUC is a useful complementary metric for imbalanced classification. The best model should not be chosen using accuracy alone.

## 6. Train/test validation and limitations
- The model is evaluated on a held-out test set, which is not used for fitting preprocessing or model parameters.
- Preprocessing is fitted inside each model pipeline on training data only.
- Beneficiary overlap audit: `{json.dumps(overlap)}`.
- The script caps model-fitting rows at {MAX_TRAIN_ROWS:,} for practical memory use when the training file is larger; the full test set is evaluated.
- High scores can still reflect dataset-specific patterns, temporal leakage, target construction, or repeated beneficiaries. Verify the data-generation process and run a beneficiary-grouped and/or time-based holdout before claiming generalisation.
- This is classification/triage support, not fraud detection and not an automatic claim-denial system.

## 7. Practical value to an insurance company
If validated in a controlled pilot, the model could:
1. **Route claims into appropriate queues** (for example, likely inpatient vs other) so staff can prioritise and assign workloads.
2. **Reduce manual sorting** for routine claims and make the process more consistent.
3. **Support staffing and capacity planning** using claim-volume patterns and predicted queue composition.
4. **Help monitor service levels** by comparing predicted and actual categories, error rates, and processing times.
5. **Identify cases for human review** when confidence is low or model outputs conflict with existing rules.

The project does **not** prove that an insurer has already saved money, reduced processing time, or detected fraud. Those benefits require a prospective pilot with measured baselines and outcomes.

## 8. Recommended operating strategy
1. Use the model as a **decision-support and routing aid**, never as the sole basis for denial or reduction of benefits.
2. Keep a human reviewer for uncertain, high-value, unusual, or disputed claims.
3. Log model version, prediction, confidence/score, final human decision, and reason for override.
4. Monitor precision, recall, PR-AUC, false positives/negatives, latency, and drift monthly.
5. Check performance by relevant demographic and geographic groups, and investigate unequal error rates.
6. Protect sensitive beneficiary data with access controls, encryption, retention limits, and approved handling.
7. Compare the model with the current workflow in a pilot; only scale if measurable improvements occur without unacceptable harm.

## 9. Key fundamentals
- **EDA:** understand structure, missing data, distributions, skewness, kurtosis, outliers, correlations and class balance.
- **Cleaning:** standardise target labels and inspect missingness; imputation is learned from training data to avoid leakage.
- **Features:** predictors are columns available at prediction time; identifiers and downstream outcomes are excluded.
- **Train/test:** training fits the model; test estimates performance on unseen rows.
- **Imbalance:** if the majority/minority ratio crosses the configured threshold, class weights are enabled for models that support them. No resampling is applied blindly.
- **Precision:** among predicted positive cases, the fraction that is truly positive.
- **Recall:** among actual positive cases, the fraction identified by the model.
- **F1:** harmonic mean of precision and recall.
- **ROC-AUC / PR-AUC:** ranking metrics; PR-AUC is particularly informative when the positive class is rare.
- **Confusion matrix:** shows correct and incorrect predictions by class.

## 10. Final status
This script creates reproducible analysis outputs and a report. Before operational use, confirm target meaning, predictor availability at decision time, beneficiary/time split validity, privacy approval, subgroup performance, and pilot results.
"""
    (REPORTS_DIR / "FINAL_PROJECT_REPORT.md").write_text(report, encoding="utf-8")

    # Machine-readable final summary.
    summary = {
        "project_name": PROJECT_NAME,
        "created_at_local": time.strftime("%Y-%m-%d %H:%M:%S"),
        "root": str(ROOT),
        "target": target,
        "split_method": split_method,
        "train_rows": len(train_df),
        "test_rows": len(test_df),
        "total_rows": total_rows,
        "total_input_columns": int(train_df.shape[1]),
        "retained_predictors": len(feature_cols),
        "excluded_columns_count": int(len(excluded_features)),
        "excluded_columns": excluded_features.to_dict(orient="records"),
        "included_predictors": feature_cols,
        "target_balance": balance,
        "best_model": best,
        "best_metrics": metrics,
        "model_summary": model_summary,
        "beneficiary_overlap": overlap,
        "output_folders": {
            "tables": str(TABLES_DIR),
            "plots": str(PLOTS_DIR),
            "models": str(MODELS_DIR),
            "report": str(REPORTS_DIR / "FINAL_PROJECT_REPORT.md"),
        },
    }
    with open(REPORTS_DIR / "FINAL_PROJECT_SUMMARY.json", "w", encoding="utf-8") as f:
        json.dump(safe_json(summary), f, indent=2)


def main() -> int:
    log(f"Project root: {ROOT}")
    import sklearn
    log(f"Python: {sys.version.split()[0]} | pandas: {pd.__version__} | scikit-learn: {sklearn.__version__}")
    start_time = time.time()

    train_df, test_df, target, split_method = load_train_test()
    train_df[target] = normalise_target(train_df[target])
    test_df[target] = normalise_target(test_df[target])

    # EDA tables/statistics use the complete union of train and test records.
    # Model fitting remains train-only; plots internally sample only for legibility when needed.
    combined = pd.concat([train_df, test_df], ignore_index=True)
    eda_df = combined
    eda_summary = create_eda(eda_df, target)
    eda_summary["eda_sample_rows"] = int(len(eda_df))
    eda_summary["full_combined_rows"] = int(len(combined))
    eda_summary["eda_is_sampled"] = False

    # Record raw full split inventory separately, without relying on the EDA sample.
    save_csv(pd.DataFrame([describe_dataset(train_df, "train"), describe_dataset(test_df, "test"), describe_dataset(combined, "combined")]),
             TABLES_DIR / "dataset_overview.csv")

    Xtr_with_y, Xte_with_y, feature_cols, exclusion_table = build_feature_plan(train_df, test_df, target)
    model_summary = run_models(train_df, test_df, target, feature_cols, eda_summary["target_balance"])
    overlap = check_beneficiary_overlap(train_df, test_df)
    write_final_report(train_df, test_df, target, feature_cols, exclusion_table, eda_summary,
                       model_summary, split_method, overlap)

    elapsed = round(time.time() - start_time, 2)
    log("=" * 70)
    log("PIPELINE FINISHED")
    log(f"Best model: {model_summary['best_model_by_f1_then_pr_auc']}")
    log(f"Predictors retained: {len(feature_cols)}")
    log(f"Excluded columns: {int((exclusion_table['decision'] == 'EXCLUDED').sum())}")
    log(f"Input columns: {train_df.shape[1]} | Feature arithmetic: {len(feature_cols)} retained + {int((exclusion_table['decision'] == 'EXCLUDED').sum())} excluded")
    log("Retained predictors: " + ", ".join(feature_cols))
    log("Excluded columns: " + ", ".join(exclusion_table.loc[exclusion_table['decision'] == 'EXCLUDED', 'feature'].astype(str).tolist()))
    log(f"Models successful: {model_summary['models_successful']}/{model_summary['models_attempted']}")
    log(f"Final report: {REPORTS_DIR / 'FINAL_PROJECT_REPORT.md'}")
    log(f"Summary JSON: {REPORTS_DIR / 'FINAL_PROJECT_SUMMARY.json'}")
    log(f"Plots: {PLOTS_DIR}")
    log(f"Tables: {TABLES_DIR}")
    log(f"Elapsed seconds: {elapsed}")
    log("=" * 70)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\nStopped by user. Any completed output files remain in reports/.")
        raise SystemExit(130)
    except Exception as exc:
        print("\nFINAL PIPELINE STOPPED WITH AN ERROR")
        print(f"{type(exc).__name__}: {exc}")
        print("\nPlease copy the complete error output before changing data or deleting files.")
        raise
