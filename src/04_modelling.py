import os, json, warnings
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier

warnings.filterwarnings("ignore")
os.makedirs("reports/models", exist_ok=True)
os.makedirs("reports/plots", exist_ok=True)

train = pd.read_csv("data/train.csv")
test = pd.read_csv("data/test.csv")
target = "is_inpatient"

# Exclude identifiers and claim-payment fields that may encode the label.
drop_cols = [
    target, "DESYNPUF_ID", "CLM_ID", "SP_STATE_CODE", "dx_chapter",
    "MEDREIMB_IP", "BENRES_IP", "PPPYMT_IP",
    "MEDREIMB_OP", "BENRES_OP", "PPPYMT_OP",
    "MEDREIMB_CAR", "BENRES_CAR", "PPPYMT_CAR"
]
features = [c for c in train.columns if c not in drop_cols]
X = train[features].copy()
y = train[target].astype(int)
X_test = test[features].copy()
y_test = test[target].astype(int)

cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
num_cols = [c for c in features if c not in cat_cols]
prep = ColumnTransformer([
    ("num", Pipeline([("imputer", SimpleImputer(strategy="median")),
                      ("scale", StandardScaler())]), num_cols),
    ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                      ("onehot", OneHotEncoder(handle_unknown="ignore"))]), cat_cols)
])

models = {
    "LogisticRegression": LogisticRegression(max_iter=1000, class_weight="balanced"),
    "RandomForest": RandomForestClassifier(n_estimators=150, min_samples_leaf=2, class_weight="balanced", n_jobs=-1, random_state=42),
    "ExtraTrees": ExtraTreesClassifier(n_estimators=150, min_samples_leaf=2, class_weight="balanced", n_jobs=-1, random_state=42)
}
results, fitted = [], {}
for name, model in models.items():
    print(f"\nTraining {name}...", flush=True)
    pipe = Pipeline([("prep", prep), ("model", model)])
    pipe.fit(X, y)
    pred = pipe.predict(X_test)
    prob = pipe.predict_proba(X_test)[:, 1]
    row = {
        "model": name,
        "accuracy": accuracy_score(y_test, pred),
        "precision": precision_score(y_test, pred, zero_division=0),
        "recall": recall_score(y_test, pred, zero_division=0),
        "f1": f1_score(y_test, pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, prob)
    }
    results.append(row)
    fitted[name] = pipe
    pd.DataFrame(classification_report(y_test, pred, output_dict=True, zero_division=0)).T.to_csv(f"reports/models/{name}_classification_report.csv")
    pd.DataFrame(confusion_matrix(y_test, pred)).to_csv(f"reports/models/{name}_confusion_matrix.csv", index=False)
    print(row, flush=True)

res = pd.DataFrame(results).sort_values("f1", ascending=False)
res.to_csv("reports/model_comparison.csv", index=False)
best_name = res.iloc[0]["model"]
with open("reports/models/run_summary.json", "w") as f:
    json.dump({"target": target, "excluded_columns": drop_cols, "features": features,
               "best_model_by_f1": best_name, "comparison": results}, f, indent=2)
print("\n===== MODEL COMPARISON =====")
print(res.to_string(index=False))
print("\nBest by F1:", best_name)
print("Saved reports/model_comparison.csv and reports/models/")
