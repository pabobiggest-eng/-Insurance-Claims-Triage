"""
03_leakage_free_dataset.py
Insurance Claims Triage - Step 2: Leakage removal + patient-level train/test split

Kaise chalana hai (project root se):
    python src\\03_leakage_free_dataset.py

Input : data/claims_clean.csv          (02_cleaning_eda.py ka output)
        reports/leakage_columns_to_drop.txt
Output: data/train.csv, data/test.csv  (leakage-free, patient-level split)
        reports/feature_decisions.csv  (har column ka faisla aur wajah)
        reports/feature_sets.json      (standard / strict feature lists)
        reports/leakage_free_check.csv (sanity check ke numbers)

Leakage hatane ke 4 qadam:
  1. Structural leakage: jo columns sirf ek class mein bhare hote hain (Inpatient ya Outpatient).
  2. has_admit_dx: admitting diagnosis sirf Inpatient claims mein hota hai.
  3. Patient-level split: ek patient ke saare claims ya train mein ya test mein (kabhi dono mein nahi).
  4. Preprocessing (imputation/scaling) yahan nahi hoti, model pipeline mein train par fit hogi.
"""
import json
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import (average_precision_score, f1_score, precision_score, recall_score,
                             roc_auc_score)
from sklearn.model_selection import StratifiedGroupKFold

warnings.filterwarnings("ignore")
pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 50)

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
REPORTS = ROOT / "reports"
RANDOM_STATE = 42
T0 = time.time()

ID_COLS = ["DESYNPUF_ID", "CLM_ID"]
TARGET = "is_inpatient"
LABEL_TEXT = "CLAIM_TYPE"           # target ka text version, feature nahi banega
CAT_COLS = ["SP_STATE_CODE", "dx_chapter"]

# 02_cleaning_eda.py ke leakage audit mein STRUCTURAL aaye (AUC ~ 1.0 ya fill-gap > 0.9)
KNOWN_STRUCTURAL = [
    "NCH_BENE_PTA_COINSRNC_LBLTY_AM", "CLM_PASS_THRU_PER_DIEM_AMT", "NCH_BENE_PTB_COINSRNC_AMT",
    "CLM_UTLZTN_DAY_CNT", "NCH_BENE_PTB_DDCTBL_AMT", "has_discharge_dt", "has_admission_dt",
    "has_drg", "NCH_BENE_IP_DDCTBL_AMT", "n_hcpcs",
]
# Audit mein STRONG aaya, lekin logically bhi leakage hai (sirf Inpatient claims mein admitting dx hota hai)
MANUAL_DROP = {"has_admit_dx": "Admitting diagnosis sirf Inpatient claims mein hota hai (AUC 0.87)"}
# STRICT set se ye bhi nikalte hain: claim process hone ke baad / annual totals jin mein yeh claim shamil hota hai
STRICT_EXTRA_DROP = {
    "CLM_PMT_AMT": "Payment claim adjudicate hone ke baad pata chalti hai",
    "MEDREIMB_IP": "Annual inpatient total: is claim ka paisa isi mein shamil ho sakta hai",
    "BENRES_IP": "Annual inpatient total: is claim ka paisa isi mein shamil ho sakta hai",
    "PPPYMT_IP": "Annual inpatient total: is claim ka paisa isi mein shamil ho sakta hai",
}


def say(msg):
    print(f"[{time.time() - T0:6.0f}s] {msg}", flush=True)


# ----------------------------------------------------------------------------
# 1. LOAD
# ----------------------------------------------------------------------------
src = DATA / "claims_clean.csv"
if not src.exists():
    raise FileNotFoundError(f"{src} nahi mili. Pehle python src\\02_cleaning_eda.py chalayein.")
say("Step 1/5: claims_clean.csv load ho rahi hai")
df = pd.read_csv(src, dtype={"DESYNPUF_ID": str, "CLM_ID": str, "SP_STATE_CODE": str, "dx_chapter": str,
                             "CLAIM_TYPE": str}, low_memory=False)
say(f"Loaded: {len(df):,} rows x {df.shape[1]} cols")

# basic integrity checks
assert df[TARGET].isin([0, 1]).all(), "Target mein 0/1 ke ilawa values hain"
assert df["CLM_ID"].is_unique, "CLM_ID unique nahi hai (duplicates reh gaye)"
assert df[ID_COLS].notna().all().all(), "ID columns mein missing values hain"
say("Integrity checks pass: target 0/1, CLM_ID unique, IDs missing nahi")

# ----------------------------------------------------------------------------
# 2. LEAKAGE COLUMNS HATANA
# ----------------------------------------------------------------------------
say("Step 2/5: Leakage columns hata rahe hain")
drop_reason = {c: "STRUCTURAL: sirf ek class mein bhara hota hai (audit AUC ~1.0)" for c in KNOWN_STRUCTURAL}
leak_file = REPORTS / "leakage_columns_to_drop.txt"
if leak_file.exists():
    for c in leak_file.read_text(encoding="utf-8").split():
        drop_reason.setdefault(c, "STRUCTURAL: 02_cleaning_eda.py ke leakage audit se")
drop_reason.update(MANUAL_DROP)
to_drop = [c for c in drop_reason if c in df.columns]
missing_from_df = [c for c in drop_reason if c not in df.columns]
if missing_from_df:
    say(f"Note: ye drop-list columns data mein hain hi nahi (skip): {missing_from_df}")
df = df.drop(columns=to_drop + ([LABEL_TEXT] if LABEL_TEXT in df.columns else []))
say(f"Dropped {len(to_drop)} leakage columns: {to_drop}")

feature_cols_standard = [c for c in df.columns if c not in ID_COLS + [TARGET]]
feature_cols_strict = [c for c in feature_cols_standard if c not in STRICT_EXTRA_DROP]

# ----------------------------------------------------------------------------
# 3. PATIENT-LEVEL TRAIN / TEST SPLIT (80 / 20, class ratio barqarar)
# ----------------------------------------------------------------------------
say("Step 3/5: Patient-level split")
sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
train_idx, test_idx = next(sgkf.split(df, df[TARGET], groups=df["DESYNPUF_ID"]))
train = df.iloc[train_idx].reset_index(drop=True)
test = df.iloc[test_idx].reset_index(drop=True)

overlap = set(train["DESYNPUF_ID"]) & set(test["DESYNPUF_ID"])
clm_overlap = set(train["CLM_ID"]) & set(test["CLM_ID"])
assert len(overlap) == 0, f"{len(overlap)} patients train aur test dono mein hain!"
assert len(clm_overlap) == 0, "CLM_ID train aur test dono mein hai!"
say(f"Train {len(train):,} ({train[TARGET].mean() * 100:.2f}% Inpatient) | "
    f"Test {len(test):,} ({test[TARGET].mean() * 100:.2f}% Inpatient)")
say(f"Patient overlap = {len(overlap)} (0 hona chahiye)  |  Claim overlap = {len(clm_overlap)}")

# ----------------------------------------------------------------------------
# 4. RE-AUDIT: ab bhi koi feature bohot zyada predictive hai? (sirf TRAIN par)
# ----------------------------------------------------------------------------
say("Step 4/5: Re-audit (single-feature AUC, sirf train par)")
rng = np.random.default_rng(RANDOM_STATE)
idx = rng.choice(len(train), size=min(150_000, len(train)), replace=False)
ys = train[TARGET].to_numpy()[idx]
aud = []
for c in feature_cols_standard:
    if c in CAT_COLS:
        continue
    x = np.nan_to_num(train[c].to_numpy(dtype="float64")[idx], nan=-999999.0)
    try:
        a = roc_auc_score(ys, x)
    except ValueError:
        continue
    aud.append({"feature": c, "single_feature_auc": max(a, 1 - a)})
aud = pd.DataFrame(aud).sort_values("single_feature_auc", ascending=False)
print(aud.head(8).round(3).to_string(index=False))
worst = aud.iloc[0]
if worst["single_feature_auc"] >= 0.97:
    say(f"WARNING: {worst['feature']} ka AUC {worst['single_feature_auc']:.3f} abhi bhi bohot high hai, "
        "isay bhi check karein.")
else:
    say(f"Re-audit OK: sab se zyada single-feature AUC = {worst['single_feature_auc']:.3f} ({worst['feature']})")

# ----------------------------------------------------------------------------
# 5. SANITY CHECK MODEL (final model nahi; sirf yeh dekhne ke liye ke leakage khatam hui)
# ----------------------------------------------------------------------------
say("Step 5/5: Quick sanity model (HistGradientBoosting, class_weight=balanced)")


def prep(tr, te, cols):
    Xtr, Xte = tr[cols].copy(), te[cols].copy()
    for c in [c for c in CAT_COLS if c in cols]:
        Xtr[c] = Xtr[c].astype("category")
        Xte[c] = pd.Categorical(Xte[c], categories=Xtr[c].cat.categories)
    return Xtr, Xte


def run(name, cols):
    Xtr, Xte = prep(train, test, cols)
    model = HistGradientBoostingClassifier(max_iter=150, learning_rate=0.1, class_weight="balanced",
                                           categorical_features="from_dtype", random_state=RANDOM_STATE)
    model.fit(Xtr, train[TARGET])
    p = model.predict_proba(Xte)[:, 1]
    pred = (p >= 0.5).astype(int)
    yt = test[TARGET]
    res = {"feature_set": name, "n_features": len(cols),
           "roc_auc": roc_auc_score(yt, p), "pr_auc": average_precision_score(yt, p),
           "f1_inpatient": f1_score(yt, pred), "precision": precision_score(yt, pred),
           "recall": recall_score(yt, pred), "accuracy": float((pred == yt).mean())}
    say(f"{name:9s}: AUC {res['roc_auc']:.4f} | PR-AUC {res['pr_auc']:.4f} | F1 {res['f1_inpatient']:.4f} "
        f"| P {res['precision']:.3f} | R {res['recall']:.3f}")
    return res


results = [run("standard", feature_cols_standard), run("strict", feature_cols_strict)]
pd.DataFrame(results).to_csv(REPORTS / "leakage_free_check.csv", index=False)

# ----------------------------------------------------------------------------
# SAVE
# ----------------------------------------------------------------------------
decisions = []
for c in drop_reason:
    decisions.append({"column": c, "decision": "DROP (leakage)", "reason": drop_reason[c]})
for c in STRICT_EXTRA_DROP:
    if c in df.columns:
        decisions.append({"column": c, "decision": "KEEP in standard / DROP in strict", "reason": STRICT_EXTRA_DROP[c]})
for c in feature_cols_strict:
    decisions.append({"column": c, "decision": "KEEP", "reason": "leakage audit mein theek"})
for c in ID_COLS:
    decisions.append({"column": c, "decision": "ID (feature nahi)", "reason": "patient-level split / traceability"})
decisions.append({"column": TARGET, "decision": "TARGET", "reason": "1 = Inpatient, 0 = Outpatient"})
pd.DataFrame(decisions).to_csv(REPORTS / "feature_decisions.csv", index=False)

(REPORTS / "feature_sets.json").write_text(json.dumps({
    "target": TARGET, "ids": ID_COLS, "categorical": [c for c in CAT_COLS if c in df.columns],
    "dropped_leakage": to_drop, "standard": feature_cols_standard, "strict": feature_cols_strict,
}, indent=2), encoding="utf-8")

train.to_csv(DATA / "train.csv", index=False)
test.to_csv(DATA / "test.csv", index=False)
say("Saved: data/train.csv, data/test.csv, reports/feature_decisions.csv, feature_sets.json, leakage_free_check.csv")

print("\n" + "=" * 78)
print("LEAKAGE-FREE SUMMARY")
print("=" * 78)
print(f"Dropped leakage columns : {len(to_drop)} -> {to_drop}")
print(f"Train / Test            : {len(train):,} / {len(test):,}  (patient overlap = {len(overlap)})")
print(f"Features                : standard = {len(feature_cols_standard)}, strict = {len(feature_cols_strict)}")
print(f"Max single-feature AUC  : {worst['single_feature_auc']:.3f} ({worst['feature']})")
for r in results:
    print(f"  {r['feature_set']:9s} -> AUC {r['roc_auc']:.4f}, F1 {r['f1_inpatient']:.4f}, "
          f"precision {r['precision']:.3f}, recall {r['recall']:.3f}")
print("=" * 78)
say("DONE - koi error nahi.")
