"""
02_cleaning_eda.py
Insurance Claims Triage - Step 1: Data Cleaning + EDA

Kaise chalana hai (project root se, jahan data/ src/ reports/ folders hain):
    python src\\02_cleaning_eda.py

Input : data/merged_claims_sample1.csv
Output: data/claims_clean.csv
        reports/*.csv   (cleaning log, data dictionary, skew/kurtosis, tests, leakage audit ...)
        reports/plots/*.png

Design rules (leakage se bachne ke liye):
  * Is script mein koi bhi scaling / imputation / encoding FIT nahi hoti.
    Woh modelling pipeline mein train split ke BAAD hogi.
  * Cleaning sirf deterministic rules hain (duplicates, merge, invalid values).
  * Train/test split baad mein DESYNPUF_ID (patient) par hoga, taake ek patient
    ke claims train aur test dono mein na aayein. Isi liye ID columns clean file mein rakhe hain.
"""
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")  # plots window mein nahi khulte, sirf file mein save hote hain
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")
pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 50)

# ----------------------------------------------------------------------------
# 0. SETTINGS
# ----------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
DATA_IN = ROOT / "data" / "merged_claims_sample1.csv"
DATA_OUT = ROOT / "data" / "claims_clean.csv"
REPORTS = ROOT / "reports"
PLOTS = REPORTS / "plots"
REPORTS.mkdir(exist_ok=True)
PLOTS.mkdir(exist_ok=True)

RANDOM_STATE = 42
CHUNK = 200_000
rng = np.random.default_rng(RANDOM_STATE)
sns.set_theme(style="whitegrid", context="notebook")
T0 = time.time()
CLEAN_LOG = []  # (step, rows, cols, note)


def say(msg):
    print(f"[{time.time() - T0:6.0f}s] {msg}", flush=True)


def log_step(step, df, note=""):
    CLEAN_LOG.append({"step": step, "rows": len(df), "cols": df.shape[1], "note": note})
    say(f"{step}: {len(df):,} rows x {df.shape[1]} cols  {note}")


def save_fig(fig, name):
    fig.tight_layout()
    fig.savefig(PLOTS / name, dpi=130)
    plt.close(fig)


def parse_date(s):
    """CMS dates YYYYMMDD hoti hain. Agar format alag ho to generic parser use hota hai."""
    num = pd.to_numeric(s, errors="coerce")
    txt = num.round().astype("Int64").astype("string")
    out = pd.to_datetime(txt, format="%Y%m%d", errors="coerce")
    if out.notna().sum() < 0.5 * max(s.notna().sum(), 1):
        out = pd.to_datetime(s, errors="coerce")
    return out


def dx_chapter(first_dx):
    """ICD-9 diagnosis code ko 17 broad chapters mein badalta hai (high-cardinality se bachne ke liye)."""
    code = first_dx.astype("string").str.strip().str.upper()
    out = pd.Series("Missing", index=code.index, dtype="object")
    is_v = code.str.startswith("V", na=False)
    is_e = code.str.startswith("E", na=False)
    num = pd.to_numeric(code.str[:3], errors="coerce")
    bins = [0, 139, 239, 279, 289, 319, 389, 459, 519, 579, 629, 679, 709, 739, 759, 779, 799, 999]
    labels = ["Infectious", "Neoplasms", "Endocrine", "Blood", "Mental", "Nervous", "Circulatory",
              "Respiratory", "Digestive", "Genitourinary", "Pregnancy", "Skin", "Musculoskeletal",
              "Congenital", "Perinatal", "Symptoms", "Injury"]
    cut = pd.cut(num, bins=bins, labels=labels, include_lowest=True)
    out = out.mask(cut.notna(), cut.astype("object"))
    out = out.mask(is_v, "V_codes").mask(is_e, "E_codes")
    return out


# ----------------------------------------------------------------------------
# 1. LOAD (chunks mein, taake RAM kam lage)
# ----------------------------------------------------------------------------
say(f"Project root: {ROOT}")
if not DATA_IN.exists():
    raise FileNotFoundError(f"Data file nahi mili: {DATA_IN}\n"
                            "Script project root se chalayein (jahan data/ aur src/ folders hain).")

header = pd.read_csv(DATA_IN, nrows=0).columns.tolist()
RAW_COLS = len(header)
DX = [c for c in header if c.startswith("ICD9_DGNS_CD_")]
PRC = [c for c in header if c.startswith("ICD9_PRCDR_CD_")]
HCP = [c for c in header if c.startswith("HCPCS_CD_")]
NPI = [c for c in ["AT_PHYSN_NPI", "OP_PHYSN_NPI", "OT_PHYSN_NPI"] if c in header]
DATE_COLS = [c for c in ["CLM_FROM_DT", "CLM_THRU_DT", "CLM_ADMSN_DT", "NCH_BENE_DSCHRG_DT",
                         "BENE_BIRTH_DT", "BENE_DEATH_DT"] if c in header]
STR_COLS = [c for c in ["DESYNPUF_ID", "CLM_ID", "PRVDR_NUM", "ADMTNG_ICD9_DGNS_CD", "CLM_DRG_CD",
                        "CLAIM_TYPE", "BENE_ESRD_IND", "SP_STATE_CODE", "BENE_COUNTY_CD"]
            if c in header] + NPI + DATE_COLS + DX + PRC + HCP
SP_COLS = [c for c in header if c.startswith("SP_") and c != "SP_STATE_CODE"]
dtype_map = {c: "str" for c in STR_COLS}
NUM_RAW = [c for c in header if c not in STR_COLS]

say("Step 1/7: Data chunks mein load ho raha hai (5-10 minute lag sakte hain)...")
parts, hashes = [], []
fill_by_class = None      # har original column kitni baar bhara hai (Inpatient vs Outpatient)
class_counts = pd.Series(dtype="float64")
raw_notna_total = pd.Series(0, index=header, dtype="int64")
raw_rows = 0
type_values = {}

for ch in pd.read_csv(DATA_IN, chunksize=CHUNK, dtype=dtype_map, low_memory=False):
    raw_rows += len(ch)
    ctype_raw = ch["CLAIM_TYPE"].astype("string").str.strip()
    for k, v in ctype_raw.value_counts(dropna=False).items():
        type_values[k] = type_values.get(k, 0) + int(v)
    low = ctype_raw.str.lower()
    is_ip = pd.Series(np.nan, index=ch.index)
    is_ip = is_ip.mask(low.str.contains("inpat", na=False) | (low == "ip"), 1.0)
    is_ip = is_ip.mask(low.str.contains("outpat", na=False) | (low == "op"), 0.0)

    # exact-duplicate detection (raw row ka hash)
    h = pd.util.hash_pandas_object(ch, index=False).to_numpy()
    hashes.append(h)

    # raw missing + class-wise fill rate (leakage audit ke liye)
    nn = ch.notna()
    raw_notna_total += nn.sum()
    g = nn.groupby(is_ip.to_numpy()).sum()
    fill_by_class = g if fill_by_class is None else fill_by_class.add(g, fill_value=0)
    class_counts = class_counts.add(is_ip.value_counts(), fill_value=0)

    o = pd.DataFrame(index=ch.index)
    o["_h"] = h
    o["DESYNPUF_ID"] = ch["DESYNPUF_ID"]
    o["CLM_ID"] = ch["CLM_ID"]
    o["SEGMENT"] = pd.to_numeric(ch.get("SEGMENT"), errors="coerce").astype("float32")
    o["is_inpatient"] = is_ip.astype("float32")
    for c in DATE_COLS:
        o[c] = parse_date(ch[c])
    for c in NUM_RAW:
        if c in ("SEGMENT",) or c in o.columns:
            continue
        o[c] = pd.to_numeric(ch[c], errors="coerce").astype("float32")
    # code columns ko counts mein badalna (raw codes ki zaroorat nahi, memory bachti hai)
    o["n_dx"] = ch[DX].notna().sum(axis=1).astype("float32")
    o["n_prcdr"] = ch[PRC].notna().sum(axis=1).astype("float32")
    o["n_hcpcs"] = ch[HCP].notna().sum(axis=1).astype("float32")
    o["n_physicians"] = ch[NPI].notna().sum(axis=1).astype("float32") if NPI else 0.0
    o["dx_chapter"] = dx_chapter(ch[DX[0]]) if DX else "Missing"
    o["has_drg"] = ch["CLM_DRG_CD"].notna().astype("int8") if "CLM_DRG_CD" in ch else 0
    o["has_admit_dx"] = ch["ADMTNG_ICD9_DGNS_CD"].notna().astype("int8") if "ADMTNG_ICD9_DGNS_CD" in ch else 0
    for c in SP_COLS:                      # SynPUF coding: 1 = Yes, 2 = No
        o[c] = np.where(ch[c].isna(), np.nan, (pd.to_numeric(ch[c], errors="coerce") == 1).astype("float32"))
    if "BENE_ESRD_IND" in ch:
        o["BENE_ESRD_IND"] = (ch["BENE_ESRD_IND"].astype("string").str.strip().str.upper() == "Y").fillna(False).astype("int8")
    if "SP_STATE_CODE" in ch:
        o["SP_STATE_CODE"] = ch["SP_STATE_CODE"].astype("string").str.strip()
    parts.append(o)

df = pd.concat(parts, ignore_index=True)
del parts
say(f"Load complete: {raw_rows:,} rows x {RAW_COLS} cols")
CLEAN_LOG.append({"step": "0. Raw data", "rows": raw_rows, "cols": RAW_COLS, "note": "original file"})
say(f"0. Raw data: {raw_rows:,} rows x {RAW_COLS} cols")

# ----------------------------------------------------------------------------
# 2. QUALITY AUDIT (cleaning se pehle data ki halat)
# ----------------------------------------------------------------------------
say("Step 2/7: Quality audit")
print("\nCLAIM_TYPE values (raw):", type_values)
miss_raw = pd.DataFrame({
    "column": raw_notna_total.index,
    "missing_pct": (1 - raw_notna_total.values / raw_rows) * 100,
}).sort_values("missing_pct", ascending=False)
miss_raw.to_csv(REPORTS / "raw_missing_values.csv", index=False)

n_exact_dup = int(pd.Series(np.concatenate(hashes)).duplicated().sum())
del hashes
say(f"Exact duplicate rows: {n_exact_dup:,}")

unmapped = int(df["is_inpatient"].isna().sum())
if unmapped:
    say(f"WARNING: {unmapped:,} rows ka CLAIM_TYPE samajh nahi aaya (ye rows hata di jayengi). Values: {type_values}")
if unmapped == len(df):
    raise ValueError(f"CLAIM_TYPE map nahi hua. Mili hui values: {type_values}")

# ----------------------------------------------------------------------------
# 3. CLEANING
# ----------------------------------------------------------------------------
say("Step 3/7: Cleaning")
# 3a. Exact duplicates
before = len(df)
df = df.drop_duplicates(subset="_h").drop(columns="_h").reset_index(drop=True)
log_step("1. Exact duplicates removed", df, f"removed {before - len(df):,}")

# 3b. Unknown claim type + missing CLM_ID
before = len(df)
df = df[df["is_inpatient"].notna() & df["CLM_ID"].notna()].reset_index(drop=True)
log_step("2. Unknown CLAIM_TYPE / missing CLM_ID removed", df, f"removed {before - len(df):,}")

# 3c. Repeated CLM_ID = continuation lines -> ek claim mein merge
dup_mask = df.duplicated("CLM_ID", keep=False)
n_repeated_ids = int(df.loc[dup_mask, "CLM_ID"].nunique())
rows_in_repeats = int(dup_mask.sum())
single = df[~dup_mask]
multi = df[dup_mask].sort_values(["CLM_ID", "SEGMENT"], kind="stable")
agg = {c: "first" for c in df.columns if c != "CLM_ID"}       # pehli non-null value
for c in ["n_dx", "n_prcdr", "n_hcpcs", "n_physicians", "has_drg", "has_admit_dx"]:
    agg[c] = "max"                                            # continuation lines mein counts ka max
merged = multi.groupby("CLM_ID", sort=False, as_index=False).agg(agg)
df = pd.concat([single, merged[df.columns]], ignore_index=True)
del single, multi, merged
log_step("3. Continuation lines merged (same CLM_ID)", df,
         f"{n_repeated_ids:,} CLM_ID ({rows_in_repeats:,} rows) -> {n_repeated_ids:,} claims")

# 3d. Orphan rows: dono claim dates missing
before = len(df)
orphan = df["CLM_FROM_DT"].isna() & df["CLM_THRU_DT"].isna()
df = df[~orphan].reset_index(drop=True)
log_step("4. Orphan rows (no claim dates) removed", df, f"removed {before - len(df):,}")

# 3e. Invalid values
before = len(df)
bad_order = (df["CLM_THRU_DT"] < df["CLM_FROM_DT"])
df = df[~bad_order.fillna(False)].reset_index(drop=True)
log_step("5. Claims with THRU date before FROM date removed", df, f"removed {before - len(df):,}")

# agar sirf ek date missing ho to doosri se bhar do (claim ek din ka maan kar) - deterministic rule
df["CLM_FROM_DT"] = df["CLM_FROM_DT"].fillna(df["CLM_THRU_DT"])
df["CLM_THRU_DT"] = df["CLM_THRU_DT"].fillna(df["CLM_FROM_DT"])

amount_cols = [c for c in df.columns if any(k in c for k in ["AMT", "_AM", "MEDREIMB", "BENRES", "PPPYMT"])]
neg_counts = {c: int((df[c] < 0).sum()) for c in amount_cols}
neg_total = sum(neg_counts.values())
for c in amount_cols:
    df.loc[df[c] < 0, c] = np.nan
CLEAN_LOG.append({"step": "6. Negative amounts set to NaN", "rows": len(df), "cols": df.shape[1],
                  "note": f"{neg_total:,} values: { {k: v for k, v in neg_counts.items() if v} }"})
say(f"6. Negative amounts set to NaN: {neg_total:,} values")

# ----------------------------------------------------------------------------
# 4. FEATURE ENGINEERING (sirf row-level, leakage-free)
# ----------------------------------------------------------------------------
say("Step 4/7: Feature engineering")
df["claim_duration_days"] = (df["CLM_THRU_DT"] - df["CLM_FROM_DT"]).dt.days.astype("float32")
df["claim_year"] = df["CLM_FROM_DT"].dt.year.astype("float32")
df["claim_month"] = df["CLM_FROM_DT"].dt.month.astype("float32")
age = (df["CLM_FROM_DT"] - df["BENE_BIRTH_DT"]).dt.days / 365.25
bad_age = (age < 0) | (age > 110)
n_bad_age = int(bad_age.sum())
df["age_at_claim"] = age.mask(bad_age).astype("float32")
df["is_deceased"] = df["BENE_DEATH_DT"].notna().astype("int8")
df["has_admission_dt"] = df["CLM_ADMSN_DT"].notna().astype("int8")
df["has_discharge_dt"] = df["NCH_BENE_DSCHRG_DT"].notna().astype("int8")
df["chronic_count"] = df[SP_COLS].sum(axis=1, min_count=1).astype("float32") if SP_COLS else np.nan
CLEAN_LOG.append({"step": "7. Invalid age (<0 or >110) set to NaN", "rows": len(df), "cols": df.shape[1],
                  "note": f"{n_bad_age:,} values"})
claims_after_death = int(((df["BENE_DEATH_DT"].notna()) & (df["CLM_FROM_DT"] > df["BENE_DEATH_DT"])).sum())
say(f"Claims after death date (sirf report, hata nahi rahe): {claims_after_death:,}")

df["CLAIM_TYPE"] = np.where(df["is_inpatient"] == 1, "Inpatient", "Outpatient")
df["is_inpatient"] = df["is_inpatient"].astype("int8")

DROP_FINAL = DATE_COLS + ["SEGMENT"]
df = df.drop(columns=[c for c in DROP_FINAL if c in df.columns])
# column order: IDs, target, baqi
front = ["DESYNPUF_ID", "CLM_ID", "CLAIM_TYPE", "is_inpatient"]
df = df[front + [c for c in df.columns if c not in front]]
log_step("8. Final clean dataset", df,
         f"raw {raw_rows:,}x{RAW_COLS} -> {len(df):,}x{df.shape[1]} (raw code/date columns counts/flags mein badle)")

# ----------------------------------------------------------------------------
# 5. LEAKAGE AUDIT
# ----------------------------------------------------------------------------
say("Step 5/7: Leakage audit")
ID_TARGET = ["DESYNPUF_ID", "CLM_ID", "CLAIM_TYPE", "is_inpatient"]
feat_cols = [c for c in df.columns if c not in ID_TARGET and c != "SP_STATE_CODE" and c != "dx_chapter"]
y = df["is_inpatient"].to_numpy()
samp_idx = rng.choice(len(df), size=min(150_000, len(df)), replace=False)
ys = y[samp_idx]
rows = []
for c in feat_cols:
    x = df[c].to_numpy(dtype="float64")
    fill_ip = np.mean(~np.isnan(x[y == 1])) if (y == 1).any() else np.nan
    fill_op = np.mean(~np.isnan(x[y == 0])) if (y == 0).any() else np.nan
    xs = np.nan_to_num(x[samp_idx], nan=-999999.0)
    try:
        auc = roc_auc_score(ys, xs)
    except ValueError:
        auc = np.nan
    auc_strength = max(auc, 1 - auc) if not np.isnan(auc) else np.nan
    rows.append({"feature": c, "fill_rate_inpatient": fill_ip, "fill_rate_outpatient": fill_op,
                 "fill_gap": abs(fill_ip - fill_op), "single_feature_auc": auc_strength})
leak = pd.DataFrame(rows)


def leak_label(r):
    # Structural: feature ka bhara hona hi class bata deta hai (ya near-perfect AUC)
    if r["fill_gap"] > 0.90 or r["single_feature_auc"] >= 0.97:
        return "STRUCTURAL LEAKAGE - drop"
    if r["single_feature_auc"] >= 0.85:
        return "STRONG PREDICTOR - business decision"
    return "ok"


leak["verdict"] = leak.apply(leak_label, axis=1)
leak = leak.sort_values("single_feature_auc", ascending=False)
leak.to_csv(REPORTS / "leakage_audit.csv", index=False)
leak_drop = leak.loc[leak["verdict"].str.startswith("STRUCTURAL"), "feature"].tolist()
(REPORTS / "leakage_columns_to_drop.txt").write_text("\n".join(leak_drop), encoding="utf-8")
say(f"Structural leakage columns ({len(leak_drop)}): {leak_drop}")
say("Strong predictors: " + str(leak.loc[leak["verdict"].str.startswith("STRONG"), "feature"].tolist()))

# ----------------------------------------------------------------------------
# 6. EDA: DESCRIPTIVE + SKEWNESS / KURTOSIS + OUTLIERS
# ----------------------------------------------------------------------------
say("Step 6/7: EDA (statistics + plots)")
NUM_FEATURES = [c for c in [
    "CLM_PMT_AMT", "NCH_PRMRY_PYR_CLM_PD_AMT", "CLM_PASS_THRU_PER_DIEM_AMT", "NCH_BENE_IP_DDCTBL_AMT",
    "NCH_BENE_PTA_COINSRNC_LBLTY_AM", "NCH_BENE_BLOOD_DDCTBL_LBLTY_AM", "CLM_UTLZTN_DAY_CNT",
    "NCH_BENE_PTB_DDCTBL_AMT", "NCH_BENE_PTB_COINSRNC_AMT", "BENE_HI_CVRAGE_TOT_MONS",
    "BENE_SMI_CVRAGE_TOT_MONS", "BENE_HMO_CVRAGE_TOT_MONS", "PLAN_CVRG_MOS_NUM",
    "MEDREIMB_IP", "BENRES_IP", "PPPYMT_IP", "MEDREIMB_OP", "BENRES_OP", "PPPYMT_OP",
    "MEDREIMB_CAR", "BENRES_CAR", "PPPYMT_CAR", "claim_duration_days", "age_at_claim",
    "n_dx", "n_prcdr", "n_hcpcs", "n_physicians", "chronic_count"] if c in df.columns]

# Data dictionary
dd = pd.DataFrame({
    "column": df.columns,
    "dtype": [str(t) for t in df.dtypes],
    "non_null": df.notna().sum().values,
    "missing_pct": (df.isna().mean() * 100).round(2).values,
    "n_unique": [df[c].nunique() for c in df.columns],
    "example": [df[c].dropna().iloc[0] if df[c].notna().any() else None for c in df.columns],
})
dd.to_csv(REPORTS / "data_dictionary.csv", index=False)

# Skewness / kurtosis (excess/Fisher), log1p se pehle aur baad
sk_rows = []
for c in NUM_FEATURES:
    s = df[c].dropna().astype("float64")
    if len(s) < 10 or s.nunique() < 2:
        continue
    r = {"feature": c, "n": len(s), "mean": s.mean(), "median": s.median(), "std": s.std(),
         "min": s.min(), "max": s.max(), "skewness": s.skew(), "excess_kurtosis": s.kurt()}
    if s.min() >= 0:
        sl = np.log1p(s)
        r["skew_log1p"] = sl.skew()
        r["kurt_log1p"] = sl.kurt()
        r["log1p_improves"] = abs(r["skew_log1p"]) < abs(r["skewness"])
    sk_rows.append(r)
skew_df = pd.DataFrame(sk_rows).sort_values("skewness", key=lambda x: x.abs(), ascending=False)
skew_df.to_csv(REPORTS / "skewness_kurtosis.csv", index=False)
n_high_skew = int((skew_df["skewness"].abs() > 1).sum())
n_improved = int(skew_df.get("log1p_improves", pd.Series(dtype=bool)).fillna(False).sum())

# Outliers (IQR rule) - sirf report; outliers hata nahi rahe (claims mein bade amounts real hote hain)
out_rows = []
for c in NUM_FEATURES:
    s = df[c].dropna()
    if len(s) < 10:
        continue
    q1, q3 = s.quantile(0.25), s.quantile(0.75)
    iqr = q3 - q1
    lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    n_out = int(((s < lo) | (s > hi)).sum())
    out_rows.append({"feature": c, "Q1": q1, "Q3": q3, "IQR": iqr, "lower_fence": lo, "upper_fence": hi,
                     "n_outliers": n_out, "outlier_pct": n_out / len(s) * 100})
pd.DataFrame(out_rows).sort_values("outlier_pct", ascending=False).to_csv(REPORTS / "outliers_iqr.csv", index=False)

# ---------------- PLOTS ----------------
n_ip = int((df["is_inpatient"] == 1).sum())
n_op = int((df["is_inpatient"] == 0).sum())
imb_ratio = max(n_ip, n_op) / max(min(n_ip, n_op), 1)
palette = {"Inpatient": "#d1495b", "Outpatient": "#00798c"}

# 01 class balance
fig, ax = plt.subplots(figsize=(5.5, 4))
vc = df["CLAIM_TYPE"].value_counts()
ax.bar(vc.index, vc.values, color=[palette[i] for i in vc.index])
for i, v in enumerate(vc.values):
    ax.text(i, v, f"{v:,}\n({v / len(df) * 100:.1f}%)", ha="center", va="bottom")
ax.set_title(f"Claim type balance (ratio {imb_ratio:.1f} : 1)")
ax.set_ylabel("claims")
ax.set_ylim(0, vc.max() * 1.2)
save_fig(fig, "01_class_balance.png")

# 02 missing values (raw) top 25
fig, ax = plt.subplots(figsize=(8, 7))
m = miss_raw.head(25).iloc[::-1]
ax.barh(m["column"], m["missing_pct"], color="#edae49")
ax.set_xlabel("missing %  (raw data)")
ax.set_title("Top 25 columns by missing % (raw)")
save_fig(fig, "02_missing_values_raw.png")

# 03 fill-rate gap by class (leakage picture)
fig, ax = plt.subplots(figsize=(8, 7))
lk = leak.sort_values("fill_gap", ascending=False).head(20).iloc[::-1]
yy = np.arange(len(lk))
ax.barh(yy - 0.2, lk["fill_rate_inpatient"], 0.4, label="Inpatient", color=palette["Inpatient"])
ax.barh(yy + 0.2, lk["fill_rate_outpatient"], 0.4, label="Outpatient", color=palette["Outpatient"])
ax.set_yticks(yy)
ax.set_yticklabels(lk["feature"])
ax.set_xlabel("fill rate (non-missing share)")
ax.set_title("Structural leakage check: columns filled in only one class")
ax.legend()
save_fig(fig, "03_fill_rate_by_class.png")

# 04 claims per year by type
fig, ax = plt.subplots(figsize=(7, 4))
yr = df.groupby(["claim_year", "CLAIM_TYPE"]).size().unstack(fill_value=0)
yr.plot(kind="bar", ax=ax, color=[palette.get(c, "grey") for c in yr.columns])
ax.set_title("Claims per year")
ax.set_xlabel("year")
ax.set_ylabel("claims")
save_fig(fig, "04_claims_per_year.png")

# 05 age distribution
fig, ax = plt.subplots(figsize=(7, 4))
for t, g in df.groupby("CLAIM_TYPE"):
    ax.hist(g["age_at_claim"].dropna(), bins=50, alpha=0.55, density=True, label=t, color=palette[t])
ax.set_title("Age at claim (density)")
ax.set_xlabel("age")
ax.legend()
save_fig(fig, "05_age_distribution.png")

# 06 payment: raw vs log1p
if "CLM_PMT_AMT" in df:
    fig, axs = plt.subplots(1, 2, figsize=(11, 4))
    s = df["CLM_PMT_AMT"].dropna()
    axs[0].hist(s, bins=80, color="#30638e")
    axs[0].set_title(f"CLM_PMT_AMT raw (skew {s.skew():.2f})")
    axs[1].hist(np.log1p(s), bins=80, color="#3d9970")
    axs[1].set_title(f"log1p(CLM_PMT_AMT) (skew {np.log1p(s).skew():.2f})")
    save_fig(fig, "06_payment_raw_vs_log.png")

# 07 skewness before vs after log1p
if "skew_log1p" in skew_df:
    top = skew_df.dropna(subset=["skew_log1p"]).head(15).iloc[::-1]
    fig, ax = plt.subplots(figsize=(9, 6.5))
    yy = np.arange(len(top))
    ax.barh(yy - 0.2, top["skewness"], 0.4, label="raw", color="#d1495b")
    ax.barh(yy + 0.2, top["skew_log1p"], 0.4, label="log1p", color="#3d9970")
    ax.set_yticks(yy)
    ax.set_yticklabels(top["feature"])
    ax.axvline(0, color="k", lw=0.8)
    ax.set_title("Skewness: raw vs log1p (top 15)")
    ax.legend()
    save_fig(fig, "07_skewness_before_after.png")

# 08 boxplots (log1p) by class
box_feats = [c for c in ["CLM_PMT_AMT", "claim_duration_days", "n_dx", "n_hcpcs", "age_at_claim", "chronic_count"]
             if c in df.columns]
fig, axs = plt.subplots(2, 3, figsize=(13, 7))
for ax, c in zip(axs.ravel(), box_feats):
    d = df[[c, "CLAIM_TYPE"]].dropna()
    d[c] = np.log1p(d[c]) if d[c].min() >= 0 else d[c]
    sns.boxplot(data=d, x="CLAIM_TYPE", y=c, ax=ax, palette=palette, showfliers=False)
    ax.set_title(f"log1p({c})" if df[c].min() >= 0 else c)
    ax.set_xlabel("")
for ax in axs.ravel()[len(box_feats):]:
    ax.axis("off")
save_fig(fig, "08_boxplots_by_class.png")

# 09 correlation heatmap
corr_cols = [c for c in NUM_FEATURES if df[c].notna().sum() > 100 and df[c].nunique() > 1]
samp = df[corr_cols].sample(min(100_000, len(df)), random_state=RANDOM_STATE)
cm = samp.corr(method="spearman")
fig, ax = plt.subplots(figsize=(13, 11))
sns.heatmap(cm, cmap="vlag", center=0, vmin=-1, vmax=1, ax=ax, annot=len(corr_cols) <= 18, fmt=".2f",
            cbar_kws={"shrink": 0.7})
ax.set_title("Spearman correlation (100k sample)")
save_fig(fig, "09_correlation_heatmap.png")
high_corr = [(a, b, cm.loc[a, b]) for i, a in enumerate(cm.columns) for b in cm.columns[i + 1:]
             if abs(cm.loc[a, b]) >= 0.9]
pd.DataFrame(high_corr, columns=["feature_a", "feature_b", "spearman"]).to_csv(
    REPORTS / "high_correlation_pairs.csv", index=False)

# 10 chronic conditions prevalence
if SP_COLS:
    fig, ax = plt.subplots(figsize=(8, 5))
    prev = df.groupby("CLAIM_TYPE")[SP_COLS].mean().T * 100
    prev.plot(kind="bar", ax=ax, color=[palette.get(c, "grey") for c in prev.columns])
    ax.set_ylabel("% of claims with condition")
    ax.set_title("Chronic condition prevalence by claim type")
    save_fig(fig, "10_chronic_conditions.png")

# 11 diagnosis chapter by class
fig, ax = plt.subplots(figsize=(10, 5))
ch = pd.crosstab(df["dx_chapter"], df["CLAIM_TYPE"], normalize="columns") * 100
ch = ch.sort_values(ch.columns[0], ascending=False)
ch.plot(kind="bar", ax=ax, color=[palette.get(c, "grey") for c in ch.columns])
ax.set_ylabel("% within claim type")
ax.set_title("Primary diagnosis chapter by claim type")
save_fig(fig, "11_dx_chapter_by_class.png")

# 12 duration by class
if "claim_duration_days" in df:
    fig, ax = plt.subplots(figsize=(7, 4))
    for t, g in df.groupby("CLAIM_TYPE"):
        ax.hist(np.log1p(g["claim_duration_days"].dropna()), bins=40, alpha=0.55, density=True, label=t,
                color=palette[t])
    ax.set_title("log1p(claim duration days)")
    ax.legend()
    save_fig(fig, "12_duration_by_class.png")

# 13 QQ plot payment
if "CLM_PMT_AMT" in df:
    fig, axs = plt.subplots(1, 2, figsize=(10, 4))
    s = df["CLM_PMT_AMT"].dropna().sample(min(20_000, df["CLM_PMT_AMT"].notna().sum()), random_state=RANDOM_STATE)
    stats.probplot(s, dist="norm", plot=axs[0])
    axs[0].set_title("QQ plot: raw payment")
    stats.probplot(np.log1p(s), dist="norm", plot=axs[1])
    axs[1].set_title("QQ plot: log1p payment")
    save_fig(fig, "13_qq_payment.png")

# 14 top states
if "SP_STATE_CODE" in df:
    fig, ax = plt.subplots(figsize=(9, 4))
    df["SP_STATE_CODE"].value_counts().head(15).plot(kind="bar", ax=ax, color="#30638e")
    ax.set_title("Top 15 state codes")
    save_fig(fig, "14_top_states.png")

# ----------------------------------------------------------------------------
# 7. STATISTICAL TESTS
# ----------------------------------------------------------------------------
say("Step 7/7: Statistical tests")
tests = []
ip_df = df[df["is_inpatient"] == 1]
op_df = df[df["is_inpatient"] == 0].sample(min(100_000, n_op), random_state=RANDOM_STATE)
for c in NUM_FEATURES:
    a = ip_df[c].dropna().to_numpy(dtype="float64")
    b = op_df[c].dropna().to_numpy(dtype="float64")
    if len(a) < 20 or len(b) < 20 or (np.ptp(a) == 0 and np.ptp(b) == 0 and a[0] == b[0]):
        continue
    u, p = stats.mannwhitneyu(a, b, alternative="two-sided")
    rbc = 2 * u / (len(a) * len(b)) - 1      # rank-biserial effect size (-1..1)
    tests.append({"test": "Mann-Whitney U", "variable": c, "statistic": u, "p_value": p,
                  "effect_size": rbc, "effect_name": "rank-biserial",
                  "n": len(a) + len(b),
                  "comment": "Inpatient values higher" if rbc > 0 else "Inpatient values lower"})

cat_cols = [c for c in ["BENE_SEX_IDENT_CD", "BENE_RACE_CD", "BENE_ESRD_IND", "is_deceased", "SP_STATE_CODE",
                        "dx_chapter"] + SP_COLS if c in df.columns]
for c in cat_cols:
    d = df[[c, "is_inpatient"]].dropna()
    ct = pd.crosstab(d[c], d["is_inpatient"])
    ct = ct.loc[ct.sum(axis=1) > 0, ct.sum(axis=0) > 0]
    if ct.shape[0] < 2 or ct.shape[1] < 2:
        continue
    chi2, p, dof, _ = stats.chi2_contingency(ct)
    n = ct.to_numpy().sum()
    v = np.sqrt(chi2 / (n * min(ct.shape[0] - 1, ct.shape[1] - 1)))
    tests.append({"test": "Chi-square", "variable": c, "statistic": chi2, "p_value": p,
                  "effect_size": v, "effect_name": "Cramer's V", "n": int(n),
                  "comment": f"dof={dof}"})

for c in [c for c in ["CLM_PMT_AMT", "claim_duration_days", "age_at_claim", "n_dx"] if c in df.columns]:
    s = df[c].dropna()
    if len(s) < 50:
        continue
    s = s.sample(min(5000, len(s)), random_state=RANDOM_STATE).to_numpy(dtype="float64")
    for name, arr in [("raw", s), ("log1p", np.log1p(s) if s.min() >= 0 else None)]:
        if arr is None or np.ptp(arr) == 0:
            continue
        w, p = stats.shapiro(arr)
        tests.append({"test": "Shapiro-Wilk (5000 sample)", "variable": f"{c} [{name}]", "statistic": w,
                      "p_value": p, "effect_size": np.nan, "effect_name": "",
                      "n": len(arr), "comment": "normal" if p > 0.05 else "NOT normal"})
tests_df = pd.DataFrame(tests)
tests_df.to_csv(REPORTS / "statistical_tests.csv", index=False)

# ----------------------------------------------------------------------------
# 8. SAVE
# ----------------------------------------------------------------------------
pd.DataFrame(CLEAN_LOG).to_csv(REPORTS / "cleaning_log.csv", index=False)
df.to_csv(DATA_OUT, index=False)
say(f"Saved: {DATA_OUT}")

# ----------------------------------------------------------------------------
# KEY FINDINGS (terminal summary)
# ----------------------------------------------------------------------------
print("\n" + "=" * 78)
print("KEY FINDINGS")
print("=" * 78)
print(f"DATA        : raw {raw_rows:,} rows x {RAW_COLS} cols  ->  clean {len(df):,} claims x {df.shape[1]} cols")
print(f"DUPLICATES  : exact duplicates = {n_exact_dup:,};  repeated CLM_ID merged = {n_repeated_ids:,} "
      f"({rows_in_repeats:,} rows)")
print(f"CLASS       : Inpatient {n_ip:,} ({n_ip / len(df) * 100:.1f}%), Outpatient {n_op:,} "
      f"({n_op / len(df) * 100:.1f}%), imbalance {imb_ratio:.1f}:1  -> F1 / class_weight par focus")
print(f"SKEWNESS    : {n_high_skew} features with |skew| > 1; log1p improves {n_improved}")
print(f"LEAKAGE     : structural = {len(leak_drop)} columns (reports/leakage_columns_to_drop.txt)")
strong = leak.loc[leak['verdict'].str.startswith('STRONG'), 'feature'].tolist()
print(f"              strong predictors (business decision) = {strong}")
print(f"HIGH CORR   : {len(high_corr)} pairs with |spearman| >= 0.9 (reports/high_correlation_pairs.csv)")
print(f"OUTPUTS     : {DATA_OUT.name}, reports/*.csv, reports/plots/*.png ({len(list(PLOTS.glob('*.png')))} plots)")
print("=" * 78)
say("DONE - koi error nahi.")
