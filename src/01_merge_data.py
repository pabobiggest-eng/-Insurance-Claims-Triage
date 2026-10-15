"""
Step 1: Merge CMS DE-SynPUF Sample 1 files (Inpatient + Outpatient + Beneficiary 2008)
Output: data/merged_claims_sample1.csv

Rules:
- Inpatient and Outpatient claims are stacked (union of columns). Columns that exist
  in only one file are kept and are NaN for the other file.
- A CLAIM_TYPE column is added (Inpatient / Outpatient) -> classification target.
- Beneficiary 2008 summary is LEFT-joined on DESYNPUF_ID (many claims : 1 beneficiary).
- Nothing is dropped or cleaned here. Cleaning happens in step 2.
"""
import sys
import pandas as pd

RAW = sys.argv[1] if len(sys.argv) > 1 else "data"
OUT = f"{RAW}/merged_claims_sample1.csv"

inp = pd.read_csv(f"{RAW}/DE1_0_2008_to_2010_Inpatient_Claims_Sample_1.csv", low_memory=False)
out = pd.read_csv(f"{RAW}/DE1_0_2008_to_2010_Outpatient_Claims_Sample_1.csv", low_memory=False)
ben = pd.read_csv(f"{RAW}/DE1_0_2008_Beneficiary_Summary_File_Sample_1.csv")

inp["CLAIM_TYPE"] = "Inpatient"
out["CLAIM_TYPE"] = "Outpatient"

assert not set(inp.CLM_ID) & set(out.CLM_ID), "CLM_ID overlaps between files"
assert ben.DESYNPUF_ID.is_unique, "Beneficiary key not unique"

claims = pd.concat([inp, out], ignore_index=True, sort=False)
merged = claims.merge(ben, on="DESYNPUF_ID", how="left", validate="m:1", indicator=True)

assert len(merged) == len(inp) + len(out), "Row count changed during merge"
assert (merged["_merge"] == "both").all(), "Some claims have no beneficiary match"
merged = merged.drop(columns="_merge")

print("Inpatient rows :", len(inp))
print("Outpatient rows:", len(out))
print("Merged shape   :", merged.shape)
merged.to_csv(OUT, index=False)
print("Saved ->", OUT)
