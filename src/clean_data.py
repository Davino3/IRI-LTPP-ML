"""
Step 4: clean pavement_merged.csv and add section attributes (age, pavement family,
functional class) from Bucket_146230.xlsx.

Run from the repository root:  python src/clean_data.py
Output: pavement_clean.csv
"""
import numpy as np
import pandas as pd

MERGED = "data/pavement_merged.csv"   # not in the repo: rebuild from LTPP InfoPave exports
IMS = "data/Bucket_146230.xlsx"          # raw LTPP export, not in the repo
OUT = "data/pavement_clean.csv"
KEY = ["STATE_CODE", "SHRP_ID", "CONSTRUCTION_NO"]

d = pd.read_csv(MERGED, dtype={"SHRP_ID": str})
log = [f"Rows in merged file: {len(d)}"]

# 1. Exact duplicate rows
n = len(d); d = d.drop_duplicates()
log.append(f"Dropped exact duplicates: {n - len(d)}")

# 2. Section ID. SHRP_ID repeats across states, so a section is STATE_CODE + SHRP_ID.
d["SECTION_ID"] = d.STATE_CODE.astype(str).str.zfill(2) + "_" + d.SHRP_ID

# 3. Fill the few missing state names from other rows with the same STATE_CODE
names = d.dropna(subset=["STATE_CODE_EXP"]).groupby("STATE_CODE").STATE_CODE_EXP.first()
d["STATE_CODE_EXP"] = d.STATE_CODE_EXP.fillna(d.STATE_CODE.map(names))

# 4. Join construction event data (date each CONSTRUCTION_NO began, pavement family, experiment)
es = pd.read_excel(IMS, sheet_name="EXPERIMENT_SECTION", dtype={"SHRP_ID": str})
es = es[KEY + ["CN_ASSIGN_DATE", "CN_CHANGE_REASON_EXP", "GPS_SPS",
               "EXPERIMENT_NO_EXP", "PAVEMENT_FAMILY_EXP"]]
d = d.merge(es, on=KEY, how="left", validate="many_to_one")

# 5. Functional class (one per section; take the most recent SHRP_INFO record)
si = pd.read_excel(IMS, sheet_name="SHRP_INFO", dtype={"SHRP_ID": str})
si = (si.sort_values("START_DATE").groupby(["STATE_CODE", "SHRP_ID"]).last()
        [["FUNC_CLASS_EXP"]].reset_index())
d = d.merge(si, on=["STATE_CODE", "SHRP_ID"], how="left", validate="many_to_one")

# 6. Engineered features
d["AGE_SINCE_CN"] = d.YEAR - d.CN_ASSIGN_DATE.dt.year        # years since this construction event
d["IS_REHAB"] = (d.CONSTRUCTION_NO > 1).astype(int)            # 1 = section has been rehabilitated
d["PAVEMENT_TYPE"] = np.where(d.PAVEMENT_FAMILY_EXP.str.startswith("Asphalt"), "Flexible", "Rigid")
d["URBAN"] = d.FUNC_CLASS_EXP.str.startswith("Urban").astype("Int64")
d["SHARE_CLASS9"] = d.AADTT_VEH_CLASS_9_TREND / d.AADTT_ALL_TRUCKS_TREND.replace(0, np.nan)
d["LOG_ESAL"] = np.log1p(d.ANNUAL_ESAL_TREND)
d["LOG_AADTT"] = np.log1p(d.AADTT_ALL_TRUCKS_TREND)

# 7. Drop columns that are redundant, too sparse, or would leak the target
drop = {
    "ANNUAL_TRUCK_VOLUME_TREND": "= AADTT x days counted; redundant and 11% missing",
    "CMLTV_VOL_VEH_CLASS_9_TREND": "24% missing; cumulative, depends on start of count",
    "TEMP_MEAN_AVG": "r = 1.00 with TEMP_AVG",
    "ANNUAL_GESAL_TREND": "r = 0.95 with ANNUAL_ESAL_TREND",
    "IRI_LEFT_WHEEL_PATH": "target leakage (part of IRI_AVG)",
    "IRI_RIGHT_WHEEL_PATH": "target leakage (part of IRI_AVG)",
    "MRI": "target leakage (identical to IRI_AVG)",
}
d = d.drop(columns=list(drop))
log += [f"Dropped {c}: {why}" for c, why in drop.items()]

# 8. Rows with no traffic data
n = len(d); d = d.dropna(subset=["ANNUAL_ESAL_TREND", "AADTT_VEH_CLASS_9_TREND"])
log.append(f"Dropped rows missing traffic: {n - len(d)}")

# 9. Sanity checks
assert d.IRI_AVG.between(0.3, 6).all(), "IRI outside plausible range"
assert (d.AGE_SINCE_CN >= 0).all(), "Survey year before construction date"

d = d.sort_values(["SECTION_ID", "CONSTRUCTION_NO", "YEAR"])
d.to_csv(OUT, index=False)
log += [f"Rows out: {len(d)}", f"Sections: {d.SECTION_ID.nunique()}",
        f"Columns: {d.shape[1]}", f"Years: {d.YEAR.min()}-{d.YEAR.max()}"]
print("\n".join(log))
print("\nMissing values remaining:\n", d.isna().sum()[d.isna().sum() > 0].to_string())
