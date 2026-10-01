import numpy as np
import pandas as pd
import illustris_python as il

basePath = "/home/tnguser/sims.TNG/L35n2160TNG/output"
snapNum = 99

targets = [11.5, 12.0, 12.5, 13.0]
delta_logM = 0.25

header = il.groupcat.loadHeader(basePath, snapNum)
h = header["HubbleParam"]
z = header["Redshift"]
a = 1.0 / (1.0 + z)

fields = [
    "Group_M_Crit200",
    "Group_R_Crit200",
    "GroupFirstSub",
    "GroupNsubs",
]

groups = il.groupcat.loadHalos(basePath, snapNum, fields=fields)

M200c_msun = groups["Group_M_Crit200"] * 1e10 / h
R200c_kpc = groups["Group_R_Crit200"] * a / h
first_sub = groups["GroupFirstSub"]
Nsubs = groups["GroupNsubs"]

logM = np.full_like(M200c_msun, np.nan, dtype=float)
ok = M200c_msun > 0
logM[ok] = np.log10(M200c_msun[ok])

valid = (
    np.isfinite(logM)
    & (first_sub >= 0)
    & (Nsubs > 0)
    & np.isfinite(R200c_kpc)
    & (R200c_kpc > 0)
)

summary_rows = []

print("")
print("Central halo counts")
print("Using Group_M_Crit200 and GroupFirstSub")
print("")

for target in targets:
    lo = target - delta_logM
    hi = target + delta_logM

    mask = valid & (logM >= lo) & (logM < hi)
    chosen = np.where(mask)[0]

    summary_rows.append({
        "target_logM": target,
        "low": lo,
        "high": hi,
        "N_central_halos": len(chosen),
    })

    print(f"logM200c = {target:.1f} +/- {delta_logM:.2f}")
    print(f"range: [{lo:.2f}, {hi:.2f})")
    print(f"N central halos = {len(chosen)}")
    print("")

df = pd.DataFrame(summary_rows)
out_csv = "TNG_work/mass_bin_counts_11p5_12_12p5_13.csv"
df.to_csv(out_csv, index=False)

print("Saved:")
print(out_csv)
print("")
print(df)
