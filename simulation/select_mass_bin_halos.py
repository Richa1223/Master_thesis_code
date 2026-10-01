import os
import numpy as np
import pandas as pd
import illustris_python as il


# ============================================================
# CONFIG
# ============================================================

basePath = "/home/tnguser/sims.TNG/L35n2160TNG/output"
snapNum = 99

out_csv = "TNG_work/selected_mass_bin_halos.csv"


target_logM_bins = [11.5, 12.0, 12.5, 13.0]

# Width around each target mass bin.
# Example: 12.5 means select 12.35 < logM < 12.65
delta_logM = 0.25


Npick = 5

random_seed = 42


# LOAD GROUP CATALOG

print("Loading header...")
header = il.groupcat.loadHeader(basePath, snapNum)

h = header["HubbleParam"]
z = header["Redshift"]
a = 1.0 / (1.0 + z)

print(f"h = {h}")
print(f"z = {z}")

fields = [
    "Group_M_Crit200",
    "Group_R_Crit200",
    "GroupFirstSub",
    "GroupNsubs",
]

print("Loading FoF group catalog...")
groups = il.groupcat.loadHalos(
    basePath,
    snapNum,
    fields=fields
)

M200c_code = groups["Group_M_Crit200"]
R200c_code = groups["Group_R_Crit200"]
first_sub = groups["GroupFirstSub"]
Nsubs = groups["GroupNsubs"]

# TNG units:
# Group_M_Crit200 is in 1e10 Msun/h
# Group_R_Crit200 is in ckpc/h
M200c_msun = M200c_code * 1e10 / h
R200c_kpc = R200c_code * a / h

logM = np.full_like(M200c_msun, np.nan, dtype=float)
ok_mass = M200c_msun > 0
logM[ok_mass] = np.log10(M200c_msun[ok_mass])

group_ids = np.arange(len(M200c_msun))

# Only keep groups with a valid central subhalo
valid = (
    np.isfinite(logM)
    & (first_sub >= 0)
    & (Nsubs > 0)
    & np.isfinite(R200c_kpc)
    & (R200c_kpc > 0)
)

rng = np.random.default_rng(random_seed)

rows = []

print("\nSelecting halos...\n")

for target in target_logM_bins:
    lo = target - delta_logM
    hi = target + delta_logM

    m = valid & (logM >= lo) & (logM < hi)

    candidates = np.where(m)[0]

    print(f"Mass bin logM ~ {target:.1f}:")
    print(f"  range = [{lo:.2f}, {hi:.2f})")
    print(f"  candidates = {len(candidates)}")

    if len(candidates) == 0:
        print("  WARNING: no candidates found in this bin.\n")
        continue

    nsel = min(Npick, len(candidates))

    chosen = rng.choice(candidates, size=nsel, replace=False)

    # Sort by mass for readability
    chosen = chosen[np.argsort(logM[chosen])]

    for gid in chosen:
        rows.append({
            "target_logM_bin": target,
            "GroupID": int(gid),
            "hydro_subhaloID": int(first_sub[gid]),
            "logM200c": float(logM[gid]),
            "M200c_Msun": float(M200c_msun[gid]),
            "R200c_kpc": float(R200c_kpc[gid]),
            "GroupNsubs": int(Nsubs[gid]),
        })

        print(
            f"  GroupID={gid:6d} | "
            f"SubhaloID={int(first_sub[gid]):7d} | "
            f"logM={logM[gid]:.3f} | "
            f"R200c={R200c_kpc[gid]:.3f} kpc | "
            f"Nsubs={int(Nsubs[gid])}"
        )

    print("")

df = pd.DataFrame(rows)

os.makedirs(os.path.dirname(out_csv), exist_ok=True)
df.to_csv(out_csv, index=False)

print("Saved:")
print(out_csv)

print("\nFinal selected sample:")
print(df)
