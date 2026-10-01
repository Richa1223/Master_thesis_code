from pathlib import Path
import numpy as np
import pandas as pd


# ============================================================
# PATHS
# ============================================================

BASE = Path("/home/tnguser")
COMP = BASE / "HSC_TNG_comparison"

COMP.mkdir(parents=True, exist_ok=True)

MASTER_FILE = BASE / "master_massbin_TNG_catalog.csv"
HSC_FILE = BASE / "final11_stellar_mass_morphology_table.csv"

OUT_MATCHES = COMP / "all11_selected_5_TNG_analogues.csv"
OUT_MANIFEST = COMP / "all11_HSC_input_manifest.csv"
OUT_SUMMARY = COMP / "all11_mass_match_summary.csv"


# ============================================================
# BASIC CHECKS
# ============================================================

if not MASTER_FILE.exists():
    raise FileNotFoundError(MASTER_FILE)

if not HSC_FILE.exists():
    raise FileNotFoundError(HSC_FILE)


master = pd.read_csv(MASTER_FILE)
hsc = pd.read_csv(HSC_FILE)


required_master = [
    "SubfindID",
    "mass_bin",
    "output_path",
    "logMstar_TNG",
    "logM200c",
    "R200c_kpc",
]

required_hsc = [
    "galaxy",
    "logMstar_Mizuki",
    "reliable_radius_kpc",
]


for col in required_master:
    if col not in master.columns:
        raise KeyError(
            f"Missing column '{col}' from {MASTER_FILE}"
        )

for col in required_hsc:
    if col not in hsc.columns:
        raise KeyError(
            f"Missing column '{col}' from {HSC_FILE}"
        )


if len(hsc) != 11:
    raise RuntimeError(
        f"Expected 11 HSC galaxies, found {len(hsc)}"
    )

if hsc["galaxy"].duplicated().any():
    raise RuntimeError(
        "Duplicate galaxy IDs found in final11 table."
    )


print("=" * 70)
print("INPUT CHECK")
print("=" * 70)

print("TNG galaxies:", len(master))
print("HSC galaxies:", len(hsc))

print()


# ============================================================
# CHECK THE 11 HSC SURFACE-BRIGHTNESS PROFILE FILES
# ============================================================

profile_rows = []

required_profile_cols = [
    "a_inner_arcsec",
    "a_mid_arcsec",
    "a_outer_arcsec",
    "a_mid_kpc",
    "xerr_kpc",
    "mu_clipped_mean",
    "mu_clipped_mean_err",
    "mu_clipped_median",
    "unmasked_fraction",
]


for _, row in hsc.iterrows():

    gid = int(row["galaxy"])

    profile_file = BASE / f"galaxy{gid}_profile_kpc.csv"

    if not profile_file.exists():
        raise FileNotFoundError(
            f"Missing profile for HSC-{gid}: {profile_file}"
        )

    prof = pd.read_csv(profile_file)

    for col in required_profile_cols:
        if col not in prof.columns:
            raise KeyError(
                f"{profile_file} does not contain '{col}'"
            )

    Rrel = float(row["reliable_radius_kpc"])

    finite = (
        np.isfinite(prof["a_mid_kpc"]) &
        np.isfinite(prof["mu_clipped_mean"])
    )

    finite_reliable = (
        finite &
        (prof["a_mid_kpc"] <= Rrel)
    )

    profile_rows.append({
        "HSC_ID": f"HSC-{gid}",
        "galaxy": gid,
        "profile_file": str(profile_file),
        "logMstar_obs": float(row["logMstar_Mizuki"]),
        "reliable_radius_kpc": Rrel,
        "N_profile_rows": len(prof),
        "N_finite_SB": int(finite.sum()),
        "N_finite_SB_within_reliable_radius": int(
            finite_reliable.sum()
        ),
    })


manifest = pd.DataFrame(profile_rows)

manifest.to_csv(
    OUT_MANIFEST,
    index=False
)


print("All 11 HSC profile CSVs found successfully.")
print()
print(manifest.to_string(index=False))
print()


# ============================================================
# STELLAR-MASS RANGE OF EXISTING TNG MASTER CATALOGUE
# ============================================================

tng_min = float(master["logMstar_TNG"].min())
tng_max = float(master["logMstar_TNG"].max())

print("=" * 70)
print("TNG STELLAR-MASS COVERAGE")
print("=" * 70)

print(f"Minimum logMstar_TNG = {tng_min:.3f}")
print(f"Maximum logMstar_TNG = {tng_max:.3f}")
print()


# ============================================================
# SELECT 5 CLOSEST TNG GALAXIES BY STELLAR MASS ONLY
# ============================================================

all_selected = []
summary_rows = []


for _, obsrow in hsc.iterrows():

    gid = int(obsrow["galaxy"])
    hsc_id = f"HSC-{gid}"

    logM_obs = float(obsrow["logMstar_Mizuki"])
    Rrel = float(obsrow["reliable_radius_kpc"])

    temp = master.copy()

    temp["delta_logMstar"] = np.abs(
        temp["logMstar_TNG"] - logM_obs
    )

    selected = (
        temp
        .sort_values(
            ["delta_logMstar", "SubfindID"]
        )
        .head(5)
        .copy()
    )

    selected.insert(
        0,
        "HSC_ID",
        hsc_id
    )

    selected.insert(
        1,
        "match_rank",
        np.arange(1, len(selected) + 1)
    )

    selected.insert(
        2,
        "logMstar_obs",
        logM_obs
    )

    selected.insert(
        3,
        "reliable_radius_kpc",
        Rrel
    )

    selected["outside_TNG_master_mass_range"] = (
        (logM_obs < tng_min) |
        (logM_obs > tng_max)
    )

    # Store an absolute path as well.
    selected["output_path_abs"] = selected[
        "output_path"
    ].apply(
        lambda p: str(BASE / str(p))
    )

    all_selected.append(selected)

    delta1 = float(
        selected.iloc[0]["delta_logMstar"]
    )

    delta5 = float(
        selected.iloc[-1]["delta_logMstar"]
    )

    summary_rows.append({
        "HSC_ID": hsc_id,
        "logMstar_obs": logM_obs,
        "nearest_TNG_logMstar":
            float(selected.iloc[0]["logMstar_TNG"]),
        "delta_logMstar_rank1": delta1,
        "delta_logMstar_rank5": delta5,
        "outside_TNG_master_mass_range":
            bool(
                logM_obs < tng_min or
                logM_obs > tng_max
            ),
    })


selected_all = pd.concat(
    all_selected,
    ignore_index=True
)

summary = pd.DataFrame(summary_rows)


# ============================================================
# FINAL VALIDATION
# ============================================================

expected_rows = 11 * 5

if len(selected_all) != expected_rows:
    raise RuntimeError(
        f"Expected {expected_rows} selected rows, "
        f"found {len(selected_all)}"
    )

counts = selected_all.groupby(
    "HSC_ID"
).size()

if not np.all(counts.values == 5):
    raise RuntimeError(
        "Not every HSC galaxy has exactly five matches."
    )


# ============================================================
# SAVE
# ============================================================

selected_all.to_csv(
    OUT_MATCHES,
    index=False
)

summary.to_csv(
    OUT_SUMMARY,
    index=False
)


print("=" * 70)
print("FINAL MASS-MATCH SELECTION")
print("=" * 70)

for hsc_id, group in selected_all.groupby(
    "HSC_ID",
    sort=False
):

    print()
    print(hsc_id)

    print(
        group[
            [
                "match_rank",
                "SubfindID",
                "logMstar_obs",
                "logMstar_TNG",
                "delta_logMstar",
                "logM200c",
                "R200c_kpc",
                "mass_bin",
            ]
        ].to_string(index=False)
    )


print()
print("=" * 70)
print("MATCH SUMMARY")
print("=" * 70)

print(
    summary.to_string(index=False)
)


outside = summary[
    summary["outside_TNG_master_mass_range"]
]

if len(outside) > 0:

    print()
    print("!" * 70)
    print("WARNING")
    print("!" * 70)

    print(
        "The following HSC galaxies lie outside "
        "the stellar-mass range covered by the existing "
        "991-galaxy TNG catalogue:"
    )

    print()

    print(
        outside[
            [
                "HSC_ID",
                "logMstar_obs",
                "nearest_TNG_logMstar",
                "delta_logMstar_rank1",
            ]
        ].to_string(index=False)
    )

    print()
    print(
        "These objects have nearest AVAILABLE TNG galaxies, "
        "but should not automatically be described as "
        "well stellar-mass-matched analogues."
    )


print()
print("Saved:")
print(OUT_MATCHES)
print(OUT_SUMMARY)
print(OUT_MANIFEST)

print()
print("DONE.")
