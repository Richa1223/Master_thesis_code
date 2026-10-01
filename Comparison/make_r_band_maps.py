from pathlib import Path
import gc

import numpy as np
import pandas as pd
import h5py
import illustris_python as il


# ============================================================
# CONFIG
# ============================================================

BASE = Path("/home/tnguser")
COMP = BASE / "HSC_TNG_comparison"

SELECTED_FILE = COMP / "all11_selected_5_TNG_analogues.csv"

OUTDIR = COMP / "all11_rband_maps"
OUTDIR.mkdir(parents=True, exist_ok=True)

MANIFEST_FILE = COMP / "all11_rband_map_manifest.csv"

basePath = "/home/tnguser/sims.TNG/L35n2160TNG/output"
snapNum = 99

# Official TNG photometric ordering:
# U, B, V, K, g, r, i, z
R_BAND_INDEX = 5

M_SUN_R = 4.65

PROJECTIONS = {
    "xy": (0, 1),
    "xz": (0, 2),
    "yz": (1, 2),
}

OVERWRITE = False


# ============================================================
# HELPERS
# ============================================================

def sigmaL_to_mu(Sigma_L_kpc2, M_sun_r=M_SUN_R):

    I_pc2 = np.asarray(Sigma_L_kpc2, dtype=float) / 1e6

    mu = np.full_like(I_pc2, np.nan, dtype=float)

    ok = np.isfinite(I_pc2) & (I_pc2 > 0)

    mu[ok] = (
        M_sun_r
        + 21.572
        - 2.5 * np.log10(I_pc2[ok])
    )

    return mu


def make_surface_density_temet(
    coords2d_kpc,
    hsml_kpc,
    weights,
    extent_kpc,
    nbins
):

    from temet.util.sphMap import sphMap

    pos = np.asarray(coords2d_kpc, dtype=np.float32)
    hsml = np.asarray(hsml_kpc, dtype=np.float32)
    weights = np.asarray(weights, dtype=np.float32)

    good = (
        np.isfinite(pos[:, 0])
        & np.isfinite(pos[:, 1])
        & np.isfinite(hsml)
        & np.isfinite(weights)
        & (hsml > 0)
        & (weights > 0)
    )

    pos = pos[good]
    hsml = hsml[good]
    weights = weights[good]

    if len(pos) == 0:
        raise RuntimeError("No valid stellar particles for SPH map.")

    box_size = 2.0 * float(extent_kpc)

    Sigma = sphMap(
        pos=pos,
        hsml=hsml,
        mass=weights,
        quant=None,
        axes=[0, 1],
        boxSizeImg=[box_size, box_size],
        boxSizeSim=0.0,
        boxCen=[0.0, 0.0],
        nPixels=[int(nbins), int(nbins)],
        ndims=2,
        colDens=True,
        nThreads=None
    )

    Sigma = np.asarray(Sigma, dtype=float)

    edges = np.linspace(
        -extent_kpc,
        extent_kpc,
        int(nbins) + 1
    )

    centers = 0.5 * (
        edges[:-1] + edges[1:]
    )

    return Sigma, centers, centers


def validate_existing_output(path):

    try:
        with h5py.File(path, "r") as f:

            if "meta" not in f:
                return False

            idx = int(
                f["meta"].attrs.get(
                    "photometric_index",
                    -999
                )
            )

            if idx != R_BAND_INDEX:
                return False

            for proj in PROJECTIONS:

                if proj not in f:
                    return False

                g = f[proj]

                for name in [
                    "x_centers_kpc",
                    "y_centers_kpc",
                    "SigmaL_Lsun_kpc2",
                    "mu_r_mag_arcsec2",
                ]:
                    if name not in g:
                        return False

        return True

    except Exception:
        return False


# ============================================================
# LOAD INPUT SELECTION
# ============================================================

if not SELECTED_FILE.exists():
    raise FileNotFoundError(SELECTED_FILE)

sel = pd.read_csv(SELECTED_FILE)

required = [
    "SubfindID",
    "output_path",
    "R200c_kpc",
]

for col in required:
    if col not in sel.columns:
        raise KeyError(
            f"Missing '{col}' from {SELECTED_FILE}"
        )


# One row per unique simulated galaxy.
unique = (
    sel
    .sort_values("SubfindID")
    .drop_duplicates("SubfindID")
    .copy()
)

print("=" * 72)
print("CORRECTED r-BAND MAP PRODUCTION")
print("=" * 72)

print("Analogue assignments:", len(sel))
print("Unique TNG galaxies:", len(unique))
print("Photometric index:", R_BAND_INDEX)
print("Band: r")
print("Projections:", list(PROJECTIONS))
print()


# ============================================================
# TNG HEADER
# ============================================================

header = il.groupcat.loadHeader(
    basePath,
    snapNum
)

h = float(header["HubbleParam"])
z = float(header["Redshift"])
a = 1.0 / (1.0 + z)

print(f"snapshot = {snapNum}")
print(f"z = {z:.6f}")
print(f"h = {h:.6f}")
print()


# ============================================================
# PROCESS UNIQUE ANALOGUES
# ============================================================

manifest_rows = []


for ii, row in unique.reset_index(drop=True).iterrows():

    sid = int(row["SubfindID"])

    old_output_dir = Path(str(row["output_path"]))

    if not old_output_dir.is_absolute():
        old_output_dir = BASE / old_output_dir

    old_stars_h5 = (
        old_output_dir /
        f"stars_sub{sid}.hdf5"
    )

    new_h5 = (
        OUTDIR /
        f"all11_rband_sub{sid}.hdf5"
    )

    print()
    print("=" * 72)
    print(
        f"[{ii+1}/{len(unique)}] "
        f"SubfindID {sid}"
    )
    print("=" * 72)

    if not old_stars_h5.exists():
        raise FileNotFoundError(
            f"Original stellar HDF5 not found: "
            f"{old_stars_h5}"
        )

    with h5py.File(old_stars_h5, "r") as f_old:

        if "meta" not in f_old:
            raise KeyError(
                f"No meta group in {old_stars_h5}"
            )

        meta_old = f_old["meta"].attrs

        extent_kpc = float(
            meta_old["extent_kpc"]
        )

        nbins = int(
            meta_old["nbins"]
        )

        R200c_old = float(
            meta_old.get(
                "R200c_kpc",
                row["R200c_kpc"]
            )
        )

        old_phot_index = int(
            meta_old.get(
                "r_band_index",
                -1
            )
        )

    print(
        f"Original geometry: "
        f"extent = {extent_kpc:.3f} kpc, "
        f"nbins = {nbins}"
    )

    print(
        f"Original stored photometric index = "
        f"{old_phot_index}"
    )

    if new_h5.exists() and not OVERWRITE:

        if validate_existing_output(new_h5):

            print(
                "Corrected map already exists "
                "and passed validation -> SKIP"
            )

            manifest_rows.append({
                "SubfindID": sid,
                "status": "existing_valid",
                "output_file": str(new_h5),
                "extent_kpc": extent_kpc,
                "nbins": nbins,
                "photometric_index": R_BAND_INDEX,
                "old_photometric_index":
                    old_phot_index,
            })

            continue

        else:

            raise RuntimeError(
                f"{new_h5} exists but failed validation. "
                "Not overwriting automatically."
            )

    # --------------------------------------------------------
    # Load this subhalo's stars.
    # --------------------------------------------------------

    sub = il.groupcat.loadSingle(
        basePath,
        snapNum,
        subhaloID=sid
    )

    subpos_kpc = (
        np.asarray(
            sub["SubhaloPos"],
            dtype=float
        )
        * a / h
    )

    fields = [
        "Coordinates",
        "StellarHsml",
        "GFM_StellarPhotometrics",
    ]

    stars = il.snapshot.loadSubhalo(
        basePath,
        snapNum,
        sid,
        "star",
        fields=fields
    )

    if int(stars["count"]) == 0:
        raise RuntimeError(
            f"Subhalo {sid} has no stellar particles."
        )

    coords_kpc = (
        np.asarray(
            stars["Coordinates"],
            dtype=float
        )
        * a / h
        - subpos_kpc
    )

    hsml_kpc = (
        np.asarray(
            stars["StellarHsml"],
            dtype=float
        )
        * a / h
    )

    phot = np.asarray(
        stars["GFM_StellarPhotometrics"],
        dtype=float
    )

    if phot.ndim != 2:
        raise RuntimeError(
            f"Unexpected photometry shape: {phot.shape}"
        )

    if phot.shape[1] <= R_BAND_INDEX:
        raise RuntimeError(
            f"Photometry has only {phot.shape[1]} bands; "
            f"cannot access index {R_BAND_INDEX}."
        )

    mag_r = phot[:, R_BAND_INDEX]

    L_r = np.full(
        len(mag_r),
        np.nan,
        dtype=float
    )

    good_mag = np.isfinite(mag_r)

    L_r[good_mag] = (
        10.0
        ** (
            -0.4
            * (
                mag_r[good_mag]
                - M_SUN_R
            )
        )
    )

    good_L = (
        np.isfinite(L_r)
        & (L_r > 0)
    )

    if np.count_nonzero(good_L) == 0:
        raise RuntimeError(
            f"Subhalo {sid}: no valid r-band luminosities."
        )

    print(
        "Stellar particles loaded:",
        len(coords_kpc)
    )

    print(
        "Valid r-band luminosities:",
        np.count_nonzero(good_L)
    )

    # --------------------------------------------------------
    # Save corrected maps.
    # --------------------------------------------------------

    with h5py.File(new_h5, "w") as f:

        meta = f.create_group("meta")

        meta.attrs.update({
            "SubfindID": sid,
            "snapNum": snapNum,
            "band_name": "r",
            "photometric_index": R_BAND_INDEX,
            "photometric_order":
                "U,B,V,K,g,r,i,z",
            "M_sun_r": M_SUN_R,
            "R200c_kpc": R200c_old,
            "extent_kpc": extent_kpc,
            "nbins": nbins,
            "projections": "xy,xz,yz",
            "source_old_stars_h5":
                str(old_stars_h5),
            "old_stored_photometric_index":
                old_phot_index,
            "product":
                "corrected_all11_rband_only",
        })

        meta.attrs["N_stars_loaded"] = int(
            len(coords_kpc)
        )

        meta.attrs["N_valid_rband"] = int(
            np.count_nonzero(good_L)
        )

        meta.attrs["Lr_total_valid_Lsun"] = float(
            np.sum(L_r[good_L])
        )

        for proj, axes in PROJECTIONS.items():

            print("  making", proj)

            coords2d = coords_kpc[
                :,
                list(axes)
            ]

            SigmaL, xcent, ycent = (
                make_surface_density_temet(
                    coords2d_kpc=coords2d,
                    hsml_kpc=hsml_kpc,
                    weights=L_r,
                    extent_kpc=extent_kpc,
                    nbins=nbins,
                )
            )

            mu_r = sigmaL_to_mu(
                SigmaL
            )

            g = f.create_group(proj)

            g.attrs["projection_mode"] = (
                "simulation_axis"
            )

            g.create_dataset(
                "x_centers_kpc",
                data=xcent
            )

            g.create_dataset(
                "y_centers_kpc",
                data=ycent
            )

            g.create_dataset(
                "SigmaL_Lsun_kpc2",
                data=SigmaL
            )

            g.create_dataset(
                "mu_r_mag_arcsec2",
                data=mu_r
            )

            pixel_size = (
                2.0 * extent_kpc / nbins
            )

            L_map = float(
                np.nansum(SigmaL)
                * pixel_size**2
            )

            g.attrs[
                "integrated_map_Lsun"
            ] = L_map

            g.attrs[
                "N_positive_map_pixels"
            ] = int(
                np.count_nonzero(
                    np.isfinite(SigmaL)
                    & (SigmaL > 0)
                )
            )

            del coords2d
            del SigmaL
            del mu_r

            gc.collect()


    if not validate_existing_output(new_h5):
        raise RuntimeError(
            f"Validation failed after writing {new_h5}"
        )

    print("Saved:", new_h5)

    manifest_rows.append({
        "SubfindID": sid,
        "status": "created",
        "output_file": str(new_h5),
        "extent_kpc": extent_kpc,
        "nbins": nbins,
        "photometric_index": R_BAND_INDEX,
        "old_photometric_index":
            old_phot_index,
    })

    del stars
    del coords_kpc
    del hsml_kpc
    del phot
    del mag_r
    del L_r

    gc.collect()

manifest = pd.DataFrame(
    manifest_rows
)

manifest.to_csv(
    MANIFEST_FILE,
    index=False
)

print()
print("=" * 72)
print("DONE")
print("=" * 72)

print(
    "Corrected r-band files:",
    len(manifest)
)

print(
    "Manifest:",
    MANIFEST_FILE
)

print(
    "Output directory:",
    OUTDIR
)
