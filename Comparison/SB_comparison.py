from pathlib import Path

import numpy as np
import pandas as pd
import h5py
import matplotlib.pyplot as plt


# ============================================================
# CONFIG
# ============================================================

BASE = Path("/home/tnguser")
COMP = BASE / "HSC_TNG_comparison"
MAPDIR = COMP / "all11_rband_maps"
PLOTDIR = COMP / "final_plots"

PLOTDIR.mkdir(parents=True, exist_ok=True)

HSC_FILE = BASE / "final11_stellar_mass_morphology_table.csv"
MATCH_FILE = COMP / "all11_selected_5_TNG_analogues.csv"

OUT_LONG = (
    COMP /
    "all11_HSC_TNG_SB_individual_realizations.csv"
)

OUT_VALUES = (
    PLOTDIR /
    "all11_HSC_TNG_SB_comparison_values.csv"
)

OUT_GEOMETRY = (
    COMP /
    "all11_TNG_rband_ellipse_geometry.csv"
)

OUT_PNG = (
    PLOTDIR /
    "all11_HSC_TNG_SB_comparison.png"
)

OUT_PDF = (
    PLOTDIR /
    "all11_HSC_TNG_SB_comparison.pdf"
)


PROJECTIONS = ["xy", "xz", "yz"]

M_SUN_R = 4.65
EXPECTED_PHOT_INDEX = 5


# ============================================================
# HELPERS
# ============================================================

def sigmaL_to_mu(SigmaL_Lsun_kpc2):

    SigmaL_Lsun_kpc2 = np.asarray(
        SigmaL_Lsun_kpc2,
        dtype=float
    )

    I_Lsun_pc2 = SigmaL_Lsun_kpc2 / 1e6

    mu = np.full_like(
        I_Lsun_pc2,
        np.nan,
        dtype=float
    )

    ok = (
        np.isfinite(I_Lsun_pc2)
        & (I_Lsun_pc2 > 0)
    )

    mu[ok] = (
        M_SUN_R
        + 21.572
        - 2.5 * np.log10(
            I_Lsun_pc2[ok]
        )
    )

    return mu


def measure_light_q_pa(
    SigmaL,
    xcent,
    ycent,
    Rmax_kpc
):
    

    X, Y = np.meshgrid(
        xcent,
        ycent,
        indexing="ij"
    )

    R = np.sqrt(
        X**2 + Y**2
    )

    W = np.asarray(
        SigmaL,
        dtype=float
    )

    good = (
        np.isfinite(W)
        & (W > 0)
        & np.isfinite(X)
        & np.isfinite(Y)
        & (R <= Rmax_kpc)
    )

    Npix = int(
        np.count_nonzero(good)
    )

    if Npix < 10:
        raise RuntimeError(
            f"Too few positive pixels inside "
            f"Rmax={Rmax_kpc:.3f} kpc: "
            f"N={Npix}"
        )

    x = X[good]
    y = Y[good]
    w = W[good]

    wsum = np.sum(w)

    if not np.isfinite(wsum) or wsum <= 0:
        raise RuntimeError(
            "Invalid luminosity sum for q/PA."
        )

    # Fixed centre = (0,0), because maps were already
    # centred on SubhaloPos.
    Cxx = np.sum(w * x * x) / wsum
    Cyy = np.sum(w * y * y) / wsum
    Cxy = np.sum(w * x * y) / wsum

    cov = np.array([
        [Cxx, Cxy],
        [Cxy, Cyy]
    ])

    eigvals, eigvecs = np.linalg.eigh(cov)

    if (
        not np.all(np.isfinite(eigvals))
        or eigvals[0] <= 0
        or eigvals[1] <= 0
    ):
        raise RuntimeError(
            "Invalid second-moment eigenvalues."
        )

    q = np.sqrt(
        eigvals[0] / eigvals[1]
    )

    q = float(
        np.clip(q, 0.05, 1.0)
    )

    # Major-axis eigenvector
    vx, vy = eigvecs[:, 1]

    pa_rad = np.arctan2(
        vy,
        vx
    )

    pa_deg = np.degrees(
        pa_rad
    )

    # PA has 180-degree symmetry.
    while pa_deg >= 90:
        pa_deg -= 180

    while pa_deg < -90:
        pa_deg += 180

    return q, float(pa_rad), float(pa_deg), Npix


def elliptical_profile_in_hsc_bins(
    SigmaL,
    xcent,
    ycent,
    q,
    pa_rad,
    a_inner_kpc,
    a_outer_kpc
):
  
    X, Y = np.meshgrid(
        xcent,
        ycent,
        indexing="ij"
    )

    c = np.cos(pa_rad)
    s = np.sin(pa_rad)

    # Rotate into major-axis frame.
    Xp = c * X + s * Y
    Yp = -s * X + c * Y

    a_ell = np.sqrt(
        Xp**2
        + (Yp / q)**2
    )

    S = np.asarray(
        SigmaL,
        dtype=float
    )

    positive = (
        np.isfinite(S)
        & (S > 0)
    )

    Sigma_prof = np.full(
        len(a_inner_kpc),
        np.nan
    )

    Npix_prof = np.zeros(
        len(a_inner_kpc),
        dtype=int
    )

    for i, (ain, aout) in enumerate(
        zip(
            a_inner_kpc,
            a_outer_kpc
        )
    ):

        if (
            not np.isfinite(ain)
            or not np.isfinite(aout)
            or aout <= ain
        ):
            continue

        m = (
            positive
            & (a_ell >= ain)
            & (a_ell < aout)
        )

        Npix = int(
            np.count_nonzero(m)
        )

        Npix_prof[i] = Npix

        if Npix == 0:
            continue

        Sigma_prof[i] = np.mean(
            S[m]
        )

    mu_prof = sigmaL_to_mu(
        Sigma_prof
    )

    return Sigma_prof, mu_prof, Npix_prof


def finite_percentiles(values):

    arr = np.asarray(
        values,
        dtype=float
    )

    good = np.isfinite(arr)

    if np.count_nonzero(good) == 0:
        return (
            np.nan,
            np.nan,
            np.nan,
            0
        )

    x = arr[good]

    return (
        float(np.nanmedian(x)),
        float(np.nanpercentile(x, 16)),
        float(np.nanpercentile(x, 84)),
        int(len(x))
    )


# ============================================================
# LOAD TABLES
# ============================================================

if not HSC_FILE.exists():
    raise FileNotFoundError(HSC_FILE)

if not MATCH_FILE.exists():
    raise FileNotFoundError(MATCH_FILE)

hsc = pd.read_csv(HSC_FILE)
matches = pd.read_csv(MATCH_FILE)


if len(hsc) != 11:
    raise RuntimeError(
        f"Expected 11 HSC galaxies, found {len(hsc)}"
    )


print("=" * 78)
print("ALL-11 HSC vs TNG CORRECTED r-BAND SB COMPARISON")
print("=" * 78)
print("HSC galaxies:", len(hsc))
print("Analogue assignments:", len(matches))
print("Projections:", PROJECTIONS)
print()


# ============================================================
# PROCESS ALL 11 HSC GALAXIES
# ============================================================

long_rows = []
geometry_rows = []
summary_rows = []


for _, hrow in hsc.iterrows():

    gid = int(
        hrow["galaxy"]
    )

    hsc_id = f"HSC-{gid}"

    logMstar_obs = float(
        hrow["logMstar_Mizuki"]
    )

    Rreliable = float(
        hrow["reliable_radius_kpc"]
    )

    profile_file = (
        BASE /
        f"galaxy{gid}_profile_kpc.csv"
    )

    if not profile_file.exists():
        raise FileNotFoundError(
            profile_file
        )

    prof = pd.read_csv(
        profile_file
    )

    required_cols = [
        "a_mid_kpc",
        "xerr_kpc",
        "mu_clipped_mean",
        "mu_clipped_mean_err",
    ]

    for col in required_cols:
        if col not in prof.columns:
            raise KeyError(
                f"{profile_file}: "
                f"missing '{col}'"
            )

    a_mid = prof[
        "a_mid_kpc"
    ].to_numpy(dtype=float)

    xerr = prof[
        "xerr_kpc"
    ].to_numpy(dtype=float)

    # Recover the physical annulus boundaries
    # represented by the horizontal HSC error bars.
    a_inner = np.maximum(
        a_mid - xerr,
        0.0
    )

    a_outer = (
        a_mid + xerr
    )

    mu_hsc = prof[
        "mu_clipped_mean"
    ].to_numpy(dtype=float)

    mu_hsc_err = prof[
        "mu_clipped_mean_err"
    ].to_numpy(dtype=float)

    hsc_reliable = (
        np.isfinite(a_mid)
        & (a_mid <= Rreliable)
    )

    hsc_finite_reliable = (
        hsc_reliable
        & np.isfinite(mu_hsc)
    )


    msel = matches[
        matches["HSC_ID"] == hsc_id
    ].copy()

    if len(msel) != 5:
        raise RuntimeError(
            f"{hsc_id}: expected 5 matches, "
            f"found {len(msel)}"
        )


    print()
    print("-" * 78)
    print(
        hsc_id,
        f"| logM*={logMstar_obs:.3f}",
        f"| Rreliable={Rreliable:.2f} kpc"
    )
    print("-" * 78)


    # --------------------------------------------------------
    # 5 analogues x 3 projections = 15 realizations
    # --------------------------------------------------------

    for _, mrow in msel.iterrows():

        sid = int(
            mrow["SubfindID"]
        )

        rank = int(
            mrow["match_rank"]
        )

        mapfile = (
            MAPDIR /
            f"all11_rband_sub{sid}.hdf5"
        )

        if not mapfile.exists():
            raise FileNotFoundError(
                mapfile
            )


        with h5py.File(
            mapfile,
            "r"
        ) as f:

            if "meta" not in f:
                raise KeyError(
                    f"{mapfile}: missing meta"
                )

            phot_index = int(
                f["meta"].attrs[
                    "photometric_index"
                ]
            )

            if (
                phot_index
                != EXPECTED_PHOT_INDEX
            ):
                raise RuntimeError(
                    f"{mapfile}: photometric index "
                    f"is {phot_index}, expected 5."
                )


            for proj in PROJECTIONS:

                if proj not in f:
                    raise KeyError(
                        f"{mapfile}: "
                        f"missing projection {proj}"
                    )

                g = f[proj]

                x = g[
                    "x_centers_kpc"
                ][:]

                y = g[
                    "y_centers_kpc"
                ][:]

                SigmaL = g[
                    "SigmaL_Lsun_kpc2"
                ][:]

                q_light, pa_rad, pa_deg, Nshape = (
                    measure_light_q_pa(
                        SigmaL=SigmaL,
                        xcent=x,
                        ycent=y,
                        Rmax_kpc=Rreliable
                    )
                )


                Sigma_prof, mu_tng, Npix_prof = (
                    elliptical_profile_in_hsc_bins(
                        SigmaL=SigmaL,
                        xcent=x,
                        ycent=y,
                        q=q_light,
                        pa_rad=pa_rad,
                        a_inner_kpc=a_inner,
                        a_outer_kpc=a_outer
                    )
                )


                geometry_rows.append({
                    "HSC_ID": hsc_id,
                    "SubfindID": sid,
                    "match_rank": rank,
                    "projection": proj,
                    "R_shape_kpc":
                        Rreliable,
                    "q_light_for_SB_annuli":
                        q_light,
                    "PA_light_deg_for_SB_annuli":
                        pa_deg,
                    "N_pixels_shape_measurement":
                        Nshape,
                })


                for ibin in range(
                    len(a_mid)
                ):

                    long_rows.append({
                        "HSC_ID":
                            hsc_id,
                        "galaxy":
                            gid,
                        "logMstar_obs":
                            logMstar_obs,
                        "SubfindID":
                            sid,
                        "match_rank":
                            rank,
                        "projection":
                            proj,
                        "bin_index":
                            ibin,
                        "a_inner_kpc":
                            a_inner[ibin],
                        "a_mid_kpc":
                            a_mid[ibin],
                        "a_outer_kpc":
                            a_outer[ibin],
                        "xerr_kpc":
                            xerr[ibin],
                        "mu_HSC":
                            mu_hsc[ibin],
                        "mu_HSC_err":
                            mu_hsc_err[ibin],
                        "reliable_radius_kpc":
                            Rreliable,
                        "HSC_bin_within_reliable_radius":
                            bool(
                                hsc_reliable[
                                    ibin
                                ]
                            ),
                        "HSC_finite_and_reliable":
                            bool(
                                hsc_finite_reliable[
                                    ibin
                                ]
                            ),
                        "q_light_for_SB_annuli":
                            q_light,
                        "PA_light_deg_for_SB_annuli":
                            pa_deg,
                        "SigmaL_TNG_Lsun_kpc2":
                            Sigma_prof[
                                ibin
                            ],
                        "mu_TNG_r":
                            mu_tng[
                                ibin
                            ],
                        "N_TNG_pixels_in_annulus":
                            int(
                                Npix_prof[
                                    ibin
                                ]
                            ),
                    })

    current = pd.DataFrame(
        [
            r for r in long_rows
            if r["HSC_ID"] == hsc_id
        ]
    )


    for ibin in range(
        len(a_mid)
    ):

        b = current[
            current["bin_index"]
            == ibin
        ]

        (
            med,
            p16,
            p84,
            Nfinite
        ) = finite_percentiles(
            b["mu_TNG_r"]
        )

        outside_mass_range = bool(
            msel[
                "outside_TNG_master_mass_range"
            ].any()
        )

        summary_rows.append({
            "HSC_ID":
                hsc_id,
            "galaxy":
                gid,
            "logMstar_obs":
                logMstar_obs,
            "bin_index":
                ibin,
            "a_inner_kpc":
                a_inner[ibin],
            "a_mid_kpc":
                a_mid[ibin],
            "a_outer_kpc":
                a_outer[ibin],
            "xerr_kpc":
                xerr[ibin],
            "mu_HSC":
                mu_hsc[ibin],
            "mu_HSC_err":
                mu_hsc_err[ibin],
            "reliable_radius_kpc":
                Rreliable,
            "HSC_bin_within_reliable_radius":
                bool(
                    hsc_reliable[
                        ibin
                    ]
                ),
            "HSC_finite_and_reliable":
                bool(
                    hsc_finite_reliable[
                        ibin
                    ]
                ),
            "mu_TNG_r_median":
                med,
            "mu_TNG_r_p16":
                p16,
            "mu_TNG_r_p84":
                p84,
            "N_TNG_finite":
                Nfinite,
            "N_TNG_expected":
                15,
            "outside_TNG_master_mass_range":
                outside_mass_range,
        })


# ============================================================
# SAVE NUMERICAL RESULTS
# ============================================================

long_df = pd.DataFrame(
    long_rows
)

geometry_df = pd.DataFrame(
    geometry_rows
)

summary_df = pd.DataFrame(
    summary_rows
)


long_df.to_csv(
    OUT_LONG,
    index=False
)

geometry_df.to_csv(
    OUT_GEOMETRY,
    index=False
)

summary_df.to_csv(
    OUT_VALUES,
    index=False
)


# ============================================================
# VALIDATION
# ============================================================

expected_realization_rows = (
    sum(
        len(
            pd.read_csv(
                BASE /
                f"galaxy{int(g)}_profile_kpc.csv"
            )
        )
        for g in hsc["galaxy"]
    )
    * 15
)

if (
    len(long_df)
    != expected_realization_rows
):
    raise RuntimeError(
        "Unexpected number of individual "
        "TNG profile rows."
    )


counts_geometry = (
    geometry_df
    .groupby("HSC_ID")
    .size()
)

if not np.all(
    counts_geometry.values == 15
):
    raise RuntimeError(
        "Not every HSC galaxy has "
        "15 ellipse geometries."
    )


print()
print("=" * 78)
print("NUMERICAL EXTRACTION COMPLETE")
print("=" * 78)
print(
    "Individual realization rows:",
    len(long_df)
)
print(
    "Ellipse geometry rows:",
    len(geometry_df)
)
print(
    "Summary rows:",
    len(summary_df)
)


# ============================================================
# FINAL ALL-11 FIGURE
# ============================================================

plt.rcParams.update({
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 10,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "savefig.dpi": 220,
})


fig, axes = plt.subplots(
    3,
    4,
    figsize=(14.5, 10.2),
    sharey=True
)

axes = axes.ravel()


for iax, (_, hrow) in enumerate(
    hsc.iterrows()
):

    ax = axes[iax]

    gid = int(
        hrow["galaxy"]
    )

    hsc_id = f"HSC-{gid}"

    d = summary_df[
        summary_df["HSC_ID"]
        == hsc_id
    ].sort_values(
        "bin_index"
    )

    Rrel = float(
        hrow["reliable_radius_kpc"]
    )

    mtng = (
        d["HSC_bin_within_reliable_radius"]
        & np.isfinite(
            d["mu_TNG_r_median"]
        )
    )

    if np.any(mtng):

        xt = d.loc[
            mtng,
            "a_mid_kpc"
        ].to_numpy()

        med = d.loc[
            mtng,
            "mu_TNG_r_median"
        ].to_numpy()

        p16 = d.loc[
            mtng,
            "mu_TNG_r_p16"
        ].to_numpy()

        p84 = d.loc[
            mtng,
            "mu_TNG_r_p84"
        ].to_numpy()

        ax.fill_between(
            xt,
            p16,
            p84,
            alpha=0.22,
            label=(
                "TNG r: 16–84%"
                if iax == 0
                else None
            )
        )

        ax.plot(
            xt,
            med,
            marker="o",
            markersize=3,
            linewidth=1.5,
            label=(
                "TNG r: median (15)"
                if iax == 0
                else None
            )
        )

    mhsc = d[
        "HSC_finite_and_reliable"
    ].to_numpy(dtype=bool)

    if np.any(mhsc):

        xh = d.loc[
            mhsc,
            "a_mid_kpc"
        ].to_numpy()

        yh = d.loc[
            mhsc,
            "mu_HSC"
        ].to_numpy()

        xerrh = d.loc[
            mhsc,
            "xerr_kpc"
        ].to_numpy()

        yerrh = d.loc[
            mhsc,
            "mu_HSC_err"
        ].to_numpy()

        ax.errorbar(
            xh,
            yh,
            xerr=xerrh,
            yerr=yerrh,
            fmt="s",
            markersize=4,
            capsize=2,
            linewidth=1,
            label=(
                "HSC r"
                if iax == 0
                else None
            )
        )


    # Reliable-radius marker
    ax.axvline(
        Rrel,
        linestyle="--",
        linewidth=1.0,
        alpha=0.7
    )


    title = hsc_id

    outside = bool(
        d[
            "outside_TNG_master_mass_range"
        ].iloc[0]
    )

    if outside:
        title += " *"

        ax.text(
            0.03,
            0.05,
            "nearest available TNG;\noutside mass coverage",
            transform=ax.transAxes,
            fontsize=7,
            va="bottom"
        )


    ax.set_title(
        title
    )

    ax.set_xlim(
        left=0,
        right=max(
            1.08 * Rrel,
            1.0
        )
    )

    ax.set_ylim(
        33.5,
        18.0
    )

    ax.grid(
        alpha=0.20
    )


# Hide unused 12th panel.
for j in range(
    len(hsc),
    len(axes)
):
    axes[j].axis("off")


# Axis labels
for i, ax in enumerate(
    axes[:len(hsc)]
):

    row = i // 4
    col = i % 4

    if row == 2:
        ax.set_xlabel(
            "Semi-major axis [kpc]"
        )

    if col == 0:
        ax.set_ylabel(
            r"$\mu_r$ [mag arcsec$^{-2}$]"
        )


# One common legend.
handles, labels = axes[0].get_legend_handles_labels()

fig.legend(
    handles,
    labels,
    loc="upper center",
    ncol=3,
    frameon=True,
    bbox_to_anchor=(0.5, 0.995)
)


fig.suptitle(
    "HSC vs stellar-mass-selected TNG50 analogues: corrected r-band surface brightness",
    y=1.02,
    fontsize=13
)

fig.tight_layout(
    rect=[0, 0, 1, 0.96]
)


fig.savefig(
    OUT_PNG,
    bbox_inches="tight"
)

fig.savefig(
    OUT_PDF,
    bbox_inches="tight"
)

plt.close(fig)


print()
print("=" * 78)
print("FINAL OUTPUTS")
print("=" * 78)

print("Saved:")
print(OUT_PNG)
print(OUT_PDF)
print(OUT_VALUES)
print(OUT_LONG)
print(OUT_GEOMETRY)

print()
print(
    "* HSC-332 and HSC-75 are marked because "
    "their stellar masses fall below the existing "
    "991-galaxy TNG catalogue coverage."
)

print()
print("DONE.")
