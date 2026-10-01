from pathlib import Path

import numpy as np
import pandas as pd
import h5py

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ============================================================
# PATHS / CONFIG
# ============================================================

BASE = Path("/home/tnguser")
COMP = BASE / "HSC_TNG_comparison"

QFILE = (
    COMP /
    "hsc_q_moment_bootstrap_profiles_all11_master.csv"
)

MATCHFILE = (
    COMP /
    "all11_selected_5_TNG_analogues.csv"
)

MAPDIR = (
    COMP /
    "all11_rband_maps"
)

PLOTDIR = (
    COMP /
    "final_plots"
)

PLOTDIR.mkdir(
    parents=True,
    exist_ok=True
)


OUT_INDIVIDUAL = (
    COMP /
    "all11_HSC_TNG_light_q_DM_individual_realizations.csv"
)

OUT_VALUES = (
    PLOTDIR /
    "all11_HSC_TNG_light_q_DM_comparison_values.csv"
)

OUT_PNG = (
    PLOTDIR /
    "all11_HSC_TNG_light_q_DM_comparison.png"
)

OUT_PDF = (
    PLOTDIR /
    "all11_HSC_TNG_light_q_DM_comparison.pdf"
)


PROJECTIONS = [
    "xy",
    "xz",
    "yz"
]

EXPECTED_R_BAND_INDEX = 5

# Number of shape iterations for TNG stellar-light annuli.
N_ITER = 6


# ============================================================
# HELPERS
# ============================================================

def wrap_pa_180(pa_deg):
    
    pa = float(pa_deg)

    while pa >= 90.0:
        pa -= 180.0

    while pa < -90.0:
        pa += 180.0

    return pa


def align_pa_to_reference(pa_deg, reference_deg):

    candidates = np.array([
        pa_deg - 180.0,
        pa_deg,
        pa_deg + 180.0
    ])

    return float(
        candidates[
            np.argmin(
                np.abs(
                    candidates - reference_deg
                )
            )
        ]
    )


def rotate_coords(X, Y, pa_deg):

    pa = np.deg2rad(pa_deg)

    c = np.cos(pa)
    s = np.sin(pa)

    Xr = c * X + s * Y
    Yr = -s * X + c * Y

    return Xr, Yr


def moment_q_pa(x, y, w):

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    w = np.asarray(w, dtype=float)

    good = (
        np.isfinite(x)
        & np.isfinite(y)
        & np.isfinite(w)
        & (w > 0)
    )

    x = x[good]
    y = y[good]
    w = w[good]

    if len(w) < 10:
        return np.nan, np.nan

    sw = np.sum(w)

    if not np.isfinite(sw) or sw <= 0:
        return np.nan, np.nan

    # Fixed centre: maps are centred on SubhaloPos.
    mxx = np.sum(w * x * x) / sw
    myy = np.sum(w * y * y) / sw
    mxy = np.sum(w * x * y) / sw

    trace = mxx + myy

    disc = np.sqrt(
        (mxx - myy)**2
        + 4.0 * mxy**2
    )

    lam1 = 0.5 * (trace + disc)
    lam2 = 0.5 * (trace - disc)

    if (
        not np.isfinite(lam1)
        or not np.isfinite(lam2)
        or lam1 <= 0
        or lam2 < 0
    ):
        return np.nan, np.nan

    q = np.sqrt(
        lam2 / lam1
    )

    q = float(
        np.clip(
            q,
            0.02,
            1.0
        )
    )

    pa = 0.5 * np.arctan2(
        2.0 * mxy,
        mxx - myy
    )

    pa_deg = wrap_pa_180(
        np.rad2deg(pa)
    )

    return q, pa_deg


def light_q_profile_in_hsc_bins(
    SigmaL,
    x_centers,
    y_centers,
    a_inner,
    a_outer,
    n_iter=N_ITER
):

    X, Y = np.meshgrid(
        np.asarray(x_centers, dtype=float),
        np.asarray(y_centers, dtype=float),
        indexing="ij"
    )

    W = np.asarray(
        SigmaL,
        dtype=float
    )

    positive = (
        np.isfinite(W)
        & (W > 0)
    )

    q_out = np.full(
        len(a_inner),
        np.nan
    )

    pa_out = np.full(
        len(a_inner),
        np.nan
    )

    npix_out = np.zeros(
        len(a_inner),
        dtype=int
    )

    # Initial geometry.
    q_current = 1.0
    pa_current = 0.0


    for i, (ain, aout) in enumerate(
        zip(
            a_inner,
            a_outer
        )
    ):

        if (
            not np.isfinite(ain)
            or not np.isfinite(aout)
            or aout <= ain
        ):
            continue

        q_iter = q_current
        pa_iter = pa_current

        best_q = np.nan
        best_pa = np.nan
        best_npix = 0


        for _ in range(n_iter):

            Xr, Yr = rotate_coords(
                X,
                Y,
                pa_iter
            )

            ell_r = np.sqrt(
                Xr**2
                + (
                    Yr / max(
                        q_iter,
                        0.05
                    )
                )**2
            )

            m = (
                positive
                & (ell_r >= ain)
                & (ell_r < aout)
            )

            n = int(
                np.count_nonzero(m)
            )

            if n < 10:
                break

            q_new, pa_new = moment_q_pa(
                X[m],
                Y[m],
                W[m]
            )

            if (
                not np.isfinite(q_new)
                or not np.isfinite(pa_new)
            ):
                break

            pa_new_aligned = (
                align_pa_to_reference(
                    pa_new,
                    pa_iter
                )
            )

            best_q = q_new
            best_pa = pa_new_aligned
            best_npix = n

            # Same type of damped iterative update
            # used in the HSC measurement.
            q_iter = (
                0.6 * q_iter
                + 0.4 * q_new
            )

            pa_iter = (
                0.6 * pa_iter
                + 0.4 * pa_new_aligned
            )


        if np.isfinite(best_q):

            q_out[i] = best_q
            pa_out[i] = wrap_pa_180(
                best_pa
            )

            npix_out[i] = best_npix

            # Start next radial annulus from this solution.
            q_current = best_q
            pa_current = best_pa


    return (
        q_out,
        pa_out,
        npix_out
    )


def interp_no_extrapolation(
    x_new,
    x_old,
    y_old
):
    """
    Linear interpolation onto x_new, but no extrapolation.
    """

    x_new = np.asarray(
        x_new,
        dtype=float
    )

    x_old = np.asarray(
        x_old,
        dtype=float
    )

    y_old = np.asarray(
        y_old,
        dtype=float
    )

    good = (
        np.isfinite(x_old)
        & np.isfinite(y_old)
    )

    x = x_old[good]
    y = y_old[good]

    out = np.full(
        len(x_new),
        np.nan
    )

    if len(x) < 2:
        return out

    order = np.argsort(x)

    x = x[order]
    y = y[order]

    inside = (
        np.isfinite(x_new)
        & (x_new >= np.min(x))
        & (x_new <= np.max(x))
    )

    out[inside] = np.interp(
        x_new[inside],
        x,
        y
    )

    return out


def finite_stats(values):

    v = np.asarray(
        values,
        dtype=float
    )

    v = v[
        np.isfinite(v)
    ]

    if len(v) == 0:
        return (
            np.nan,
            np.nan,
            np.nan,
            0
        )

    return (
        float(np.nanmedian(v)),
        float(np.nanpercentile(v, 16)),
        float(np.nanpercentile(v, 84)),
        int(len(v))
    )


def original_stars_h5(
    row,
    sid
):

    if (
        "output_path_abs" in row.index
        and pd.notna(
            row["output_path_abs"]
        )
    ):
        outdir = Path(
            str(
                row["output_path_abs"]
            )
        )

    else:

        outdir = Path(
            str(
                row["output_path"]
            )
        )

        if not outdir.is_absolute():
            outdir = BASE / outdir

    return (
        outdir /
        f"stars_sub{sid}.hdf5"
    )


# ============================================================
# LOAD INPUTS
# ============================================================

if not QFILE.exists():
    raise FileNotFoundError(QFILE)

if not MATCHFILE.exists():
    raise FileNotFoundError(MATCHFILE)


hsc = pd.read_csv(
    QFILE
)

matches = pd.read_csv(
    MATCHFILE
)

hsc["galaxy"] = (
    hsc["galaxy"]
    .astype(str)
)

matches["HSC_ID"] = (
    matches["HSC_ID"]
    .astype(str)
)


galaxy_order = [
    "454",
    "284",
    "22",
    "250",
    "356",
    "332",
    "201",
    "140",
    "145",
    "75",
    "70"
]


print("=" * 78)
print("ALL-11 HSC vs TNG LIGHT-q vs DM-q COMPARISON")
print("=" * 78)

print(
    "HSC rows:",
    len(hsc)
)

print(
    "HSC galaxies:",
    hsc["galaxy"].nunique()
)

print(
    "Analogue assignments:",
    len(matches)
)

print(
    "Unique TNG galaxies:",
    matches["SubfindID"].nunique()
)

print(
    "TNG stellar band: r"
)

print(
    "TNG photometric index:",
    EXPECTED_R_BAND_INDEX
)

print(
    "Projections:",
    PROJECTIONS
)


# ============================================================
# INDIVIDUAL 15-REALIZATION MEASUREMENTS
# ============================================================

individual_rows = []


for galaxy in galaxy_order:

    h = (
        hsc[
            hsc["galaxy"] == galaxy
        ]
        .copy()
        .sort_values("a_mid_kpc")
    )

    if len(h) == 0:
        raise RuntimeError(
            f"No HSC q profile for {galaxy}"
        )


    hsc_id = f"HSC-{galaxy}"

    msel = (
        matches[
            matches["HSC_ID"] == hsc_id
        ]
        .copy()
        .sort_values("match_rank")
    )

    if len(msel) != 5:
        raise RuntimeError(
            f"{hsc_id}: expected 5 TNG matches, "
            f"found {len(msel)}"
        )


    a_inner = h[
        "a_inner_kpc"
    ].to_numpy(dtype=float)

    a_mid = h[
        "a_mid_kpc"
    ].to_numpy(dtype=float)

    a_outer = h[
        "a_outer_kpc"
    ].to_numpy(dtype=float)


    print()
    print("-" * 78)
    print(
        hsc_id,
        "| bins =",
        len(h),
        "| TNG realizations = 15"
    )
    print("-" * 78)


    for _, mrow in msel.iterrows():

        sid = int(
            mrow["SubfindID"]
        )

        rank = int(
            mrow["match_rank"]
        )

        rmap = (
            MAPDIR /
            f"all11_rband_sub{sid}.hdf5"
        )

        if not rmap.exists():
            raise FileNotFoundError(
                rmap
            )


        old_stars = original_stars_h5(
            mrow,
            sid
        )

        if not old_stars.exists():
            raise FileNotFoundError(
                old_stars
            )


        with (
            h5py.File(
                rmap,
                "r"
            ) as fr,
            h5py.File(
                old_stars,
                "r"
            ) as fo
        ):

            phot_idx = int(
                fr["meta"].attrs.get(
                    "photometric_index",
                    -999
                )
            )

            if phot_idx != 5:
                raise RuntimeError(
                    f"{rmap}: photometric index "
                    f"{phot_idx}, expected 5."
                )


            for proj in PROJECTIONS:

                print(
                    f"  {hsc_id} | rank {rank} | "
                    f"sub {sid} | {proj}"
                )


                # --------------------------------------------
                # Corrected r-band stellar-light q
                # --------------------------------------------

                gr = fr[proj]

                x = gr[
                    "x_centers_kpc"
                ][:]

                y = gr[
                    "y_centers_kpc"
                ][:]

                SigmaL = gr[
                    "SigmaL_Lsun_kpc2"
                ][:]


                (
                    q_light,
                    pa_light,
                    npix_light
                ) = light_q_profile_in_hsc_bins(
                    SigmaL=SigmaL,
                    x_centers=x,
                    y_centers=y,
                    a_inner=a_inner,
                    a_outer=a_outer,
                    n_iter=N_ITER
                )


                # --------------------------------------------
                # Existing projected DM q
                # --------------------------------------------

                if (
                    "derived"
                    not in fo[proj]
                ):
                    raise KeyError(
                        f"{old_stars}: "
                        f"{proj}/derived missing"
                    )

                d = fo[
                    proj
                ][
                    "derived"
                ]

                r_dm = d[
                    "rmid_kpc"
                ][:]

                q_dm_old = d[
                    "q_dm"
                ][:]


                q_dm = (
                    interp_no_extrapolation(
                        x_new=a_mid,
                        x_old=r_dm,
                        y_old=q_dm_old
                    )
                )


                for i in range(
                    len(h)
                ):

                    individual_rows.append({
                        "HSC_ID":
                            hsc_id,
                        "galaxy":
                            galaxy,
                        "SubfindID":
                            sid,
                        "match_rank":
                            rank,
                        "projection":
                            proj,
                        "bin_index":
                            i,
                        "a_inner_kpc":
                            a_inner[i],
                        "a_mid_kpc":
                            a_mid[i],
                        "a_outer_kpc":
                            a_outer[i],
                        "q_TNG_r_light":
                            q_light[i],
                        "PA_TNG_r_light_deg":
                            pa_light[i],
                        "N_TNG_light_pixels":
                            int(
                                npix_light[i]
                            ),
                        "q_TNG_DM":
                            q_dm[i],
                    })


individual = pd.DataFrame(
    individual_rows
)

individual.to_csv(
    OUT_INDIVIDUAL,
    index=False
)


# ============================================================
# AGGREGATE 15 REALIZATIONS
# ============================================================

summary_rows = []


for galaxy in galaxy_order:

    h = (
        hsc[
            hsc["galaxy"] == galaxy
        ]
        .copy()
        .sort_values("a_mid_kpc")
        .reset_index(drop=True)
    )

    hsc_id = f"HSC-{galaxy}"

    msel = (
        matches[
            matches["HSC_ID"] == hsc_id
        ]
    )

    outside_mass_range = bool(
        msel[
            "outside_TNG_master_mass_range"
        ].any()
    )


    for i, hrow in h.iterrows():

        d = individual[
            (individual["HSC_ID"] == hsc_id)
            & (individual["bin_index"] == i)
        ]


        (
            ql_med,
            ql_p16,
            ql_p84,
            Nlight
        ) = finite_stats(
            d[
                "q_TNG_r_light"
            ]
        )


        (
            qd_med,
            qd_p16,
            qd_p84,
            Ndm
        ) = finite_stats(
            d[
                "q_TNG_DM"
            ]
        )


        summary_rows.append({

            "HSC_ID":
                hsc_id,

            "galaxy":
                galaxy,

            "bin_index":
                i,

            "a_inner_kpc":
                float(
                    hrow["a_inner_kpc"]
                ),

            "a_mid_kpc":
                float(
                    hrow["a_mid_kpc"]
                ),

            "a_outer_kpc":
                float(
                    hrow["a_outer_kpc"]
                ),

            "q_HSC":
                float(
                    hrow["q_HSC"]
                )
                if np.isfinite(
                    hrow["q_HSC"]
                )
                else np.nan,

            "q_HSC_err_bootstrap":
                float(
                    hrow["q_err_bootstrap"]
                )
                if np.isfinite(
                    hrow["q_err_bootstrap"]
                )
                else np.nan,

            "q_HSC_p16_bootstrap":
                float(
                    hrow["q_p16_bootstrap"]
                )
                if np.isfinite(
                    hrow["q_p16_bootstrap"]
                )
                else np.nan,

            "q_HSC_p84_bootstrap":
                float(
                    hrow["q_p84_bootstrap"]
                )
                if np.isfinite(
                    hrow["q_p84_bootstrap"]
                )
                else np.nan,

            "HSC_quality_flag":
                str(
                    hrow["quality_flag"]
                ),

            "reliable_radius_kpc":
                float(
                    hrow[
                        "reliable_radius_kpc"
                    ]
                ),

            "q_TNG_r_light_median":
                ql_med,

            "q_TNG_r_light_p16":
                ql_p16,

            "q_TNG_r_light_p84":
                ql_p84,

            "N_TNG_r_light":
                Nlight,

            "q_TNG_DM_median":
                qd_med,

            "q_TNG_DM_p16":
                qd_p16,

            "q_TNG_DM_p84":
                qd_p84,

            "N_TNG_DM":
                Ndm,

            "N_TNG_expected":
                15,

            "outside_TNG_master_mass_range":
                outside_mass_range,
        })


summary = pd.DataFrame(
    summary_rows
)

summary.to_csv(
    OUT_VALUES,
    index=False
)


# ============================================================
# VALIDATION
# ============================================================

print()
print("=" * 78)
print("VALIDATION")
print("=" * 78)

print(
    "Individual rows:",
    len(individual)
)

print(
    "Summary rows:",
    len(summary)
)


for hsc_id in summary[
    "HSC_ID"
].unique():

    d = summary[
        summary["HSC_ID"]
        == hsc_id
    ]

    print(
        hsc_id,
        "| bins:",
        len(d),
        "| light N range:",
        int(
            d[
                "N_TNG_r_light"
            ].min()
        ),
        "-",
        int(
            d[
                "N_TNG_r_light"
            ].max()
        ),
        "| DM N range:",
        int(
            d[
                "N_TNG_DM"
            ].min()
        ),
        "-",
        int(
            d[
                "N_TNG_DM"
            ].max()
        )
    )


# Physical range check

for col in [
    "q_HSC",
    "q_TNG_r_light_median",
    "q_TNG_DM_median"
]:

    finite = (
        summary[col]
        .to_numpy(dtype=float)
    )

    finite = finite[
        np.isfinite(finite)
    ]

    if len(finite):

        if (
            np.nanmin(finite) < 0
            or np.nanmax(finite) > 1.000001
        ):
            raise RuntimeError(
                f"{col} outside physical range 0-1."
            )


# ============================================================
# FINAL 3x4 FIGURE
# ============================================================

plt.rcParams.update({
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 10,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "savefig.dpi": 250,
})


fig, axes = plt.subplots(
    3,
    4,
    figsize=(14.8, 10.2),
    sharey=True
)

axes = axes.ravel()


for iax, galaxy in enumerate(
    galaxy_order
):

    ax = axes[iax]

    hsc_id = f"HSC-{galaxy}"

    d = (
        summary[
            summary["HSC_ID"]
            == hsc_id
        ]
        .sort_values(
            "a_mid_kpc"
        )
    )


    x = d[
        "a_mid_kpc"
    ].to_numpy(dtype=float)


    # --------------------------------------------
    # TNG DM
    # --------------------------------------------

    dm_ok = (
        np.isfinite(
            d[
                "q_TNG_DM_median"
            ]
        )
    )

    if np.any(dm_ok):

        xd = x[dm_ok]

        dm_med = d.loc[
            dm_ok,
            "q_TNG_DM_median"
        ].to_numpy(dtype=float)

        dm16 = d.loc[
            dm_ok,
            "q_TNG_DM_p16"
        ].to_numpy(dtype=float)

        dm84 = d.loc[
            dm_ok,
            "q_TNG_DM_p84"
        ].to_numpy(dtype=float)

        ax.fill_between(
            xd,
            dm16,
            dm84,
            alpha=0.16
        )

        ax.plot(
            xd,
            dm_med,
            linestyle="--",
            linewidth=1.5,
            label=(
                "TNG DM"
                if iax == 0
                else None
            )
        )


    # --------------------------------------------
    # TNG r-band stellar light
    # --------------------------------------------

    st_ok = (
        np.isfinite(
            d[
                "q_TNG_r_light_median"
            ]
        )
    )

    if np.any(st_ok):

        xs = x[st_ok]

        st_med = d.loc[
            st_ok,
            "q_TNG_r_light_median"
        ].to_numpy(dtype=float)

        st16 = d.loc[
            st_ok,
            "q_TNG_r_light_p16"
        ].to_numpy(dtype=float)

        st84 = d.loc[
            st_ok,
            "q_TNG_r_light_p84"
        ].to_numpy(dtype=float)

        ax.fill_between(
            xs,
            st16,
            st84,
            alpha=0.20
        )

        ax.plot(
            xs,
            st_med,
            linewidth=1.6,
            label=(
                "TNG r-band light"
                if iax == 0
                else None
            )
        )


    # --------------------------------------------
    # HSC q with bootstrap uncertainties
    # --------------------------------------------

    hgood = (
        np.isfinite(
            d["q_HSC"]
        )
        & (
            d[
                "HSC_quality_flag"
            ]
            == "good"
        )
    )

    if np.any(hgood):

        xh = d.loc[
            hgood,
            "a_mid_kpc"
        ].to_numpy(dtype=float)

        yh = d.loc[
            hgood,
            "q_HSC"
        ].to_numpy(dtype=float)

        y16 = d.loc[
            hgood,
            "q_HSC_p16_bootstrap"
        ].to_numpy(dtype=float)

        y84 = d.loc[
            hgood,
            "q_HSC_p84_bootstrap"
        ].to_numpy(dtype=float)

        yerr = np.vstack([
            yh - y16,
            y84 - yh
        ])

        yerr[
            ~np.isfinite(yerr)
        ] = 0

        yerr[
            yerr < 0
        ] = 0


        ax.errorbar(
            xh,
            yh,
            yerr=yerr,
            fmt="o",
            markersize=4,
            linewidth=1.0,
            capsize=2,
            label=(
                "HSC"
                if iax == 0
                else None
            )
        )


    Rrel = float(
        d[
            "reliable_radius_kpc"
        ].iloc[0]
    )


    ax.axvline(
        Rrel,
        linestyle=":",
        linewidth=1.0,
        alpha=0.75
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
            0.04,
            "nearest available TNG",
            transform=ax.transAxes,
            fontsize=7,
            va="bottom"
        )


    ax.set_title(
        title
    )

    ax.set_ylim(
        0.0,
        1.05
    )

    ax.set_xlim(
        left=0,
        right=max(
            1.04 * Rrel,
            1.0
        )
    )

    ax.grid(
        alpha=0.18
    )


# Hide 12th empty panel.
for j in range(
    len(galaxy_order),
    len(axes)
):
    axes[j].axis("off")


for i, ax in enumerate(
    axes[:len(galaxy_order)]
):

    row = i // 4
    col = i % 4

    if row == 2:
        ax.set_xlabel(
            "Semi-major axis [kpc]"
        )

    if col == 0:
        ax.set_ylabel(
            r"$q=b/a$"
        )


# Common legend
handles, labels = (
    axes[0]
    .get_legend_handles_labels()
)

fig.legend(
    handles,
    labels,
    loc="upper center",
    ncol=3,
    bbox_to_anchor=(
        0.5,
        0.995
    )
)


fig.suptitle(
    "Observed HSC stellar-light shape vs stellar-mass-selected TNG50 analogues",
    fontsize=13,
    y=1.02
)


fig.tight_layout(
    rect=[
        0,
        0,
        1,
        0.96
    ]
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

print(OUT_PNG)
print(OUT_PDF)
print(OUT_VALUES)
print(OUT_INDIVIDUAL)

print()
print(
    "* HSC-332 and HSC-75 are outside the stellar-mass "
    "coverage of the processed TNG catalogue."
)

print()
print("DONE.")
