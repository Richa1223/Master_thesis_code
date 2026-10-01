from pathlib import Path

import numpy as np
import pandas as pd
import h5py

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ============================================================
# PATHS
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
    "all11_shape_zoom_maps"
)

PLOTDIR = (
    COMP /
    "final_plots"
)

PLOTDIR.mkdir(
    parents=True,
    exist_ok=True
)


OUT_VALUES = (
    PLOTDIR /
    "all11_FINAL_HSC_TNG_r_light_DM_q_values.csv"
)

OUT_INDIVIDUAL = (
    COMP /
    "all11_FINAL_HSC_TNG_r_light_DM_q_individual.csv"
)

OUT_PNG = (
    PLOTDIR /
    "all11_FINAL_HSC_TNG_r_light_DM_q.png"
)

OUT_PDF = (
    PLOTDIR /
    "all11_FINAL_HSC_TNG_r_light_DM_q.pdf"
)


PROJECTIONS = [
    "xy",
    "xz",
    "yz"
]

N_ITER = 6
NMIN_PIXELS = 50


# ============================================================
# HELPERS
# ============================================================

def wrap_pa(pa):

    while pa >= 90:
        pa -= 180

    while pa < -90:
        pa += 180

    return pa


def align_pa(pa, ref):

    cands = np.array([
        pa - 180,
        pa,
        pa + 180
    ])

    return float(
        cands[
            np.argmin(
                np.abs(
                    cands - ref
                )
            )
        ]
    )


def rotate(X, Y, pa_deg):

    p = np.deg2rad(
        pa_deg
    )

    c = np.cos(p)
    s = np.sin(p)

    return (
        c * X + s * Y,
        -s * X + c * Y
    )


def moment_q_pa(x, y, w):

    good = (
        np.isfinite(x)
        & np.isfinite(y)
        & np.isfinite(w)
        & (w > 0)
    )

    x = x[good]
    y = y[good]
    w = w[good]

    if len(w) < NMIN_PIXELS:
        return np.nan, np.nan

    sw = np.sum(w)

    if sw <= 0:
        return np.nan, np.nan

    mxx = np.sum(
        w * x * x
    ) / sw

    myy = np.sum(
        w * y * y
    ) / sw

    mxy = np.sum(
        w * x * y
    ) / sw

    tr = mxx + myy

    disc = np.sqrt(
        (mxx - myy)**2
        + 4 * mxy**2
    )

    l1 = 0.5 * (
        tr + disc
    )

    l2 = 0.5 * (
        tr - disc
    )

    if l1 <= 0 or l2 < 0:
        return np.nan, np.nan

    q = np.sqrt(
        l2 / l1
    )

    q = float(
        np.clip(
            q,
            0.02,
            1.0
        )
    )

    pa = 0.5 * np.arctan2(
        2 * mxy,
        mxx - myy
    )

    pa = wrap_pa(
        np.rad2deg(pa)
    )

    return q, pa


def q_profile(
    weight_map,
    centers,
    a_inner,
    a_outer
):

    X, Y = np.meshgrid(
        centers,
        centers,
        indexing="ij"
    )

    W = np.asarray(
        weight_map,
        dtype=float
    )

    valid = (
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

    q_current = 1.0
    pa_current = 0.0

    for i, (ain, aout) in enumerate(
        zip(
            a_inner,
            a_outer
        )
    ):

        q_iter = q_current
        pa_iter = pa_current

        best_q = np.nan
        best_pa = np.nan
        best_n = 0

        for _ in range(
            N_ITER
        ):

            Xr, Yr = rotate(
                X,
                Y,
                pa_iter
            )

            rell = np.sqrt(
                Xr**2
                + (
                    Yr / max(
                        q_iter,
                        0.05
                    )
                )**2
            )

            m = (
                valid
                & (rell >= ain)
                & (rell < aout)
            )

            n = int(
                np.count_nonzero(m)
            )

            if n < NMIN_PIXELS:
                break

            qnew, panew = moment_q_pa(
                X[m],
                Y[m],
                W[m]
            )

            if not np.isfinite(qnew):
                break

            panew = align_pa(
                panew,
                pa_iter
            )

            best_q = qnew
            best_pa = panew
            best_n = n

            q_iter = (
                0.6 * q_iter
                + 0.4 * qnew
            )

            pa_iter = (
                0.6 * pa_iter
                + 0.4 * panew
            )

        if np.isfinite(
            best_q
        ):

            q_out[i] = best_q

            pa_out[i] = wrap_pa(
                best_pa
            )

            npix_out[i] = best_n

            q_current = best_q
            pa_current = best_pa

    return (
        q_out,
        pa_out,
        npix_out
    )


def stats(x):

    x = np.asarray(
        x,
        dtype=float
    )

    x = x[
        np.isfinite(x)
    ]

    if len(x) == 0:

        return (
            np.nan,
            np.nan,
            np.nan,
            0
        )

    return (
        float(
            np.nanmedian(x)
        ),
        float(
            np.nanpercentile(
                x,
                16
            )
        ),
        float(
            np.nanpercentile(
                x,
                84
            )
        ),
        int(len(x))
    )


# ============================================================
# LOAD INPUT TABLES
# ============================================================

qmaster = pd.read_csv(
    QFILE
)

matches = pd.read_csv(
    MATCHFILE
)

qmaster["galaxy"] = (
    qmaster["galaxy"]
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


# ============================================================
# MEASURE INDIVIDUAL TNG REALIZATIONS
# ============================================================

rows = []


for galaxy in galaxy_order:

    h = (
        qmaster[
            qmaster["galaxy"]
            == galaxy
        ]
        .sort_values(
            "a_mid_kpc"
        )
        .reset_index(
            drop=True
        )
    )

    hsc_id = (
        f"HSC-{galaxy}"
    )

    msel = (
        matches[
            matches["HSC_ID"]
            == hsc_id
        ]
        .sort_values(
            "match_rank"
        )
    )

    if len(msel) != 5:
        raise RuntimeError(
            hsc_id
        )

    ain = h[
        "a_inner_kpc"
    ].to_numpy(
        dtype=float
    )

    amid = h[
        "a_mid_kpc"
    ].to_numpy(
        dtype=float
    )

    aout = h[
        "a_outer_kpc"
    ].to_numpy(
        dtype=float
    )

    print()
    print(
        hsc_id,
        "| bins",
        len(h)
    )

    for _, mrow in msel.iterrows():

        sid = int(
            mrow[
                "SubfindID"
            ]
        )

        rank = int(
            mrow[
                "match_rank"
            ]
        )

        path = (
            MAPDIR /
            f"all11_shapezoom_sub{sid}.hdf5"
        )

        if not path.exists():
            raise FileNotFoundError(
                path
            )

        with h5py.File(
            path,
            "r"
        ) as f:

            if int(
                f["meta"].attrs[
                    "photometric_index"
                ]
            ) != 5:

                raise RuntimeError(
                    "Wrong stellar band."
                )

            for proj in PROJECTIONS:

                g = f[
                    proj
                ]

                centers = g[
                    "x_centers_kpc"
                ][:]

                SigmaL = g[
                    "SigmaL_Lsun_kpc2"
                ][:]

                SigmaDM = g[
                    "SigmaDM_Msun_kpc2"
                ][:]

                (
                    qstar,
                    pastar,
                    npixstar
                ) = q_profile(
                    SigmaL,
                    centers,
                    ain,
                    aout
                )

                (
                    qdm,
                    padm,
                    npixdm
                ) = q_profile(
                    SigmaDM,
                    centers,
                    ain,
                    aout
                )

                for i in range(
                    len(h)
                ):

                    rows.append({
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
                            ain[i],
                        "a_mid_kpc":
                            amid[i],
                        "a_outer_kpc":
                            aout[i],
                        "q_TNG_r_light":
                            qstar[i],
                        "q_TNG_DM":
                            qdm[i],
                        "Npix_TNG_r_light":
                            int(
                                npixstar[i]
                            ),
                        "Npix_TNG_DM":
                            int(
                                npixdm[i]
                            ),
                    })


individual = pd.DataFrame(
    rows
)

individual.to_csv(
    OUT_INDIVIDUAL,
    index=False
)


# ============================================================
# STACK 15 TNG REALIZATIONS
# ============================================================

summary_rows = []


for galaxy in galaxy_order:

    h = (
        qmaster[
            qmaster["galaxy"]
            == galaxy
        ]
        .sort_values(
            "a_mid_kpc"
        )
        .reset_index(
            drop=True
        )
    )

    hsc_id = (
        f"HSC-{galaxy}"
    )

    msel = matches[
        matches["HSC_ID"]
        == hsc_id
    ]

    outside = bool(
        msel[
            "outside_TNG_master_mass_range"
        ].any()
    )

    for i, hrow in h.iterrows():

        d = individual[
            (
                individual["HSC_ID"]
                == hsc_id
            )
            &
            (
                individual["bin_index"]
                == i
            )
        ]

        smed, s16, s84, ns = stats(
            d[
                "q_TNG_r_light"
            ]
        )

        dmed, d16, d84, nd = stats(
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
                hrow[
                    "a_inner_kpc"
                ],
            "a_mid_kpc":
                hrow[
                    "a_mid_kpc"
                ],
            "a_outer_kpc":
                hrow[
                    "a_outer_kpc"
                ],
            "q_HSC":
                hrow[
                    "q_HSC"
                ],
            "q_HSC_p16":
                hrow[
                    "q_p16_bootstrap"
                ],
            "q_HSC_p84":
                hrow[
                    "q_p84_bootstrap"
                ],
            "reliable_radius_kpc":
                hrow[
                    "reliable_radius_kpc"
                ],
            "q_TNG_r_light_median":
                smed,
            "q_TNG_r_light_p16":
                s16,
            "q_TNG_r_light_p84":
                s84,
            "N_TNG_r_light":
                ns,
            "q_TNG_DM_median":
                dmed,
            "q_TNG_DM_p16":
                d16,
            "q_TNG_DM_p84":
                d84,
            "N_TNG_DM":
                nd,
            "N_expected":
                15,
            "outside_TNG_mass_range":
                outside,
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
print("=" * 75)
print("VALIDATION")
print("=" * 75)


for hsc_id in summary[
    "HSC_ID"
].unique():

    d = summary[
        summary["HSC_ID"]
        == hsc_id
    ]

    print(
        hsc_id,
        "| light N:",
        int(
            d["N_TNG_r_light"].min()
        ),
        "-",
        int(
            d["N_TNG_r_light"].max()
        ),
        "| DM N:",
        int(
            d["N_TNG_DM"].min()
        ),
        "-",
        int(
            d["N_TNG_DM"].max()
        )
    )


# ============================================================
# FINAL FIGURE
# ============================================================

# Smaller fonts for presentation use
plt.rcParams.update({
    "font.size": 7,
    "axes.titlesize": 9,
    "axes.labelsize": 9,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 8,
    "figure.titlesize": 11,
})


COLOR_DM = "tab:blue"
COLOR_STAR = "tab:orange"
COLOR_HSC = "black"


fig, axes = plt.subplots(
    3,
    4,
    figsize=(
        14.8,
        10.2
    ),
    sharey=True
)

axes = axes.ravel()


for iax, galaxy in enumerate(
    galaxy_order
):

    ax = axes[
        iax
    ]

    hsc_id = (
        f"HSC-{galaxy}"
    )

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
    ].to_numpy(
        dtype=float
    )


    # ========================================================
    # TNG DM
    # ========================================================

    dmok = np.isfinite(
        d[
            "q_TNG_DM_median"
        ]
    )

    ax.fill_between(
        x[dmok],
        d.loc[
            dmok,
            "q_TNG_DM_p16"
        ],
        d.loc[
            dmok,
            "q_TNG_DM_p84"
        ],
        color=COLOR_DM,
        alpha=0.15
    )

    ax.plot(
        x[dmok],
        d.loc[
            dmok,
            "q_TNG_DM_median"
        ],
        color=COLOR_DM,
        linestyle="--",
        linewidth=1.3,
        label=(
            "TNG DM"
            if iax == 0
            else None
        )
    )


    # ========================================================
    # TNG r-BAND STELLAR LIGHT
    # ========================================================

    sok = np.isfinite(
        d[
            "q_TNG_r_light_median"
        ]
    )

    ax.fill_between(
        x[sok],
        d.loc[
            sok,
            "q_TNG_r_light_p16"
        ],
        d.loc[
            sok,
            "q_TNG_r_light_p84"
        ],
        color=COLOR_STAR,
        alpha=0.20
    )

    ax.plot(
        x[sok],
        d.loc[
            sok,
            "q_TNG_r_light_median"
        ],
        color=COLOR_STAR,
        linewidth=1.3,
        label=(
            "TNG r-band light"
            if iax == 0
            else None
        )
    )


    # ========================================================
    # HSC
    # ========================================================

    qh = d[
        "q_HSC"
    ].to_numpy(
        dtype=float
    )

    q16 = d[
        "q_HSC_p16"
    ].to_numpy(
        dtype=float
    )

    q84 = d[
        "q_HSC_p84"
    ].to_numpy(
        dtype=float
    )

    good = np.isfinite(
        qh
    )

    yerr = np.vstack([
        qh[good]
        - q16[good],
        q84[good]
        - qh[good]
    ])

    yerr[
        ~np.isfinite(yerr)
    ] = 0

    yerr[
        yerr < 0
    ] = 0


    ax.errorbar(
        x[good],
        qh[good],
        yerr=yerr,
        fmt="o",
        color=COLOR_HSC,
        markersize=2.8,
        capsize=1.5,
        linewidth=0.8,
        label=(
            "HSC"
            if iax == 0
            else None
        )
    )


    # ========================================================
    # RELIABLE RADIUS
    # ========================================================

    Rrel = float(
        d[
            "reliable_radius_kpc"
        ].iloc[0]
    )

    ax.axvline(
        Rrel,
        color="0.45",
        linestyle=":",
        linewidth=0.8
    )

    title = (
        hsc_id
    )

    outside = bool(
        d[
            "outside_TNG_mass_range"
        ].iloc[0]
    )

    if outside:
        title += " *"

    ax.set_title(
        title,
        fontsize=9,
        pad=3
    )


    # ========================================================
    # AXES
    # ========================================================

    ax.set_xlim(
        0,
        1.04 * Rrel
    )

    ax.set_ylim(
        0,
        1.05
    )

    ax.tick_params(
        axis="both",
        labelsize=7
    )

    ax.grid(
        alpha=0.15,
        linewidth=0.5
    )

for j in range(
    len(galaxy_order),
    len(axes)
):
    axes[j].axis(
        "off"
    )


for i, ax in enumerate(
    axes[:len(galaxy_order)]
):

    if i // 4 == 2:
        ax.set_xlabel(
            "Semi-major axis [kpc]",
            fontsize=9
        )

    if i % 4 == 0:
        ax.set_ylabel(
            r"$q=b/a$",
            fontsize=9
        )


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
        0.985
    ),
    fontsize=8,
    frameon=False
)


# ============================================================
# MAIN TITLE
# ============================================================

fig.suptitle(
    "Observed HSC stellar-light shape vs stellar-mass-selected TNG50 analogues",
    fontsize=11,
    y=0.997
)


fig.tight_layout(
    rect=[
        0,
        0,
        1,
        0.94
    ]
)


# ============================================================
# SAVE
# ============================================================

fig.savefig(
    OUT_PNG,
    dpi=250,
    bbox_inches="tight"
)

fig.savefig(
    OUT_PDF,
    bbox_inches="tight"
)

plt.close(
    fig
)


print()
print("Saved:")
print(OUT_PNG)
print(OUT_PDF)
print(OUT_VALUES)
print(OUT_INDIVIDUAL)
