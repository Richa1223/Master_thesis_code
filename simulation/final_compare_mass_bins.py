import os
import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.size": 10,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 9,
    "lines.linewidth": 2.0,
    "lines.markersize": 4,
    "savefig.dpi": 220,
    "savefig.bbox": "tight",
})

mass_bins = [
    {"label": "11.5", "root": "mass_bin_11p5_spinproj", "tag": "logM11p5"},
    {"label": "12.0", "root": "mass_bin_12_spinproj",   "tag": "logM12"},
    {"label": "12.5", "root": "mass_bin_12p5_spinproj", "tag": "logM12p5"},
    {"label": "13.0", "root": "mass_bin_13_spinproj",   "tag": "logM13"},
]

projs = ["xy", "xz", "yz", "faceon", "edgeon"]

outdir = "TNG_work/final_massbin_comparisons_projected"
os.makedirs(outdir, exist_ok=True)


def read_stats(root, tag, proj, kind):
    folder = os.path.join(root, f"stack_outputs_{tag}_projected_{proj}")

    if kind == "q_stars":
        fname = os.path.join(folder, f"stack_{tag}_{proj}_q_stars_stats.csv")
    elif kind == "q_dm":
        fname = os.path.join(folder, f"stack_{tag}_{proj}_q_dm_stats.csv")
    elif kind == "mu_r":
        fname = os.path.join(folder, f"stack_{tag}_{proj}_mu_r_stats.csv")
    elif kind == "logSigmaDM":
        fname = os.path.join(folder, f"stack_{tag}_{proj}_logSigmaDM_stats.csv")
    else:
        raise ValueError(kind)

    if not os.path.exists(fname):
        raise FileNotFoundError(fname)

    df = pd.read_csv(fname)
    return df, fname


def get_radius_col(df, fname):
    options = [
        "R_over_R200c",
        "R_R200c",
        "Rmid_over_R200c",
        "R_mid_over_R200c",
        "radius_over_R200c",
        "RoverR200c",
    ]

    for c in options:
        if c in df.columns:
            return c

    # fallback: first column containing both R and 200
    for c in df.columns:
        cl = c.lower()
        if "r" in cl and "200" in cl:
            return c

    raise KeyError(f"Could not find radius column in {fname}. Columns = {list(df.columns)}")


def get_median_col(df, fname):
    options = [
        "median",
        "p50",
        "q50",
        "perc50",
        "percentile50",
        "50",
        "median_value",
        "median_profile",
        "mu_median",
        "median_mu",
        "median_mu_r",
        "logSigmaDM_median",
        "median_logSigmaDM",
    ]

    for c in options:
        if c in df.columns:
            return c

    # fallback: any column containing median
    for c in df.columns:
        if "median" in c.lower():
            return c

    # fallback: any column containing p50
    for c in df.columns:
        if "p50" in c.lower():
            return c

    raise KeyError(f"Could not find median column in {fname}. Columns = {list(df.columns)}")


def get_xy(root, tag, proj, kind):
    df, fname = read_stats(root, tag, proj, kind)
    rcol = get_radius_col(df, fname)
    ycol = get_median_col(df, fname)
    return df[rcol], df[ycol]


for proj in projs:

    # --------------------------------------------------
    # 1) stellar q
    # --------------------------------------------------
    plt.figure(figsize=(7, 5))

    for mb in mass_bins:
        R, y = get_xy(mb["root"], mb["tag"], proj, "q_stars")
        plt.plot(
            R,
            y,
            marker="o",
            label=fr'$\log M_{{200c}}\sim {mb["label"]}$'
        )

    plt.xlabel(r"$R/R_{200c}$")
    plt.ylabel(r"projected stellar $q=b/a$")
    plt.xlim(0, 1)
    plt.ylim(0, 1.03)
    plt.grid(alpha=0.25)
    plt.title(f"Projected stellar shape | {proj}")
    plt.legend(frameon=True)
    plt.savefig(os.path.join(outdir, f"compare_massbins_{proj}_qstars.png"))
    plt.close()

    # --------------------------------------------------
    # 2) DM q
    # --------------------------------------------------
    plt.figure(figsize=(7, 5))

    for mb in mass_bins:
        R, y = get_xy(mb["root"], mb["tag"], proj, "q_dm")
        plt.plot(
            R,
            y,
            marker="s",
            label=fr'$\log M_{{200c}}\sim {mb["label"]}$'
        )

    plt.xlabel(r"$R/R_{200c}$")
    plt.ylabel(r"projected DM $q=b/a$")
    plt.xlim(0, 1)
    plt.ylim(0, 1.03)
    plt.grid(alpha=0.25)
    plt.title(f"Projected DM shape | {proj}")
    plt.legend(frameon=True)
    plt.savefig(os.path.join(outdir, f"compare_massbins_{proj}_qdm.png"))
    plt.close()

    # --------------------------------------------------
    # 3) qstars - qDM
    # --------------------------------------------------
    plt.figure(figsize=(7, 5))

    for mb in mass_bins:
        Rs, ys = get_xy(mb["root"], mb["tag"], proj, "q_stars")
        Rd, yd = get_xy(mb["root"], mb["tag"], proj, "q_dm")

        diff = ys.to_numpy() - yd.to_numpy()

        plt.plot(
            Rs,
            diff,
            marker="o",
            label=fr'$\log M_{{200c}}\sim {mb["label"]}$'
        )

    plt.axhline(0.0, color="black", ls=":")
    plt.axhline(-0.05, color="gray", ls=":", lw=1.0)
    plt.axhline(0.05, color="gray", ls=":", lw=1.0)
    plt.xlabel(r"$R/R_{200c}$")
    plt.ylabel(r"$q_\star - q_{\rm DM}$")
    plt.xlim(0, 1)
    plt.grid(alpha=0.25)
    plt.title(f"Projected shape difference | {proj}")
    plt.legend(frameon=True)
    plt.savefig(os.path.join(outdir, f"compare_massbins_{proj}_qdiff.png"))
    plt.close()

    # --------------------------------------------------
    # 4) mu_r
    # --------------------------------------------------
    plt.figure(figsize=(7, 5))

    for mb in mass_bins:
        R, y = get_xy(mb["root"], mb["tag"], proj, "mu_r")
        plt.plot(
            R,
            y,
            marker="o",
            label=fr'$\log M_{{200c}}\sim {mb["label"]}$'
        )

    plt.xlabel(r"$R/R_{200c}$")
    plt.ylabel(r"$\mu_r$ [mag arcsec$^{-2}$]")
    plt.xlim(0, 1)
    plt.gca().invert_yaxis()
    plt.grid(alpha=0.25)
    plt.title(f"Stellar surface brightness | {proj}")
    plt.legend(frameon=True)
    plt.savefig(os.path.join(outdir, f"compare_massbins_{proj}_mu_r.png"))
    plt.close()

    # --------------------------------------------------
    # 5) logSigmaDM
    # --------------------------------------------------
    plt.figure(figsize=(7, 5))

    for mb in mass_bins:
        R, y = get_xy(mb["root"], mb["tag"], proj, "logSigmaDM")
        plt.plot(
            R,
            y,
            marker="s",
            label=fr'$\log M_{{200c}}\sim {mb["label"]}$'
        )

    plt.xlabel(r"$R/R_{200c}$")
    plt.ylabel(r"$\log_{10}\Sigma_{\rm DM}$ [M$_\odot$/kpc$^2$]")
    plt.xlim(0, 1)
    plt.grid(alpha=0.25)
    plt.title(f"DM surface density | {proj}")
    plt.legend(frameon=True)
    plt.savefig(os.path.join(outdir, f"compare_massbins_{proj}_logSigmaDM.png"))
    plt.close()

print("Done.")
print("Saved outputs in:")
print(outdir)
