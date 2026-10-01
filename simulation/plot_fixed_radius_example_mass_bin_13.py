import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "lines.markersize": 6,
    "savefig.dpi": 220,
    "savefig.bbox": "tight",
})

data_root = "mass_bin_13"
metrics_csv = os.path.join(data_root, "fixed_radius_metrics_logM13_ALL.csv")

outdir = os.path.join(data_root, "fixed_radius_metric_plots_logM13_ALL")
os.makedirs(outdir, exist_ok=True)

df = pd.read_csv(metrics_csv)

projs = ["xy", "xz", "yz"]


def scatter_q_vs_logM(radius_label, suffix, title_extra):
   
    plt.figure(figsize=(6.3, 4.5))

    for proj in projs:

        star_col = f"{proj}_q_stars_{suffix}"
        dm_col = f"{proj}_q_dm_{suffix}"

        if star_col in df:
            plt.scatter(
                df["logM200c"],
                df[star_col],
                marker="o",
                alpha=0.75,
                label=f"stars {proj}"
            )

        if dm_col in df:
            plt.scatter(
                df["logM200c"],
                df[dm_col],
                marker="s",
                alpha=0.75,
                label=f"DM {proj}"
            )

    plt.xlabel(r"$\log_{10}(M_{200c}/M_\odot)$")
    plt.ylabel(r"projected $q=b/a$")
    plt.ylim(0.0, 1.03)
    plt.grid(alpha=0.25)
    plt.title(title_extra)
    plt.legend(ncol=2, frameon=True, framealpha=0.85)

    outpath = os.path.join(
        outdir,
        f"q_vs_logM_{radius_label}.png"
    )
    plt.savefig(outpath)
    plt.close()
    print("Saved", outpath)


def scatter_mu30_radius():
    plt.figure(figsize=(6.3, 4.5))

    for proj in projs:
        col = f"{proj}_R_mu30_over_R200c"

        if col in df:
            plt.scatter(
                df["logM200c"],
                df[col],
                alpha=0.75,
                label=proj
            )

    plt.xlabel(r"$\log_{10}(M_{200c}/M_\odot)$")
    plt.ylabel(r"$R(\mu_r=30)/R_{200c}$")
    plt.ylim(0.0, 1.0)
    plt.grid(alpha=0.25)
    plt.title(r"$R(\mu_r=30)$ for $\log M_{200c}\sim13.0\pm0.25$")
    plt.legend(frameon=True, framealpha=0.85)

    outpath = os.path.join(outdir, "R_mu30_over_R200c_vs_logM.png")
    plt.savefig(outpath)
    plt.close()
    print("Saved", outpath)


def scatter_q_at_mu30():
    plt.figure(figsize=(6.3, 4.5))

    for proj in projs:

        star_col = f"{proj}_q_stars_at_mu30"
        dm_col = f"{proj}_q_dm_at_mu30"

        if star_col in df:
            plt.scatter(
                df["logM200c"],
                df[star_col],
                marker="o",
                alpha=0.75,
                label=f"stars {proj}"
            )

        if dm_col in df:
            plt.scatter(
                df["logM200c"],
                df[dm_col],
                marker="s",
                alpha=0.75,
                label=f"DM {proj}"
            )

    plt.xlabel(r"$\log_{10}(M_{200c}/M_\odot)$")
    plt.ylabel(r"projected $q=b/a$ at $\mu_r=30$")
    plt.ylim(0.0, 1.03)
    plt.grid(alpha=0.25)
    plt.title(r"Shape at stellar-halo depth $\mu_r=30$")
    plt.legend(ncol=2, frameon=True, framealpha=0.85)

    outpath = os.path.join(outdir, "q_at_mu30_vs_logM.png")
    plt.savefig(outpath)
    plt.close()
    print("Saved", outpath)


scatter_q_vs_logM(
    radius_label="0p1R200c",
    suffix="0p1R200c",
    title_extra=r"Projected shape at $R=0.1R_{200c}$"
)

scatter_q_vs_logM(
    radius_label="50kpc",
    suffix="50kpc",
    title_extra=r"Projected shape at $R=50$ kpc"
)

scatter_mu30_radius()

scatter_q_at_mu30()

print("")
print("Done. Outputs saved in:")
print(outdir)
