import csv
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

profile_root = Path("final_profiles_all12")
out_root = Path("final_profiles_all12_kpc")
out_root.mkdir(exist_ok=True)

galaxies = ["454","284","22","250","356","332","201","140","145","75","70"]

scale = {}
independent = {}

with open("final12_kpc_scale.csv", newline="") as f:
    reader = csv.DictReader(f)
    for row in reader:
        g = row["galaxy"].strip()
        scale[g] = float(row["kpc_per_arcsec"])
        independent[g] = row.get("independent_sample", "yes").strip()

sbl = {}

with open("final_observation_sample_sbl_table.csv", newline="") as f:
    reader = csv.DictReader(f)
    for row in reader:
        sbl[row["galaxy"].strip()] = float(row["empirical_5x5_mu_all"])

for g in galaxies:
    infile = profile_root / f"galaxy{g}" / f"galaxy{g}_elliptical_profile.csv"

    if not infile.exists():
        print("Missing:", infile)
        continue

    data = np.genfromtxt(infile, delimiter=",", names=True)

    a_arcsec = data["a_mid_arcsec"]
    a_kpc = a_arcsec * scale[g]

    mu_mean = data["mu_clipped_mean"]
    mu_mean_err = data["mu_clipped_mean_err"]
    mu_median = data["mu_clipped_median"]
    unmasked = data["unmasked_fraction"]

    good_mean = np.isfinite(mu_mean) & (unmasked > 0.5)
    good_median = np.isfinite(mu_median) & (unmasked > 0.5)

    outdir = out_root / f"galaxy{g}"
    outdir.mkdir(exist_ok=True)

    out_csv = outdir / f"galaxy{g}_elliptical_profile_kpc.csv"

    with open(out_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "a_mid_arcsec",
            "a_mid_kpc",
            "mu_clipped_mean",
            "mu_clipped_mean_err",
            "mu_clipped_median",
            "unmasked_fraction"
        ])

        for aa, ak, mm, me, md, uf in zip(
            a_arcsec,
            a_kpc,
            mu_mean,
            mu_mean_err,
            mu_median,
            unmasked
        ):
            writer.writerow([aa, ak, mm, me, md, uf])

    fig, ax = plt.subplots(figsize=(7, 5))

    ax.errorbar(
        a_kpc[good_mean],
        mu_mean[good_mean],
        yerr=mu_mean_err[good_mean],
        fmt="o-",
        markersize=4,
        linewidth=1.5,
        label="Sigma-clipped mean"
    )

    ax.plot(
        a_kpc[good_median],
        mu_median[good_median],
        "s--",
        markersize=4,
        linewidth=1.5,
        label="Sigma-clipped median"
    )

    ax.axhline(
        sbl[g],
        linestyle=":",
        linewidth=2,
        label=f"Empirical 5x5 SBL = {sbl[g]:.2f}"
    )

    title = f"Galaxy {g} profile"
    if independent.get(g) == "no":
        title += " duplicate"

    ax.set_xlabel("Semi-major axis [kpc]")
    ax.set_ylabel("Surface brightness [mag arcsec$^{-2}$]")
    ax.set_title(title)
    ax.invert_yaxis()
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9)

    fig.tight_layout()
    fig.savefig(outdir / f"galaxy{g}_elliptical_profile_kpc.png", dpi=200)
    fig.savefig(outdir / f"galaxy{g}_elliptical_profile_kpc.pdf")
    plt.close(fig)

    print("Saved kpc profile for galaxy", g)

print("DONE.")
