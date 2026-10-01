import os
import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 8,
    "lines.linewidth": 1.6,
    "lines.markersize": 4,
    "savefig.dpi": 220,
    "savefig.bbox": "tight",
})

data_root = "mass_bin_11p5_spinproj"
outdir = os.path.join(data_root, "stack_outputs_logM11p5_projected_compare")
os.makedirs(outdir, exist_ok=True)

projs = ["xy", "xz", "yz", "faceon", "edgeon"]

def read_stats(proj, component):
    folder = os.path.join(data_root, f"stack_outputs_logM11p5_projected_{proj}")

    if component == "stars":
        fname = os.path.join(folder, f"stack_logM11p5_{proj}_q_stars_stats.csv")
    elif component == "dm":
        fname = os.path.join(folder, f"stack_logM11p5_{proj}_q_dm_stats.csv")
    else:
        raise ValueError(component)

    if not os.path.exists(fname):
        raise FileNotFoundError(fname)

    return pd.read_csv(fname)

plt.figure(figsize=(7.0, 5.0))

for proj in projs:
    s = read_stats(proj, "stars")
    d = read_stats(proj, "dm")

    plt.plot(s["R_over_R200c"], s["median"], marker="o", ls="-", alpha=0.85, label=f"stars {proj}")
    plt.plot(d["R_over_R200c"], d["median"], marker="s", ls="--", alpha=0.85, label=f"DM {proj}")

plt.xlabel(r"$R/R_{200c}$")
plt.ylabel(r"projected $q=b/a$")
plt.xlim(0, 1)
plt.ylim(0.0, 1.03)
plt.grid(alpha=0.25)
plt.title(r"$\log M_{200c}=11.5\pm0.25$ | median projected shape")
plt.legend(ncol=2, frameon=True, framealpha=0.85)

outpath = os.path.join(outdir, "compare_logM11p5_projected_q_stars_vs_dm_allprojs.png")
plt.savefig(outpath)
plt.close()
print("Saved", outpath)

plt.figure(figsize=(7.0, 5.0))

for proj in projs:
    s = read_stats(proj, "stars")
    d = read_stats(proj, "dm")

    diff = s["median"] - d["median"]
    plt.plot(s["R_over_R200c"], diff, marker="o", lw=1.7, label=proj)

plt.axhline(0.0, color="black", lw=1.2, ls=":")
plt.axhline(-0.05, color="gray", lw=1.0, ls=":")
plt.axhline(0.05, color="gray", lw=1.0, ls=":")

plt.xlabel(r"$R/R_{200c}$")
plt.ylabel(r"$q_\star - q_{\rm DM}$")
plt.xlim(0, 1)
plt.grid(alpha=0.25)
plt.title(r"$\log M_{200c}=11.5\pm0.25$ | projected shape difference")
plt.legend(frameon=True, framealpha=0.85)

outpath = os.path.join(outdir, "compare_logM11p5_projected_q_difference_stars_minus_dm.png")
plt.savefig(outpath)
plt.close()
print("Saved", outpath)

print("")
print("Done. Outputs saved in:")
print(outdir)
