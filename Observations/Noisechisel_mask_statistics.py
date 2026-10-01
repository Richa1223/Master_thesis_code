import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from astropy.io import fits

PIXEL_SCALE = 0.168

GALAXIES = {
    "454": {"x0": 1717.88, "y0": 1762.808, "keep_arcsec": 59.72},
    "284": {"x0": 1739.48, "y0": 1768.28, "keep_arcsec": 42.484},
    "22":  {"x0": 1764.6051, "y0": 1809.3949, "keep_arcsec": 80.484},
    "250": {"x0": 1742.0, "y0": 1818.0, "keep_arcsec": 51.728},
    "356": {"x0": 1800.0, "y0": 1752.0, "keep_arcsec": 35.381},
    "332": {"x0": 1799.0, "y0": 1783.0, "keep_arcsec": 39.42},
    "201": {"x0": 1735.3072, "y0": 1768.1904, "keep_arcsec": 60.81},
    "140": {"x0": 1803.5888, "y0": 1843.9872, "keep_arcsec": 48.398},
    "145": {"x0": 1751.12, "y0": 1785.44, "keep_arcsec": 49.808},
    "75":  {"x0": 1784.5117, "y0": 1719.8154, "keep_arcsec": 83.31},
    "70":  {"x0": 1786.0323, "y0": 1861.2717, "keep_arcsec": 117.612},
}

OUTDIR = Path.home() / "Downloads/for_observations/My_selected_galaxies/noisechisel_mask_statistics"
OUTDIR.mkdir(parents=True, exist_ok=True)

rows = []

for g, cfg in GALAXIES.items():
    nc_path = Path(f"galaxy{g}/nc.fits")
    if not nc_path.exists():
        print(f"Missing {nc_path}")
        continue

    with fits.open(nc_path) as hdul:
        img = np.array(hdul[1].data, dtype=float)   # INPUT-NO-SKY
        det = np.array(hdul[2].data)                # DETECTIONS

    x0 = cfg["x0"]
    y0 = cfg["y0"]
    keep_arcsec = cfg["keep_arcsec"]

    yy, xx = np.indices(img.shape)
    r_arcsec = np.sqrt((xx - x0)**2 + (yy - y0)**2) * PIXEL_SCALE

    finite = np.isfinite(img)
    detections = det != 0

    # NoiseChisel detections anywhere in the image
    noisechisel_detected = detections & finite

    # Final contaminant mask used for profile measurement:
    # mask detected sources outside the preserved central galaxy region
    final_contaminant_mask = detections & (r_arcsec > keep_arcsec) & finite

    n_finite = finite.sum()
    n_detected = noisechisel_detected.sum()
    n_final_mask = final_contaminant_mask.sum()

    rows.append({
        "galaxy": g,
        "finite_pixels": int(n_finite),
        "noisechisel_detected_pixels": int(n_detected),
        "final_contaminant_mask_pixels": int(n_final_mask),
        "noisechisel_detected_fraction_percent": 100.0 * n_detected / n_finite,
        "final_contaminant_mask_fraction_percent": 100.0 * n_final_mask / n_finite,
        "unmasked_fraction_percent": 100.0 * (n_finite - n_final_mask) / n_finite,
    })

df = pd.DataFrame(rows)
df["galaxy_int"] = df["galaxy"].astype(int)
df = df.sort_values("galaxy_int")

csv_path = OUTDIR / "all11_noisechisel_mask_statistics.csv"
df.to_csv(csv_path, index=False)

plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
})

# 1) Best slide figure: bar chart per galaxy
fig, ax = plt.subplots(figsize=(8.2, 4.6))

x = np.arange(len(df))
ax.bar(x, df["final_contaminant_mask_fraction_percent"])

ax.set_xticks(x)
ax.set_xticklabels([f"HSC-{g}" for g in df["galaxy"]], rotation=45, ha="right")
ax.set_ylabel("Masked pixels [%]")
ax.set_xlabel("Galaxy")
ax.set_title("Final contaminant-mask fraction")
ax.grid(axis="y", alpha=0.3)

fig.tight_layout()
fig.savefig(OUTDIR / "all11_final_mask_fraction_bar.png", dpi=250, bbox_inches="tight")
fig.savefig(OUTDIR / "all11_final_mask_fraction_bar.pdf", bbox_inches="tight")
plt.close(fig)

# 2) Actual histogram of mask fractions
fig, ax = plt.subplots(figsize=(6.2, 4.4))

ax.hist(df["final_contaminant_mask_fraction_percent"], bins=6, edgecolor="black")

ax.set_xlabel("Masked pixels [%]")
ax.set_ylabel("Number of galaxies")
ax.set_title("Distribution of final mask fractions")
ax.grid(axis="y", alpha=0.3)

fig.tight_layout()
fig.savefig(OUTDIR / "all11_final_mask_fraction_histogram.png", dpi=250, bbox_inches="tight")
fig.savefig(OUTDIR / "all11_final_mask_fraction_histogram.pdf", bbox_inches="tight")
plt.close(fig)

print()
print("Saved outputs in:")
print(OUTDIR)
print()
print(df[[
    "galaxy",
    "noisechisel_detected_fraction_percent",
    "final_contaminant_mask_fraction_percent",
    "unmasked_fraction_percent"
]].to_string(index=False))
