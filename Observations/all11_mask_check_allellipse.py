import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from pathlib import Path
from astropy.io import fits

PIXEL_SCALE = 0.168

GALAXIES = {
    "454": {"x0": 1717.88, "y0": 1762.808, "q": 0.6004433009638437, "pa": -153.30508144079147, "keep_arcsec": 59.72},
    "284": {"x0": 1739.48, "y0": 1768.28, "q": 0.9280534717546512, "pa": -155.48636442051355, "keep_arcsec": 42.484},
    "22":  {"x0": 1764.6051, "y0": 1809.3949, "q": 0.6760150422993393, "pa": 131.6879754672273, "keep_arcsec": 80.484},
    "250": {"x0": 1742.0, "y0": 1818.0, "q": 0.5565354668663934, "pa": -155.02527913507214, "keep_arcsec": 51.728},
    "356": {"x0": 1800.0, "y0": 1752.0, "q": 0.7472184352586136, "pa": 133.79188191628373, "keep_arcsec": 35.381},
    "332": {"x0": 1799.0, "y0": 1783.0, "q": 0.693921452627094, "pa": -161.49160146282725, "keep_arcsec": 39.42},
    "201": {"x0": 1735.3072, "y0": 1768.1904, "q": 0.7042658464767191, "pa": 57.748171417136355, "keep_arcsec": 60.81},
    "140": {"x0": 1803.5888, "y0": 1843.9872, "q": 0.7746248948302538, "pa": 131.7080564178205, "keep_arcsec": 48.398},
    "145": {"x0": 1751.12, "y0": 1785.44, "q": 0.9009872306077155, "pa": 118.68456250646598, "keep_arcsec": 49.808},
    "75":  {"x0": 1784.5117, "y0": 1719.8154, "q": 0.6293138334389051, "pa": 118.92393032100944, "keep_arcsec": 83.31},
    "70":  {"x0": 1786.0323, "y0": 1861.2717, "q": 0.7902830540136395, "pa": -148.73472138952835, "keep_arcsec": 117.612},
}

OUTDIR = Path.home() / "Downloads/for_observations/My_selected_galaxies/all11_mask_check_allellipse"
OUTDIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
})

def find_profile_csv(g):
    candidates = [
        Path(f"final_profiles_all12/galaxy{g}/galaxy{g}_elliptical_profile.csv"),
        Path(f"final_profiles_all12/galaxy{g}/elliptical_profile.csv"),
        Path(f"final_profiles_all12_kpc/galaxy{g}/galaxy{g}_elliptical_profile_kpc.csv"),
        Path(f"final_profiles_all12_kpc/galaxy{g}/galaxy{g}_elliptical_profile.csv"),
    ]
    for p in candidates:
        if p.exists():
            return p

    hits = sorted(Path(".").glob(f"**/*galaxy{g}*elliptical*profile*.csv"))
    hits = [h for h in hits if "profile_test" not in str(h)]
    if not hits:
        raise FileNotFoundError(f"No profile CSV found for galaxy {g}")
    return hits[0]

def get_radii_arcsec(profile_path):
    df = pd.read_csv(profile_path)

    for col in ["a_outer_arcsec", "a_mid_arcsec", "a_arcsec"]:
        if col in df.columns:
            r = np.asarray(df[col], dtype=float)
            r = r[np.isfinite(r)]
            r = np.unique(r)
            return r

    if "a_mid_kpc" in df.columns:
        raise RuntimeError(
            f"{profile_path} only has kpc radii. Use the original arcsec profile CSV instead."
        )

    raise RuntimeError(f"No arcsec radius column found in {profile_path}. Columns: {list(df.columns)}")

def make_one(g, cfg):
    nc_path = Path(f"galaxy{g}/nc.fits")
    if not nc_path.exists():
        raise FileNotFoundError(f"Missing {nc_path}")

    profile_path = find_profile_csv(g)
    radii_arcsec = get_radii_arcsec(profile_path)
    radii_pix = radii_arcsec / PIXEL_SCALE

    x0 = cfg["x0"]
    y0 = cfg["y0"]
    q = cfg["q"]
    pa = cfg["pa"]
    keep_arcsec = cfg["keep_arcsec"]

    with fits.open(nc_path) as hdul:
        img = np.array(hdul[1].data, dtype=float)   # INPUT-NO-SKY
        det = np.array(hdul[2].data)                # DETECTIONS

    yy, xx = np.indices(img.shape)
    r_circ_arcsec = np.sqrt((xx - x0)**2 + (yy - y0)**2) * PIXEL_SCALE

    detmask = det != 0
    final_mask = (~np.isfinite(img)) | (detmask & (r_circ_arcsec > keep_arcsec))

    half_size_pix = 750
    ny, nx = img.shape
    x1 = max(0, int(round(x0 - half_size_pix)))
    x2 = min(nx, int(round(x0 + half_size_pix)))
    y1 = max(0, int(round(y0 - half_size_pix)))
    y2 = min(ny, int(round(y0 + half_size_pix)))

    img_c = img[y1:y2, x1:x2]
    mask_c = final_mask[y1:y2, x1:x2]

    vals = img_c[np.isfinite(img_c)]
    vmin, vmax = np.nanpercentile(vals, [1, 99.5])

    fig, axes = plt.subplots(1, 2, figsize=(12.0, 5.2), constrained_layout=True)

    axes[0].imshow(
        img_c,
        origin="lower",
        cmap="gray",
        vmin=vmin,
        vmax=vmax,
        extent=[x1, x2, y1, y2],
    )

    axes[1].imshow(
        mask_c.astype(float),
        origin="lower",
        cmap="gray",
        extent=[x1, x2, y1, y2],
    )

    for ax in axes:
        for rp in radii_pix:
            ell = Ellipse(
                (x0, y0),
                width=2.0 * rp,
                height=2.0 * rp * q,
                angle=pa,
                fill=False,
                edgecolor="red",
                linewidth=0.75,
                alpha=0.85,
            )
            ax.add_patch(ell)

        ax.plot(x0, y0, marker="+", color="deepskyblue", markersize=11, markeredgewidth=2)
        ax.set_xlabel("x [pixel]")
        ax.set_ylabel("y [pixel]")

    axes[0].set_title(f"HSC-{g}: INPUT-NO-SKY + all annuli", pad=8)
    axes[1].set_title("Final contaminant mask + all annuli", pad=8)

    out_png = OUTDIR / f"galaxy{g}_mask_check_allellipse.png"
    out_pdf = OUTDIR / f"galaxy{g}_mask_check_allellipse.pdf"

    fig.savefig(out_png, dpi=250, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved {out_png}")
    print(f"Saved {out_pdf}")
    print(f"  profile radii: {profile_path}")
    print(f"  number of ellipses: {len(radii_pix)}")

for g in ["22", "70", "75", "140", "145", "201", "250", "284", "332", "356", "454"]:
    try:
        make_one(g, GALAXIES[g])
    except Exception as e:
        print(f"FAILED galaxy {g}: {e}")

print()
print("DONE.")
print("Output folder:")
print(OUTDIR)
