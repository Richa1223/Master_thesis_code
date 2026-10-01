import os
import numpy as np
import pandas as pd
import h5py
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
    {
        "label": "11.5",
        "root": "mass_bin_11p5_spinproj",
        "tag": "logM11p5",
        "sample_csv": "TNG_work/stack_sample_logM11p5_pm0p25_ALL.csv",
    },
    {
        "label": "12.0",
        "root": "mass_bin_12_spinproj",
        "tag": "logM12",
        "sample_csv": "TNG_work/stack_sample_logM12_pm0p25_ALL.csv",
    },
    {
        "label": "12.5",
        "root": "mass_bin_12p5_spinproj",
        "tag": "logM12p5",
        "sample_csv": "TNG_work/stack_sample_logM12p5_pm0p25_ALL.csv",
    },
    {
        "label": "13.0",
        "root": "mass_bin_13_spinproj",
        "tag": "logM13",
        "sample_csv": "TNG_work/stack_sample_logM13_pm0p25_ALL.csv",
    },
]

projs = ["xy", "xz", "yz", "faceon", "edgeon"]

outdir = "TNG_work/final_paired_qdiff_projected"
os.makedirs(outdir, exist_ok=True)


def collect_1d_datasets(group):
    data = {}

    def visitor(name, obj):
        if isinstance(obj, h5py.Dataset):
            try:
                arr = obj[()]
            except Exception:
                return

            if not isinstance(arr, np.ndarray):
                return

            if arr.ndim != 1:
                return

            if len(arr) < 5:
                return

            if not np.issubdtype(arr.dtype, np.number):
                return

            data[name] = np.asarray(arr, dtype=float)

    group.visititems(visitor)
    return data


def choose_q_dataset(datasets, kind, source_name):
    
    scored = []

    for name, arr in datasets.items():
        low = name.lower()
        base = low.split("/")[-1]

        finite = np.isfinite(arr)
        if finite.sum() < 5:
            continue

        med = np.nanmedian(arr)

        if not (-0.1 <= med <= 1.2):
            continue

        score = 0

        if "q" in base:
            score += 10
        if "axis" in low or "ba" in low or "b_over_a" in low:
            score += 4

        if kind == "stars":
            if "star" in low or "stellar" in low:
                score += 20
            if "dm" in low or "dark" in low:
                score -= 30

        if kind == "dm":
            if "dm" in low or "dark" in low:
                score += 20
            if "star" in low or "stellar" in low:
                score -= 30
            # In the DM hdf5 file the dataset may simply be called q
            if source_name == "dm_h5" and "q" in base:
                score += 8

        exact_good = [
            "q",
            "q_star",
            "q_stars",
            "qstellar",
            "q_dm",
            "qdark",
            "q_dark",
        ]

        if base in exact_good:
            score += 15

        scored.append((score, name, arr))

    scored = sorted(scored, key=lambda x: x[0], reverse=True)

    if len(scored) == 0 or scored[0][0] <= 0:
        raise KeyError(
            f"Could not identify q dataset for kind={kind}, source={source_name}. "
            f"Available 1D datasets = {list(datasets.keys())}"
        )

    return scored[0][1], scored[0][2]


def choose_radius_dataset(datasets, R200c_kpc, fallback_R=None, needed_len=None):
    scored = []

    for name, arr in datasets.items():
        low = name.lower()
        base = low.split("/")[-1]

        if needed_len is not None and len(arr) != needed_len:
            continue

        finite = np.isfinite(arr)
        if finite.sum() < 5:
            continue

        test = arr[finite]

        # radius should be mostly increasing and positive
        if np.nanmedian(test) <= 0:
            continue

        score = 0

        if "r_over_r200" in low or "r_r200" in low or "roverr200" in low:
            score += 30
        if "r200" in low:
            score += 20
        if "radius" in low:
            score += 10
        if "r_mid" in low or "rmid" in low:
            score += 10
        if base.startswith("r"):
            score += 5
        if "kpc" in low:
            score += 5

        if "q" in low or "mu" in low or "sigma" in low or "mass" in low:
            score -= 20

        scored.append((score, name, arr))

    scored = sorted(scored, key=lambda x: x[0], reverse=True)

    if len(scored) > 0 and scored[0][0] > 0:
        name, arr = scored[0][1], scored[0][2]
        low = name.lower()

        R = np.asarray(arr, dtype=float)

        # If it is in kpc, convert to R/R200c
        if "kpc" in low or np.nanmax(R) > 3:
            R = R / R200c_kpc

        return name, R

    if fallback_R is not None:
        R = np.asarray(fallback_R, dtype=float)

        if needed_len is not None and len(R) == needed_len:
            return "fallback_from_stack_stats", R

    raise KeyError(
        f"Could not identify radius dataset. Available 1D datasets = {list(datasets.keys())}"
    )


def get_fallback_radius(root, tag, proj):
    fname = os.path.join(
        root,
        f"stack_outputs_{tag}_projected_{proj}",
        f"stack_{tag}_{proj}_q_stars_stats.csv"
    )

    if not os.path.exists(fname):
        return None

    df = pd.read_csv(fname)

    for c in ["R_over_R200c", "R_R200c", "Rmid_over_R200c", "R_mid_over_R200c"]:
        if c in df.columns:
            return df[c].to_numpy(dtype=float)

    for c in df.columns:
        low = c.lower()
        if "r" in low and "200" in low:
            return df[c].to_numpy(dtype=float)

    return None


def read_q_from_h5(h5_path, proj, kind, R200c_kpc, fallback_R=None):
    source_name = "dm_h5" if "dm_sub" in os.path.basename(h5_path) else "stars_h5"

    with h5py.File(h5_path, "r") as f:
        if proj not in f:
            raise KeyError(f"Projection {proj} not found in {h5_path}. Groups = {list(f.keys())}")

        g = f[proj]

        if "derived" in g:
            gread = g["derived"]
        else:
            gread = g

        datasets = collect_1d_datasets(gread)

        q_name, q = choose_q_dataset(datasets, kind=kind, source_name=source_name)
        R_name, R = choose_radius_dataset(
            datasets,
            R200c_kpc=R200c_kpc,
            fallback_R=fallback_R,
            needed_len=len(q)
        )

    return R, q, q_name, R_name


def read_q_pair(root, sid, R200c_kpc, proj, fallback_R=None):
    outdir = os.path.join(root, f"outputs_{sid}")

    stars_h5 = os.path.join(outdir, f"stars_sub{sid}.hdf5")
    dm_h5 = os.path.join(outdir, f"dm_sub{sid}.hdf5")

    if not os.path.exists(stars_h5):
        raise FileNotFoundError(stars_h5)

    if not os.path.exists(dm_h5):
        raise FileNotFoundError(dm_h5)

    R_star, q_star, qstar_name, Rstar_name = read_q_from_h5(
        stars_h5,
        proj,
        kind="stars",
        R200c_kpc=R200c_kpc,
        fallback_R=fallback_R,
    )

    try:
        R_dm, q_dm, qdm_name, Rdm_name = read_q_from_h5(
            stars_h5,
            proj,
            kind="dm",
            R200c_kpc=R200c_kpc,
            fallback_R=R_star,
        )
        qdm_source = "stars_h5"
    except Exception:
        R_dm, q_dm, qdm_name, Rdm_name = read_q_from_h5(
            dm_h5,
            proj,
            kind="dm",
            R200c_kpc=R200c_kpc,
            fallback_R=R_star,
        )
        qdm_source = "dm_h5"

    R_star = np.asarray(R_star, dtype=float)
    q_star = np.asarray(q_star, dtype=float)
    R_dm = np.asarray(R_dm, dtype=float)
    q_dm = np.asarray(q_dm, dtype=float)

    ok_s = np.isfinite(R_star) & np.isfinite(q_star)
    ok_d = np.isfinite(R_dm) & np.isfinite(q_dm)

    R_star = R_star[ok_s]
    q_star = q_star[ok_s]

    R_dm = R_dm[ok_d]
    q_dm = q_dm[ok_d]

    order_s = np.argsort(R_star)
    order_d = np.argsort(R_dm)

    R_star = R_star[order_s]
    q_star = q_star[order_s]

    R_dm = R_dm[order_d]
    q_dm = q_dm[order_d]

    # Interpolate DM q onto stellar radius grid if needed
    if len(R_star) != len(R_dm) or not np.allclose(R_star, R_dm, rtol=1e-4, atol=1e-5):
        q_dm_interp = np.interp(R_star, R_dm, q_dm, left=np.nan, right=np.nan)
        q_dm = q_dm_interp
        R = R_star
    else:
        R = R_star

    qdiff = q_star - q_dm

    meta = {
        "qstar_dataset": qstar_name,
        "qdm_dataset": qdm_name,
        "qdm_source": qdm_source,
        "Rstar_dataset": Rstar_name,
        "Rdm_dataset": Rdm_name,
    }

    return R, qdiff, q_star, q_dm, meta


summary_rows = []
failure_rows = []

for proj in projs:

    plt.figure(figsize=(7.2, 5.2))

    for mb in mass_bins:
        label = mb["label"]
        root = mb["root"]
        tag = mb["tag"]
        sample_csv = mb["sample_csv"]

        print("")
        print("=" * 90)
        print(f"Mass bin {label} | projection {proj}")
        print("=" * 90)

        sample = pd.read_csv(sample_csv)
        fallback_R = get_fallback_radius(root, tag, proj)

        R_grid = None
        qdiff_list = []
        qstar_list = []
        qdm_list = []
        halos_used = 0

        for _, row in sample.iterrows():
            sid = int(row["hydro_subhaloID"])
            R200c_kpc = float(row["R200c_kpc"])

            try:
                R, qdiff, qstar, qdm, meta = read_q_pair(
                    root=root,
                    sid=sid,
                    R200c_kpc=R200c_kpc,
                    proj=proj,
                    fallback_R=fallback_R,
                )

                if R_grid is None:
                    R_grid = R

                # Interpolate every halo onto first successful radius grid
                if len(R) != len(R_grid) or not np.allclose(R, R_grid, rtol=1e-4, atol=1e-5):
                    qdiff = np.interp(R_grid, R, qdiff, left=np.nan, right=np.nan)
                    qstar = np.interp(R_grid, R, qstar, left=np.nan, right=np.nan)
                    qdm = np.interp(R_grid, R, qdm, left=np.nan, right=np.nan)

                qdiff_list.append(qdiff)
                qstar_list.append(qstar)
                qdm_list.append(qdm)
                halos_used += 1

            except Exception as e:
                failure_rows.append({
                    "mass_bin": label,
                    "projection": proj,
                    "hydro_subhaloID": sid,
                    "error": repr(e),
                })

        if halos_used == 0:
            print(f"No halos usable for mass bin {label}, proj {proj}")
            continue

        qdiff_arr = np.asarray(qdiff_list, dtype=float)
        qstar_arr = np.asarray(qstar_list, dtype=float)
        qdm_arr = np.asarray(qdm_list, dtype=float)

        med = np.nanmedian(qdiff_arr, axis=0)
        p16 = np.nanpercentile(qdiff_arr, 16, axis=0)
        p84 = np.nanpercentile(qdiff_arr, 84, axis=0)

        med_star = np.nanmedian(qstar_arr, axis=0)
        med_dm = np.nanmedian(qdm_arr, axis=0)

        out_stats = pd.DataFrame({
            "R_over_R200c": R_grid,
            "median_qstar_minus_qdm_paired": med,
            "p16_qstar_minus_qdm_paired": p16,
            "p84_qstar_minus_qdm_paired": p84,
            "median_qstar": med_star,
            "median_qdm": med_dm,
            "median_qstar_minus_median_qdm": med_star - med_dm,
            "N_halos_used": halos_used,
        })

        stats_path = os.path.join(
            outdir,
            f"paired_qdiff_stats_{tag}_{proj}.csv"
        )
        out_stats.to_csv(stats_path, index=False)

        print(f"Halos used = {halos_used} / {len(sample)}")
        print("Saved:", stats_path)

        plt.plot(
            R_grid,
            med,
            marker="o",
            label=fr'$\log M_{{200c}}\sim {label}$, N={halos_used}'
        )

        plt.fill_between(
            R_grid,
            p16,
            p84,
            alpha=0.12
        )

        # convergence-ish numbers
        for threshold in [0.05, 0.10, 0.20]:
            good = np.isfinite(R_grid) & np.isfinite(med) & (np.abs(med) <= threshold)
            if np.any(good):
                R_first = float(R_grid[good][0])
            else:
                R_first = np.nan

            summary_rows.append({
                "mass_bin": label,
                "projection": proj,
                "N_halos_used": halos_used,
                "threshold_abs_qstar_minus_qdm": threshold,
                "R_first_over_R200c_paired_median": R_first,
            })

    plt.axhline(0.0, color="black", ls=":", lw=1.4)
    plt.axhline(-0.05, color="gray", ls=":", lw=1.0)
    plt.axhline(0.05, color="gray", ls=":", lw=1.0)

    plt.xlabel(r"$R/R_{200c}$")
    plt.ylabel(r"median per-halo $(q_\star - q_{\rm DM})$")
    plt.xlim(0, 1)
    plt.grid(alpha=0.25)
    plt.title(f"Paired projected shape difference | {proj}")
    plt.legend(frameon=True)

    fig_path = os.path.join(outdir, f"paired_qdiff_massbins_{proj}.png")
    plt.savefig(fig_path)
    plt.close()

    print("")
    print("Saved figure:", fig_path)


summary = pd.DataFrame(summary_rows)
summary_path = os.path.join(outdir, "paired_qdiff_convergence_summary.csv")
summary.to_csv(summary_path, index=False)
print("")
print("Saved:", summary_path)

failures = pd.DataFrame(failure_rows)
fail_path = os.path.join(outdir, "paired_qdiff_failures.csv")
failures.to_csv(fail_path, index=False)

print("Saved:", fail_path)
print("")
print("Failures:", len(failures))
if len(failures) > 0:
    print(failures.head(20))

print("")
print("DONE paired qdiff analysis.")
print("Outputs in:")
print(outdir)
