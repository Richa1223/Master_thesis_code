import os
import numpy as np
import pandas as pd
import h5py
import matplotlib.pyplot as plt
import warnings

# ============================================================
# PLOT STYLE
# ============================================================

plt.rcParams.update({
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "lines.linewidth": 1.4,
    "lines.markersize": 4,
    "savefig.dpi": 220,
    "savefig.bbox": "tight",
})

# ============================================================
# CONFIG
# ============================================================

sample_csv = "TNG_work/stack_sample_logM13_pm0p25_ALL.csv"

outdir = "mass_bin_13/stack_outputs_logM13_ALL_3D"
os.makedirs(outdir, exist_ok=True)

# Common normalized radius grid.
# The true profile bin centres do not include exactly 0 or exactly 1,
# so using approximately the full range.
x_common = np.linspace(0.02, 0.98, 24)

# ============================================================
# READ FUNCTIONS
# ============================================================

def read_shape_profile(h5file, groupname):

    with h5py.File(h5file, "r") as f:
        g = f[groupname]

        r = g["rmid_kpc"][:]
        ba = g["b_over_a"][:]
        ca = g["c_over_a"][:]

    return r, ba, ca


def interpolate_profile(x, y, x_common):
  
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    good = np.isfinite(x) & np.isfinite(y)

    if np.count_nonzero(good) < 2:
        return np.full_like(x_common, np.nan, dtype=float)

    xg = x[good]
    yg = y[good]

    order = np.argsort(xg)
    xg = xg[order]
    yg = yg[order]

    # Remove duplicate x values if any.
    xu, idx = np.unique(xg, return_index=True)
    yu = yg[idx]

    if len(xu) < 2:
        return np.full_like(x_common, np.nan, dtype=float)

    yout = np.full_like(x_common, np.nan, dtype=float)

    inside = (x_common >= np.nanmin(xu)) & (x_common <= np.nanmax(xu))

    yout[inside] = np.interp(
        x_common[inside],
        xu,
        yu
    )

    return yout


# ============================================================
# STACKING PLOT
# ============================================================

def make_stack_plot(
    sample_df,
    groupname,
    quantity,
    outname,
    title_extra=""
):

    if quantity == "b_over_a":
        ylabel = r"$b/a$"
        qname = "b/a"
    elif quantity == "c_over_a":
        ylabel = r"$c/a$"
        qname = "c/a"
    else:
        raise ValueError("quantity must be 'b_over_a' or 'c_over_a'")

    all_interp = []
    used_rows = []

    plt.figure(figsize=(6.3, 4.6))

    for _, row in sample_df.iterrows():

        subhaloID = int(row["hydro_subhaloID"])
        R200c = float(row["R200c_kpc"])

        h5file = f"mass_bin_13/outputs_{subhaloID}/dm_sub{subhaloID}.hdf5"

        if not os.path.exists(h5file):
            print(f"Missing HDF5, skipping: {h5file}")
            continue

        try:
            r_kpc, ba, ca = read_shape_profile(h5file, groupname)
        except Exception as e:
            print(f"Could not read {groupname} from {h5file}: {repr(e)}")
            continue

        x = r_kpc / R200c

        if quantity == "b_over_a":
            y = ba
        else:
            y = ca

        # Individual halo line: light and transparent.
        plt.plot(
            x,
            y,
            color="tab:blue",
            alpha=0.25,
            lw=1.0
        )

        y_interp = interpolate_profile(x, y, x_common)
        all_interp.append(y_interp)

        used_rows.append({
            "hydro_subhaloID": subhaloID,
            "R200c_kpc": R200c,
            "logM200c": float(row["logM200c"]),
        })

    if len(all_interp) == 0:
        raise RuntimeError("No valid profiles were found. Nothing to stack.")

    arr = np.array(all_interp, dtype=float)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)

        y_med = np.nanmedian(arr, axis=0)
        y16 = np.nanpercentile(arr, 16, axis=0)
        y84 = np.nanpercentile(arr, 84, axis=0)

    n_valid = np.sum(np.isfinite(arr), axis=0)

    # Spread region
    plt.fill_between(
        x_common,
        y16,
        y84,
        color="black",
        alpha=0.15,
        label="16-84 percentile"
    )

    plt.plot(
        x_common,
        y_med,
        color="black",
        lw=2.2,
        marker="s",
        ms=4.5,
        label="median"
    )

    plt.xlabel(r"$r_{\rm 3D}/R_{200c}$")
    plt.ylabel(ylabel)
    plt.xlim(0.0, 1.0)
    plt.ylim(0.35, 1.03)
    plt.grid(alpha=0.25)

    n_halos = len(all_interp)

    if groupname == "shape3d":
        prof_label = "cumulative 3D DM shape"
    elif groupname == "shape3d_shells":
        prof_label = "shell 3D DM shape"
    else:
        prof_label = groupname

    plt.title(
        rf"$\log M_{{200c}}\sim 13.0\pm0.25$ | {prof_label} | {qname}"
    )

    plt.legend(loc="best", frameon=True, framealpha=0.85)

    outpath = os.path.join(outdir, outname)
    plt.savefig(outpath, dpi=220, bbox_inches="tight")
    plt.close()

    # Save the stacked numerical profile too.
    stats = pd.DataFrame({
        "r_over_R200c": x_common,
        "median": y_med,
        "p16": y16,
        "p84": y84,
        "N_valid": n_valid,
    })

    stats_csv = outpath.replace(".png", "_stats.csv")
    stats.to_csv(stats_csv, index=False)

    used_csv = outpath.replace(".png", "_halos_used.csv")
    pd.DataFrame(used_rows).to_csv(used_csv, index=False)

    print("")
    print(f"Saved plot: {outpath}")
    print(f"Saved stats: {stats_csv}")
    print(f"Halos used: {n_halos}")
    print(f"Saved halo list: {used_csv}")


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    sample = pd.read_csv(sample_csv)

    print("")
    print("Loaded sample:")
    print(sample_csv)
    print(sample[[
        "hydro_subhaloID",
        "logM200c",
        "R200c_kpc",
        "GroupID",
        "GroupNsubs"
    ]])

    # 1) Cumulative 3D DM b/a
    make_stack_plot(
        sample_df=sample,
        groupname="shape3d",
        quantity="b_over_a",
        outname="stack_logM13_shape3d_b_over_a.png"
    )

    # 2) Cumulative 3D DM c/a
    make_stack_plot(
        sample_df=sample,
        groupname="shape3d",
        quantity="c_over_a",
        outname="stack_logM13_shape3d_c_over_a.png"
    )

    # 3) Shell 3D DM b/a
    make_stack_plot(
        sample_df=sample,
        groupname="shape3d_shells",
        quantity="b_over_a",
        outname="stack_logM13_shape3d_shells_b_over_a.png"
    )

    # 4) Shell 3D DM c/a
    make_stack_plot(
        sample_df=sample,
        groupname="shape3d_shells",
        quantity="c_over_a",
        outname="stack_logM13_shape3d_shells_c_over_a.png"
    )

    print("")
    print("Done. Stacked plots are in:")
    print(outdir)
