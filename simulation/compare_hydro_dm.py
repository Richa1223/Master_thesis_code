import os
import numpy as np
import h5py
import matplotlib.pyplot as plt


plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.titlesize": 14,
    "lines.linewidth": 1.6,
    "lines.markersize": 4,
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
})


def read_shape_group(h5file, groupname):
    with h5py.File(h5file, "r") as f:
        g = f[groupname]
        r = g["rmid_kpc"][:]
        ba = g["b_over_a"][:]
        ca = g["c_over_a"][:]
    return r, ba, ca


def read_rho_group(h5file):
    with h5py.File(h5file, "r") as f:
        g = f["rho3d"]
        r = g["rmid_kpc"][:]
        rho = g["rho_msun_kpc3"][:]
    return r, rho


def make_shape_compare(
    hydro_h5,
    dmo_h5,
    outdir,
    hydro_subhaloID,
    dark_subhaloID,
    groupname,
    label_suffix
):
    rh, bah, cah = read_shape_group(hydro_h5, groupname)
    rd, bad, cad = read_shape_group(dmo_h5, groupname)

    plt.figure(figsize=(7.0, 4.8))

    plt.plot(rh, bah, color="tab:red", ls="-", label="Hydro b/a")
    plt.plot(rh, cah, color="tab:blue", ls="-", label="Hydro c/a")

    plt.plot(rd, bad, color="tab:red", ls="--", label="DMO b/a")
    plt.plot(rd, cad, color="tab:blue", ls="--", label="DMO c/a")

    plt.xlabel("R [kpc]")
    plt.ylabel("axis ratio")
    plt.ylim(0.35, 1.02)
    plt.grid(True, alpha=0.25)
    plt.legend(ncol=2)

    plt.title(
        f"Hydro vs DMO 3D DM shape | hydro {hydro_subhaloID} | dark {dark_subhaloID}"
    )

    outpath = os.path.join(
        outdir,
        f"compare_hydro_dmo_3D_shape_{label_suffix}_hydro{hydro_subhaloID}_dark{dark_subhaloID}.png"
    )

    plt.savefig(outpath, dpi=200, bbox_inches="tight")
    plt.close()

    print("Saved", outpath)


def make_rho_compare(
    hydro_h5,
    dmo_h5,
    outdir,
    hydro_subhaloID,
    dark_subhaloID
):
    rh, rhoh = read_rho_group(hydro_h5)
    rd, rhod = read_rho_group(dmo_h5)

    plt.figure(figsize=(6.4, 4.8))

    plt.loglog(rh, rhoh, label="Hydro DM")
    plt.loglog(rd, rhod, ls="--", label="DMO DM")

    plt.xlabel("R [kpc]")
    plt.ylabel(r"$\rho_{\rm DM}$ [Msun/kpc$^3$]")
    plt.grid(True, which="both", alpha=0.25)
    plt.legend()

    plt.title(
        f"Hydro vs DMO density | hydro {hydro_subhaloID} | dark {dark_subhaloID}"
    )

    outpath = os.path.join(
        outdir,
        f"compare_hydro_dmo_rho_hydro{hydro_subhaloID}_dark{dark_subhaloID}.png"
    )

    plt.savefig(outpath, dpi=200, bbox_inches="tight")
    plt.close()

    print("Saved", outpath)


if __name__ == "__main__":

    hydro_subhaloID = 372755
    dark_subhaloID = 591990

    hydro_h5 = "outputs_372755/dm_sub372755.hdf5"
    dmo_h5 = "outputs_372755_dark/dmo_hydro372755_dark591990.hdf5"

    outdir = "outputs_372755_compare"
    os.makedirs(outdir, exist_ok=True)

    make_shape_compare(
        hydro_h5=hydro_h5,
        dmo_h5=dmo_h5,
        outdir=outdir,
        hydro_subhaloID=hydro_subhaloID,
        dark_subhaloID=dark_subhaloID,
        groupname="shape3d",
        label_suffix="cumulative"
    )

    make_shape_compare(
        hydro_h5=hydro_h5,
        dmo_h5=dmo_h5,
        outdir=outdir,
        hydro_subhaloID=hydro_subhaloID,
        dark_subhaloID=dark_subhaloID,
        groupname="shape3d_shells",
        label_suffix="shells"
    )

    make_rho_compare(
        hydro_h5=hydro_h5,
        dmo_h5=dmo_h5,
        outdir=outdir,
        hydro_subhaloID=hydro_subhaloID,
        dark_subhaloID=dark_subhaloID
    )
