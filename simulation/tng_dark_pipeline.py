import os
import sys
import time
import gc
import shutil
from multiprocessing import Pool

import numpy as np
import h5py
import matplotlib.pyplot as plt
import illustris_python as il

# ============================================================
# PLOT STYLE
# ============================================================

plt.rcParams.update({
    "font.size": 8,
    "axes.titlesize": 10,
    "axes.labelsize": 10,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "figure.titlesize": 10,
    "lines.linewidth": 1.4,
    "lines.markersize": 3.2,
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
})

# Import useful functions from your hydro pipeline
from tng_pipeline import (
    flag,
    Timer,
    wrap_relative_periodic,
    make_DM_sigma_fast_sph,
    compute_3D_shape,
    compute_3D_shape_shells,
    compute_rho_spherical,
    principal_axes_within_R,
    shape_profiles_from_linear_map,
    radial_profile_linear,
    interp_nan_1d,
    clean_pa_for_plot,
    plot_log_map_percentile,
    safe_tight_layout,
)


# ============================================================
# CONFIG
# ============================================================

basePath_hydro = "/home/tnguser/sims.TNG/L35n2160TNG/output"
basePath_dark  = "/home/tnguser/sims.TNG/L35n2160TNG_DM/output"

match_file = "/home/tnguser/sims.TNG/L35n2160TNG/postprocessing/released/subhalo_matching_to_dark.hdf5"

snapNum = 99


# ============================================================
# MATCHING FUNCTION
# ============================================================

def get_matched_dark_subhalo_id(hydro_subhaloID, snapNum=99, method="SubLink"):
   
    snap_key = f"Snapshot_{snapNum}"

    if method == "SubLink":
        dataset = "SubhaloIndexDark_SubLink"
    elif method == "LHaloTree":
        dataset = "SubhaloIndexDark_LHaloTree"
    else:
        raise ValueError("method must be 'SubLink' or 'LHaloTree'")

    with h5py.File(match_file, "r") as f:
        dark_id = int(f[snap_key][dataset][hydro_subhaloID])

    if dark_id < 0:
        raise RuntimeError(
            f"No matched DMO subhalo found for hydro subhaloID={hydro_subhaloID}"
        )

    return dark_id


# ============================================================
# DM PARTICLE MASS
# ============================================================

def get_dm_particle_mass_msun(basePath, snapNum):
    
    header = il.groupcat.loadHeader(basePath, snapNum)
    h = header["HubbleParam"]

    try:
        m_dm_code = header["MassTable"][1]
    except KeyError:
        with h5py.File(
            f"{basePath}/snapdir_{snapNum:03d}/snap_{snapNum:03d}.0.hdf5",
            "r"
        ) as f0:
            m_dm_code = f0["Header"].attrs["MassTable"][1]

    return m_dm_code * 1e10 / h


# ============================================================
# DMO MAP PROJECTIONS
# ============================================================

def run_dmo_dm_projections_from_loaded(
    coords_dm_kpc,
    hsml_kpc,
    m_dm_msun,
    R200c_kpc,
    out_h5,
    extent_factor=1.0,
    nbins=512,
    hydro_subhaloID=None,
    dark_subhaloID=None,
    dark_groupID=None,
    snapNum=99,
    R200c_dark_kpc=np.nan,
):
  

    flag(
        f"DMO projections START | hydro={hydro_subhaloID} | "
        f"dark={dark_subhaloID} | group={dark_groupID}"
    )

    extent_kpc = extent_factor * R200c_kpc

    coords = coords_dm_kpc

    r3 = np.linalg.norm(coords, axis=1)

    flag(
        f"DMO diagnostics: r_max={np.max(r3):.2f} kpc | "
        f"r95={np.percentile(r3, 95):.2f} kpc | "
        f"N(r<R200c_used)={np.sum(r3 < R200c_kpc):,}"
    )

    # Simulation-axis projections:
    # xy = project along z
    # xz = project along y
    # yz = project along x
    proj_defs = {
        "xy": (coords, (0, 1)),
        "xz": (coords, (0, 2)),
        "yz": (coords, (1, 2)),
    }

    with h5py.File(out_h5, "a") as f:

        if "meta" in f:
            del f["meta"]

        # Delete old projection groups from previous runs.
        for k in ["xy", "xz", "yz", "faceon", "edgeon", "ac_plane"]:
            if k in f:
                del f[k]

        meta = f.create_group("meta")

        meta.attrs.update({
            "hydro_subhaloID": int(hydro_subhaloID),
            "dark_subhaloID": int(dark_subhaloID),
            "dark_groupID": int(dark_groupID),
            "snapNum": int(snapNum),
            "R200c_used_kpc": float(R200c_kpc),
            "R200c_dark_kpc": float(R200c_dark_kpc),
            "extent_factor": float(extent_factor),
            "extent_kpc": float(extent_kpc),
            "nbins": int(nbins),
            "dm_sph_smooth": True,
            "N_dm_total": int(coords.shape[0]),
            "m_dm_msun": float(m_dm_msun),
            "rmax_kpc_all": float(np.max(r3)),
            "r95_kpc_all": float(np.percentile(r3, 95)),
            "N_r_lt_R200c_used": int(np.sum(r3 < R200c_kpc)),
            "projections": "xy,xz,yz",
        })

        for name, (cc, (i1, i2)) in proj_defs.items():

            flag(f"DMO projection: {name}")

            coords2d = cc[:, [i1, i2]]

            with Timer(f"DMO SPH map {name}"):

                SigmaDM, xcent, ycent = make_DM_sigma_fast_sph(
                    coords2d,
                    hsml_kpc,
                    m_dm_msun,
                    extent_kpc,
                    nbins
                )

                mask_used = (
                    (np.abs(coords2d[:, 0]) < extent_kpc) &
                    (np.abs(coords2d[:, 1]) < extent_kpc)
                )

                Nused = int(mask_used.sum())

                Rp = np.sqrt(coords2d[:, 0]**2 + coords2d[:, 1]**2)

                meta.attrs[f"Rproj_max_used_{name}"] = (
                    float(np.max(Rp[mask_used])) if np.any(mask_used) else np.nan
                )

            meta.attrs[f"N_used_{name}"] = int(Nused)

            g = f.create_group(name)

            g.create_dataset("x_centers_kpc", data=xcent)
            g.create_dataset("y_centers_kpc", data=ycent)
            g.create_dataset("SigmaDM_Msun_kpc2", data=SigmaDM)

            del SigmaDM, coords2d
            gc.collect()

    flag("DMO projections DONE")

    return out_h5


# ============================================================
# OPTIONAL DMO A-C PROJECTION
# ============================================================

def add_dmo_ac_projection_to_h5(
    dm_h5,
    coords_dm_kpc,
    hsml_kpc,
    m_dm_msun,
    R200c_kpc,
    extent_kpc,
    nbins=512
):
  

    ea, eb, ec, eigvals = principal_axes_within_R(
        coords_dm_kpc,
        R200c_kpc
    )

    basis = np.column_stack([ea, eb, ec])

    coords_abc = coords_dm_kpc @ basis

    # a-c plane: x = a, y = c, line-of-sight = b
    coords2d_ac = coords_abc[:, [0, 2]]

    with Timer("DMO SPH map ac_plane"):
        SigmaAC, xcent, ycent = make_DM_sigma_fast_sph(
            coords2d_ac,
            hsml_kpc,
            m_dm_msun,
            extent_kpc,
            nbins
        )

    with h5py.File(dm_h5, "a") as f:
        if "ac_plane" in f:
            del f["ac_plane"]

        g = f.create_group("ac_plane")

        g.create_dataset("x_centers_kpc", data=xcent)
        g.create_dataset("y_centers_kpc", data=ycent)
        g.create_dataset("SigmaDM_Msun_kpc2", data=SigmaAC)

        g.create_dataset("e_a", data=ea)
        g.create_dataset("e_b", data=eb)
        g.create_dataset("e_c", data=ec)
        g.create_dataset("eigvals_I", data=eigvals)

        g.attrs["R_axes_defined_kpc"] = float(R200c_kpc)
        g.attrs["projection_plane"] = "a-c (project along b)"

    del SigmaAC, coords_abc, coords2d_ac
    gc.collect()


# ============================================================
# DMO ANALYSIS + PLOTS
# ============================================================

def run_dmo_analysis_and_plots(
    dm_h5,
    outdir,
    rbinstep_kpc=5.0,
    Nmin_pixels=300
):

    projections = ["xy", "xz", "yz"]

    with h5py.File(dm_h5, "r+") as f:
        meta = f["meta"].attrs

        hydro_subhaloID = int(meta["hydro_subhaloID"])
        dark_subhaloID = int(meta["dark_subhaloID"])
        extent = float(meta["extent_kpc"])
        R200c_used = float(meta["R200c_used_kpc"])

        rbins = np.arange(0, extent + 1e-6, rbinstep_kpc)

        # ============================================================
        # 3D SHAPE: CUMULATIVE
        # ============================================================

        if "shape3d" in f:
            g3 = f["shape3d"]

            rmid3d = g3["rmid_kpc"][:]
            ba3d = g3["b_over_a"][:]
            ca3d = g3["c_over_a"][:]

            plt.figure(figsize=(6, 4))

            plt.plot(rmid3d, ba3d, label="b/a cumulative")
            plt.plot(rmid3d, ca3d, label="c/a cumulative")

            plt.axvline(R200c_used, ls="--", label="R200c used")

            plt.xlabel("R [kpc]")
            plt.ylabel("axis ratio")
            plt.title(
                f"DMO cumulative shape | hydro {hydro_subhaloID} | dark {dark_subhaloID}"
            )
            plt.grid(True)
            plt.legend()
            safe_tight_layout()

            plt.savefig(
                f"{outdir}/DMO_3D_ba_ca_hydro{hydro_subhaloID}_dark{dark_subhaloID}.png",
                dpi=200
            )
            plt.close()

        # ============================================================
        # 3D SHAPE: SHELLS
        # ============================================================

        if "shape3d_shells" in f:
            g3s = f["shape3d_shells"]

            rmid3d_s = g3s["rmid_kpc"][:]
            ba3d_s = g3s["b_over_a"][:]
            ca3d_s = g3s["c_over_a"][:]

            plt.figure(figsize=(6, 4))

            plt.plot(rmid3d_s, ba3d_s, marker="o", label="b/a shell")
            plt.plot(rmid3d_s, ca3d_s, marker="s", label="c/a shell")

            plt.axvline(R200c_used, ls="--", label="R200c used")

            plt.xlabel("R [kpc]")
            plt.ylabel("axis ratio")
            plt.title(
                f"DMO shell shape | hydro {hydro_subhaloID} | dark {dark_subhaloID}"
            )
            plt.grid(True)
            plt.legend()
            safe_tight_layout()

            plt.savefig(
                f"{outdir}/DMO_3D_shell_ba_ca_hydro{hydro_subhaloID}_dark{dark_subhaloID}.png",
                dpi=200
            )
            plt.close()

        # ============================================================
        # LOOP OVER xy/xz/yz
        # ============================================================

        for proj in projections:
            flag(f"DMO analysis projection: {proj}")

            g = f[proj]

            x = g["x_centers_kpc"][:]
            y = g["y_centers_kpc"][:]
            SigmaDM = g["SigmaDM_Msun_kpc2"][:]

            # ---------------- DM map ----------------

            Sdm = np.array(SigmaDM, dtype=float)

            Sdm[~np.isfinite(Sdm)] = np.nan
            Sdm[Sdm <= 0] = np.nan

            logDM = np.log10(Sdm)

            plot_log_map_percentile(
                logDM,
                extent,
                cmap_name="inferno",
                outpath=f"{outdir}/DMO_DM_{hydro_subhaloID}_{proj}.png",
                title=f"DMO matched halo | hydro {hydro_subhaloID} | {proj}",
                cbar_label=r"log $\Sigma_{DM}$ [Msun/kpc$^2$]",
                p_lo=5,
                p_hi=99
            )

            # ---------------- radial projected SigmaDM profile ----------------

            rdm, dm_prof_linear = radial_profile_linear(
                SigmaDM,
                x,
                y,
                rbins,
                statistic="mean"
            )

            dm_prof = np.full_like(dm_prof_linear, np.nan, dtype=float)

            ok = dm_prof_linear > 0
            dm_prof[ok] = np.log10(dm_prof_linear[ok])

            plt.figure(figsize=(6, 4))

            plt.plot(rdm, dm_prof, marker="o")

            plt.xlabel("R [kpc]")
            plt.ylabel(r"$\log_{10}\langle \Sigma_{DM} \rangle$ [Msun/kpc$^2$]")
            plt.title(f"DMO DM surface density profile | {proj}")
            plt.grid(True)
            safe_tight_layout()

            plt.savefig(
                f"{outdir}/DMO_SigmaDMProf_{hydro_subhaloID}_{proj}.png",
                dpi=200
            )
            plt.close()

            # ---------------- q and PA profiles ----------------

            rmid, qD, paD = shape_profiles_from_linear_map(
                SigmaDM,
                x,
                y,
                rbins,
                Nmin_pixels=Nmin_pixels,
                cumulative=False,
                q_mask=0.98
            )

            rmid_c, qD_c, paD_c = shape_profiles_from_linear_map(
                SigmaDM,
                x,
                y,
                rbins,
                Nmin_pixels=Nmin_pixels,
                cumulative=True,
                q_mask=0.98
            )

            # q plot
            plt.figure(figsize=(6, 4))

            plt.plot(rmid, qD, marker="o", label="DMO DM annuli")
            plt.plot(rmid_c, qD_c, ls="--", label="DMO DM <r")

            plt.xlabel("R [kpc]")
            plt.ylabel("q=b/a")
            plt.title(f"DMO q(R) | hydro {hydro_subhaloID} | {proj}")
            plt.grid(True)
            plt.legend()
            safe_tight_layout()

            plt.savefig(
                f"{outdir}/DMO_q_{hydro_subhaloID}_{proj}.png",
                dpi=200
            )
            plt.close()

            # PA plot
            plt.figure(figsize=(6, 4))

            plt.plot(
                rmid,
                clean_pa_for_plot(paD),
                marker="o",
                markersize=3,
                lw=1.1,
                label="DMO DM annuli"
            )

            plt.plot(
                rmid_c,
                clean_pa_for_plot(paD_c),
                ls="--",
                lw=1.1,
                label="DMO DM <r"
            )

            plt.xlabel("R [kpc]")
            plt.ylabel("PA [deg]")
            plt.title(f"DMO PA(R) | hydro {hydro_subhaloID} | {proj}")
            plt.grid(True)
            plt.legend()
            safe_tight_layout()

            plt.savefig(
                f"{outdir}/DMO_pa_{hydro_subhaloID}_{proj}.png",
                dpi=200
            )
            plt.close()

            # ---------------- save derived arrays ----------------

            if "derived" not in g:
                dg = g.create_group("derived")
            else:
                dg = g["derived"]

            for name in [
                "rbins_kpc",
                "rmid_kpc",
                "q_dm",
                "pa_dm",
                "rmid_cum_kpc",
                "q_dm_cum",
                "pa_dm_cum",
                "SigmaDM_prof_linear"
            ]:
                if name in dg:
                    del dg[name]

            dg.create_dataset("rbins_kpc", data=rbins)
            dg.create_dataset("rmid_kpc", data=rmid)
            dg.create_dataset("q_dm", data=qD)
            dg.create_dataset("pa_dm", data=paD)
            dg.create_dataset("rmid_cum_kpc", data=rmid_c)
            dg.create_dataset("q_dm_cum", data=qD_c)
            dg.create_dataset("pa_dm_cum", data=paD_c)
            dg.create_dataset("SigmaDM_prof_linear", data=dm_prof_linear)

            del SigmaDM, Sdm, logDM
            gc.collect()

    return True


# ============================================================
# MAIN DMO PIPELINE
# ============================================================

def run_dark_analysis(
    hydro_subhaloID,
    R200c_kpc,
    hydro_outdir,
    outdir="outputs_dark_one",
    extent_factor=1.0,
    nbins=512,
    rbinstep_kpc=5.0,
    Nmin_pixels=300
):

    os.makedirs(outdir, exist_ok=True)

    # ------------------------------------------------------------
    # 1) Match hydro subhalo to DMO subhalo
    # ------------------------------------------------------------

    with Timer("MATCH hydro to DMO"):
        dark_subhaloID = get_matched_dark_subhalo_id(
            hydro_subhaloID,
            snapNum=snapNum,
            method="SubLink"
        )

    flag(
        f"hydro_subhaloID={hydro_subhaloID} matched "
        f"dark_subhaloID={dark_subhaloID}"
    )

    # ------------------------------------------------------------
    # 2) Load DMO subhalo and FoF halo metadata
    # ------------------------------------------------------------

    with Timer("DMO: load header/subhalo"):
        header = il.groupcat.loadHeader(basePath_dark, snapNum)

        h = header["HubbleParam"]
        z = header["Redshift"]
        a = 1.0 / (1.0 + z)

        boxsize_kpc = header["BoxSize"] * a / h

        sub_dark = il.groupcat.loadSingle(
            basePath_dark,
            snapNum,
            subhaloID=dark_subhaloID
        )

        dark_groupID = int(sub_dark["SubhaloGrNr"])
        subpos_dark_kpc = sub_dark["SubhaloPos"] * a / h

        # Try to read DMO R200c for reference.
        # The actual radius used for comparison is R200c_kpc passed from hydro.
        try:
            group_dark = il.groupcat.loadSingle(
                basePath_dark,
                snapNum,
                haloID=dark_groupID
            )

            R200c_dark_kpc = group_dark["Group_R_Crit200"] * a / h

        except Exception:
            R200c_dark_kpc = np.nan

    # ------------------------------------------------------------
    # 3) Load DMO FoF DM particles
    # ------------------------------------------------------------

    with Timer("DMO: load FoF DM particles"):
        dm_fields = ["Coordinates", "SubfindHsml"]

        dm = il.snapshot.loadHalo(
            basePath_dark,
            snapNum,
            dark_groupID,
            "dm",
            fields=dm_fields
        )

    # ------------------------------------------------------------
    # 4) Center/wrap coordinates
    # ------------------------------------------------------------

    with Timer("DMO: center/wrap coords"):
        coords_ckpch = dm["Coordinates"]
        hsml_ckpch = dm["SubfindHsml"]

        coords_dm = coords_ckpch * a / h - subpos_dark_kpc
        coords_dm = wrap_relative_periodic(coords_dm, boxsize_kpc)

        hsml_kpc = hsml_ckpch * a / h

    flag(f"DMO loaded: N = {coords_dm.shape[0]:,}")

    m_dm_msun = get_dm_particle_mass_msun(basePath_dark, snapNum)

    dm_h5 = os.path.join(
        outdir,
        f"dmo_hydro{hydro_subhaloID}_dark{dark_subhaloID}.hdf5"
    )

    # ------------------------------------------------------------
    # 5) 3D shape: cumulative + shells
    # ------------------------------------------------------------

    with Timer("DMO: 3D shape cumulative + shells"):
        rbins_3d = np.linspace(0, R200c_kpc, 25)

        (
            rmid3d,
            a_arr,
            b_arr,
            c_arr,
            ba_arr,
            ca_arr,
            N_within_r
        ) = compute_3D_shape(
            coords_dm,
            rbins_3d,
            Nmin=1000
        )

        (
            rmid3d_shell,
            a_shell,
            b_shell,
            c_shell,
            ba_shell,
            ca_shell,
            N_shell
        ) = compute_3D_shape_shells(
            coords_dm,
            rbins_3d,
            Nmin=1000
        )

    # ------------------------------------------------------------
    # 6) rho profile
    # ------------------------------------------------------------

    with Timer("DMO: rho profile"):
        rmin = max(0.01 * R200c_kpc, 1e-3)

        rbins_rho = np.logspace(
            np.log10(rmin),
            np.log10(R200c_kpc),
            30
        )

        rmid_rho, rho_dm, Nshell_dm = compute_rho_spherical(
            coords_dm,
            m_dm_msun,
            rbins_rho
        )

    # ------------------------------------------------------------
    # 7) Save 3D/rho
    # ------------------------------------------------------------

    with Timer("DMO: save 3D results"):
        with h5py.File(dm_h5, "w") as f:

            meta = f.create_group("meta_initial")

            meta.attrs["hydro_subhaloID"] = int(hydro_subhaloID)
            meta.attrs["dark_subhaloID"] = int(dark_subhaloID)
            meta.attrs["dark_groupID"] = int(dark_groupID)
            meta.attrs["R200c_used_kpc"] = float(R200c_kpc)
            meta.attrs["R200c_dark_kpc"] = float(R200c_dark_kpc)

            # cumulative shape
            g3d = f.create_group("shape3d")

            g3d.create_dataset("rbins_kpc", data=rbins_3d)
            g3d.create_dataset("rmid_kpc", data=rmid3d)
            g3d.create_dataset("a_kpc", data=a_arr)
            g3d.create_dataset("b_kpc", data=b_arr)
            g3d.create_dataset("c_kpc", data=c_arr)
            g3d.create_dataset("b_over_a", data=ba_arr)
            g3d.create_dataset("c_over_a", data=ca_arr)
            g3d.create_dataset("N_within_r", data=N_within_r)

            g3d.attrs["shape_type"] = "cumulative"

            # shell shape
            g3s = f.create_group("shape3d_shells")

            g3s.create_dataset("rbins_kpc", data=rbins_3d)
            g3s.create_dataset("rmid_kpc", data=rmid3d_shell)
            g3s.create_dataset("a_kpc", data=a_shell)
            g3s.create_dataset("b_kpc", data=b_shell)
            g3s.create_dataset("c_kpc", data=c_shell)
            g3s.create_dataset("b_over_a", data=ba_shell)
            g3s.create_dataset("c_over_a", data=ca_shell)
            g3s.create_dataset("N_shell", data=N_shell)

            g3s.attrs["shape_type"] = "shell"

            # spherical density profile
            gr = f.create_group("rho3d")

            gr.create_dataset("rbins_kpc", data=rbins_rho)
            gr.create_dataset("rmid_kpc", data=rmid_rho)
            gr.create_dataset("rho_msun_kpc3", data=rho_dm)
            gr.create_dataset("N_shell", data=Nshell_dm)

    # ------------------------------------------------------------
    # 8) DMO SPH maps xy/xz/yz
    # ------------------------------------------------------------

    with Timer("DMO: SPH projections"):
        run_dmo_dm_projections_from_loaded(
            coords_dm_kpc=coords_dm,
            hsml_kpc=hsml_kpc,
            m_dm_msun=m_dm_msun,
            R200c_kpc=R200c_kpc,
            out_h5=dm_h5,
            extent_factor=extent_factor,
            nbins=nbins,
            hydro_subhaloID=hydro_subhaloID,
            dark_subhaloID=dark_subhaloID,
            dark_groupID=dark_groupID,
            snapNum=snapNum,
            R200c_dark_kpc=R200c_dark_kpc,
        )

    # ------------------------------------------------------------
    # 9) Optional DMO a-c projection
    # ------------------------------------------------------------
    # Skipped for normal runs.
    # The important 3D shape information is already saved in:
    # shape3d and shape3d_shells.
    #
    # with Timer("DMO: a-c projection"):
    #     extent_kpc = extent_factor * R200c_kpc
    #
    #     add_dmo_ac_projection_to_h5(
    #         dm_h5=dm_h5,
    #         coords_dm_kpc=coords_dm,
    #         hsml_kpc=hsml_kpc,
    #         m_dm_msun=m_dm_msun,
    #         R200c_kpc=R200c_kpc,
    #         extent_kpc=extent_kpc,
    #         nbins=nbins
    #     )

    # ------------------------------------------------------------
    # 10) cleanup before plotting
    # ------------------------------------------------------------

    del dm
    del coords_ckpch
    del hsml_ckpch
    del coords_dm
    del hsml_kpc

    gc.collect()

    # ------------------------------------------------------------
    # 11) analysis and plotting
    # ------------------------------------------------------------

    with Timer("DMO: analysis and plotting"):
        run_dmo_analysis_and_plots(
            dm_h5=dm_h5,
            outdir=outdir,
            rbinstep_kpc=rbinstep_kpc,
            Nmin_pixels=Nmin_pixels
        )

    return {
        "hydro_subhaloID": hydro_subhaloID,
        "dark_subhaloID": dark_subhaloID,
        "dm_h5": dm_h5,
        "outdir": outdir
    }


# ============================================================
# RUN BLOCK
# ============================================================

def run_dark_job(
    hydro_subhaloID,
    R200c_kpc,
    hydro_outdir,
    outdir,
    extent_factor,
    nbins
):
    return run_dark_analysis(
        hydro_subhaloID=hydro_subhaloID,
        R200c_kpc=R200c_kpc,
        hydro_outdir=hydro_outdir,
        outdir=outdir,
        extent_factor=extent_factor,
        nbins=nbins,
        rbinstep_kpc=5.0,
        Nmin_pixels=300
    )


if __name__ == "__main__":

    jobs = [
        # hydro_subhaloID, R200c_kpc, hydro_outdir, dark_outdir, extent_factor, nbins
        (388544, 350.348, "outputs_388544", "outputs_388544_dark", 1.0, 512),
        (392277, 333.529, "outputs_392277", "outputs_392277_dark", 1.0, 512),
        (400973, 328.2358, "outputs_400973", "outputs_400973_dark", 1.0, 512),
        (400974, 328.2358, "outputs_400974", "outputs_400974_dark", 1.0, 512),
    ]

    for job in jobs:
        outdir = job[3]
        shutil.rmtree(outdir, ignore_errors=True)

    nproc = 1

    print("Starting DMO pipeline...")
    print(f"Number of jobs = {len(jobs)}")
    print(f"Number of processes = {nproc}")
    sys.stdout.flush()

    with Pool(processes=nproc) as pool:
        results = pool.starmap(run_dark_job, jobs)

    print("Finished DMO jobs.")
    print(results)
