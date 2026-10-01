#imports
#flag and timer
import os
import sys
import time
import gc
import shutil
from multiprocessing import Pool

import numpy as np
import h5py
import matplotlib.pyplot as plt
from copy import copy

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

def flag(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}")
    sys.stdout.flush()


class Timer:
    def __init__(self, name):
        self.name = name

    def __enter__(self):
        self.t0 = time.perf_counter()
        flag(f"START {self.name}")

    def __exit__(self, *args):
        dt = time.perf_counter() - self.t0
        flag(f"END   {self.name} ({dt:.2f} s)")


# ============================================================
# CONFIG
# ============================================================

basePath = "/home/tnguser/sims.TNG/L35n2160TNG/output"
snapNum = 99


# ============================================================
# BASIC HELPERS
# ============================================================

def cubic_spline_W_2D(r, h):
    q = r / h
    sigma = 10.0 / (7.0 * np.pi * h**2)

    W = np.zeros_like(r)

    m1 = (q >= 0) & (q < 1)
    m2 = (q >= 1) & (q < 2)

    W[m1] = 1 - 1.5*q[m1]**2 + 0.75*q[m1]**3
    W[m2] = 0.25 * (2 - q[m2])**3

    return sigma * W


def rotation_matrix_from_vectors(a, b):
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)

    v = np.cross(a, b)
    c = np.dot(a, b)
    s = np.linalg.norm(v)

    if s == 0:
        return np.eye(3)

    vx = np.array([
        [0, -v[2], v[1]],
        [v[2], 0, -v[0]],
        [-v[1], v[0], 0]
    ])

    R = np.eye(3) + vx + vx @ vx * ((1 - c) / s**2)

    return R


def wrap_relative_periodic(dx, boxsize_kpc):
    """Minimal-image convention for periodic boundaries."""
    return dx - boxsize_kpc * np.round(dx / boxsize_kpc)


# ============================================================
# STAR FUNCTIONS
# ============================================================

def compute_stellar_Lhat(coords_kpc, vels_kms, masses_msun, Rcut_kpc=20.0):
    R = np.linalg.norm(coords_kpc, axis=1)

    m = R < Rcut_kpc

    if np.count_nonzero(m) < 200:
        m = R < np.nanpercentile(R, 30)

    r = coords_kpc[m]
    v = vels_kms[m]
    w = masses_msun[m]

    L = np.sum(np.cross(r, v) * w[:, None], axis=0)

    if np.linalg.norm(L) == 0:
        return np.array([0.0, 0.0, 1.0])

    return L / np.linalg.norm(L)




def _normalize_vector(v, fallback=np.array([0.0, 0.0, 1.0])):
    v = np.asarray(v, dtype=float)

    if v.shape != (3,) or not np.all(np.isfinite(v)):
        return np.asarray(fallback, dtype=float)

    n = np.linalg.norm(v)

    if n <= 0:
        return np.asarray(fallback, dtype=float)

    return v / n


def choose_edgeon_los_from_Lhat(Lhat):
    """
    Face-on LOS is parallel to stellar angular momentum.
    Edge-on LOS is perpendicular to stellar angular momentum.
    """
    Lhat = _normalize_vector(Lhat)

    ref = np.array([1.0, 0.0, 0.0])

    if abs(np.dot(ref, Lhat)) > 0.90:
        ref = np.array([0.0, 1.0, 0.0])

    los = np.cross(Lhat, ref)

    return _normalize_vector(los, fallback=np.array([1.0, 0.0, 0.0]))


def project_coords_to_los(coords_kpc, los):
    """
    Project 3D coordinates onto a 2D plane perpendicular to the chosen LOS.
    """
    coords_kpc = np.asarray(coords_kpc, dtype=float)
    los = _normalize_vector(los)

    ref = np.array([0.0, 0.0, 1.0])

    if abs(np.dot(ref, los)) > 0.90:
        ref = np.array([0.0, 1.0, 0.0])

    e1 = np.cross(ref, los)
    e1 = _normalize_vector(e1, fallback=np.array([1.0, 0.0, 0.0]))

    e2 = np.cross(los, e1)
    e2 = _normalize_vector(e2, fallback=np.array([0.0, 1.0, 0.0]))

    x = coords_kpc @ e1
    y = coords_kpc @ e2

    coords2d = np.column_stack((x, y))

    return coords2d, e1, e2, los

def SigmaL_to_mu(Sigma_L_kpc2, M_sun_r=4.65):
    I_pc2 = Sigma_L_kpc2 / 1e6

    mu = np.full_like(I_pc2, np.nan, dtype=float)

    ok = I_pc2 > 0

    mu[ok] = M_sun_r + 21.572 - 2.5*np.log10(I_pc2[ok])

    return mu

def make_surface_density_temet(coords2d_kpc, hsml_kpc, weights, extent_kpc, nbins):
    """
    Fast SPH surface-density map using temet.util.sphMap.sphMap.
    """

    from temet.util.sphMap import sphMap

    pos = np.asarray(coords2d_kpc, dtype=np.float32)
    hsml = np.asarray(hsml_kpc, dtype=np.float32)
    weights = np.asarray(weights, dtype=np.float32)

    good = (
        np.isfinite(pos[:, 0]) &
        np.isfinite(pos[:, 1]) &
        np.isfinite(hsml) &
        np.isfinite(weights) &
        (hsml > 0) &
        (weights > 0)
    )

    pos = pos[good]
    hsml = hsml[good]
    weights = weights[good]

    box_size = 2.0 * float(extent_kpc)

    Sigma = sphMap(
        pos=pos,
        hsml=hsml,
        mass=weights,
        quant=None,
        axes=[0, 1],
        boxSizeImg=[box_size, box_size],
        boxSizeSim=0.0,
        boxCen=[0.0, 0.0],
        nPixels=[int(nbins), int(nbins)],
        ndims=2,
        colDens=True,
        nThreads=None
    )

    Sigma = np.asarray(Sigma, dtype=float)

    xedges = np.linspace(-extent_kpc, extent_kpc, nbins + 1)
    yedges = np.linspace(-extent_kpc, extent_kpc, nbins + 1)

    x_centers = 0.5 * (xedges[:-1] + xedges[1:])
    y_centers = 0.5 * (yedges[:-1] + yedges[1:])

    return Sigma, x_centers, y_centers

def make_star_SigmaL_map(coords2d_kpc, hsml_kpc, Lsun, extent_kpc, nbins):
    return make_surface_density_temet(
        coords2d_kpc,
        hsml_kpc,
        Lsun,
        extent_kpc,
        nbins
    )


def make_star_Sigma_map(coords2d_kpc, hsml_kpc, weights, extent_kpc, nbins):
    return make_surface_density_temet(
        coords2d_kpc,
        hsml_kpc,
        weights,
        extent_kpc,
        nbins
    )

def run_star_projections(
    basePath,
    snapNum,
    subhaloID,
    R200c_kpc,
    out_h5,
    extent_factor=0.5,
    nbins=512,
    r_band_index=4,
    M_sun_r=4.65,
    Rcut_L_kpc=20.0
):
    header = il.groupcat.loadHeader(basePath, snapNum)

    h = header["HubbleParam"]
    z = header["Redshift"]
    a = 1.0 / (1.0 + z)

    sub = il.groupcat.loadSingle(basePath, snapNum, subhaloID=subhaloID)

    subpos_kpc = sub["SubhaloPos"] * a / h
    subvel_kms = sub["SubhaloVel"]

    fields = [
        "Coordinates",
        "Velocities",
        "Masses",
        "StellarHsml",
        "GFM_StellarPhotometrics"
    ]

    stars = il.snapshot.loadSubhalo(
        basePath,
        snapNum,
        subhaloID,
        "star",
        fields=fields
    )

    if stars["count"] == 0:
        raise RuntimeError(f"Subhalo {subhaloID}: no stars loaded")

    coords = stars["Coordinates"] * a / h - subpos_kpc
    vels = stars["Velocities"] - subvel_kms
    mstar = stars["Masses"] * 1e10 / h
    hsml = stars["StellarHsml"] * a / h

    mag_r = stars["GFM_StellarPhotometrics"][:, r_band_index]
    L_r = 10.0 ** (-0.4 * (mag_r - M_sun_r))

    extent_kpc = extent_factor * R200c_kpc

    Lhat = compute_stellar_Lhat(
        coords,
        vels,
        mstar,
        Rcut_kpc=Rcut_L_kpc
    )

    proj_defs = {
        "xy": ("axis", (0, 1), np.array([0.0, 0.0, 1.0])),
        "xz": ("axis", (0, 2), np.array([0.0, 1.0, 0.0])),
        "yz": ("axis", (1, 2), np.array([1.0, 0.0, 0.0])),
        "faceon": ("los", None, Lhat),
        "edgeon": ("los", None, edgeon_los),
    }

    with h5py.File(out_h5, "w") as f:
        meta = f.create_group("meta")

        meta.attrs.update({
            "subhaloID": int(subhaloID),
            "snapNum": int(snapNum),
            "R200c_kpc": float(R200c_kpc),
            "extent_factor": float(extent_factor),
            "extent_kpc": float(extent_kpc),
            "nbins": int(nbins),
            "r_band_index": int(r_band_index),
            "M_sun_r": float(M_sun_r),
            "Rcut_L_kpc": float(Rcut_L_kpc),
            "projections": "xy,xz,yz,faceon,edgeon",
        })

        meta.create_dataset("Lhat_stars", data=Lhat)
        meta.create_dataset("faceon_los", data=_normalize_vector(Lhat))
        meta.create_dataset("edgeon_los", data=edgeon_los)

        for name, (mode, axes, los_vec) in proj_defs.items():
            flag(f"STAR projection: {name}")

            g = f.create_group(name)

            if mode == "axis":
                coords2d = coords[:, list(axes)]
                g.attrs["projection_mode"] = "simulation_axis"
                g.attrs["los_vector"] = los_vec
            elif mode == "los":
                coords2d, e1, e2, los = project_coords_to_los(coords, los_vec)
                g.attrs["projection_mode"] = "stellar_spin_defined"
                g.attrs["los_vector"] = los
                g.attrs["basis_e1"] = e1
                g.attrs["basis_e2"] = e2
            else:
                raise ValueError(f"Unknown projection mode: {mode}")

            SigmaL, xcent, ycent = make_star_SigmaL_map(
                coords2d,
                hsml,
                L_r,
                extent_kpc,
                nbins
            )

            mu = SigmaL_to_mu(
                SigmaL,
                M_sun_r=M_sun_r
            )

            SigmaStar, _, _ = make_star_Sigma_map(
                coords2d,
                hsml,
                mstar,
                extent_kpc,
                nbins
            )

            g.create_dataset("x_centers_kpc", data=xcent)
            g.create_dataset("y_centers_kpc", data=ycent)
            g.create_dataset("SigmaL_Lsun_kpc2", data=SigmaL)
            g.create_dataset("mu_r_mag_arcsec2", data=mu)
            g.create_dataset("SigmaStar_Msun_kpc2", data=SigmaStar)

            del coords2d, SigmaL, mu, SigmaStar
            gc.collect()

    del stars, coords, vels, mstar, hsml, L_r
    gc.collect()

    return out_h5


# ============================================================
# DM MAP FUNCTIONS
# ============================================================

def make_DM_sigma_hist_counts(coords2d_kpc, m_dm_msun, extent_kpc, nbins):
    mask = (
        (np.abs(coords2d_kpc[:, 0]) < extent_kpc) &
        (np.abs(coords2d_kpc[:, 1]) < extent_kpc)
    )

    x = coords2d_kpc[mask, 0]
    y = coords2d_kpc[mask, 1]

    H, xedges, yedges = np.histogram2d(
        x,
        y,
        bins=nbins,
        range=[
            [-extent_kpc, extent_kpc],
            [-extent_kpc, extent_kpc]
        ]
    )

    pixel_size = 2 * extent_kpc / nbins

    Sigma = (H * m_dm_msun) / (pixel_size**2)

    x_centers = 0.5 * (xedges[:-1] + xedges[1:])
    y_centers = 0.5 * (yedges[:-1] + yedges[1:])

    return Sigma, x_centers, y_centers, int(mask.sum())


def make_DM_sigma_sph(coords2d_kpc, hsml_kpc, m_dm_msun, extent_kpc, nbins):
    xedges = np.linspace(-extent_kpc, extent_kpc, nbins + 1)
    yedges = np.linspace(-extent_kpc, extent_kpc, nbins + 1)

    x_centers = 0.5 * (xedges[:-1] + xedges[1:])
    y_centers = 0.5 * (yedges[:-1] + yedges[1:])

    X, Y = np.meshgrid(x_centers, y_centers, indexing="ij")

    Sigma = np.zeros((nbins, nbins), dtype=float)

    mask = (
        (np.abs(coords2d_kpc[:, 0]) < extent_kpc + 2*hsml_kpc) &
        (np.abs(coords2d_kpc[:, 1]) < extent_kpc + 2*hsml_kpc)
    )

    xs = coords2d_kpc[mask, 0]
    ys = coords2d_kpc[mask, 1]
    hs = hsml_kpc[mask]

    for x0, y0, h0 in zip(xs, ys, hs):
        if not np.isfinite(h0) or h0 <= 0:
            continue

        x_min, x_max = x0 - 2*h0, x0 + 2*h0
        y_min, y_max = y0 - 2*h0, y0 + 2*h0

        ix_min = max(np.searchsorted(x_centers, x_min, side="left"), 0)
        ix_max = min(np.searchsorted(x_centers, x_max, side="right"), nbins)

        iy_min = max(np.searchsorted(y_centers, y_min, side="left"), 0)
        iy_max = min(np.searchsorted(y_centers, y_max, side="right"), nbins)

        if ix_min >= ix_max or iy_min >= iy_max:
            continue

        Xp = X[ix_min:ix_max, iy_min:iy_max]
        Yp = Y[ix_min:ix_max, iy_min:iy_max]

        rp = np.sqrt((Xp - x0)**2 + (Yp - y0)**2)

        Sigma[ix_min:ix_max, iy_min:iy_max] += (
            m_dm_msun * cubic_spline_W_2D(rp, h0)
        )

    return Sigma, x_centers, y_centers


def make_DM_sigma_fast_sph(coords2d_kpc, hsml_kpc, m_dm_msun, extent_kpc, nbins):
    """
    Fast SPH DM surface-density map using temet.util.sphMap.sphMap.

    Returns:
    SigmaDM_Msun_kpc2, x_centers_kpc, y_centers_kpc
    """

    from temet.util.sphMap import sphMap

    pos = np.asarray(coords2d_kpc, dtype=np.float32)
    hsml = np.asarray(hsml_kpc, dtype=np.float32)

    good = (
        np.isfinite(pos[:, 0]) &
        np.isfinite(pos[:, 1]) &
        np.isfinite(hsml) &
        (hsml > 0)
    )

    pos = pos[good]
    hsml = hsml[good]

    mass = np.full(pos.shape[0], m_dm_msun, dtype=np.float32)

    box_size = 2.0 * float(extent_kpc)

    SigmaDM = sphMap(
        pos=pos,
        hsml=hsml,
        mass=mass,
        quant=None,
        axes=[0, 1],
        boxSizeImg=[box_size, box_size],
        boxSizeSim=0.0,
        boxCen=[0.0, 0.0],
        nPixels=[int(nbins), int(nbins)],
        ndims=2,
        colDens=True,
        nThreads=None
    )

    SigmaDM = np.asarray(SigmaDM, dtype=float)

    xedges = np.linspace(-extent_kpc, extent_kpc, nbins + 1)
    yedges = np.linspace(-extent_kpc, extent_kpc, nbins + 1)

    x_centers = 0.5 * (xedges[:-1] + xedges[1:])
    y_centers = 0.5 * (yedges[:-1] + yedges[1:])

    return SigmaDM, x_centers, y_centers


def run_dm_projections_from_loaded(
    coords_dm_kpc,
    hsml_kpc,
    m_dm_msun,
    R200c_kpc,
    out_h5,
    extent_factor=0.5,
    nbins=512,
    R_face=None,
    R_edge=None,
    Lhat_stars=None,
    subhaloID=None,
    snapNum=None,
    use_fof=True,
    periodic=True
):
    """
    Make SPH-smoothed DM projections using already-loaded DM particles.
    """

    flag(
        f"DM projections from loaded arrays START | "
        f"subhalo={subhaloID} | use_fof={use_fof}"
    )

    extent_kpc = extent_factor * R200c_kpc
    coords = coords_dm_kpc

    flag(f"DM already loaded: N = {coords.shape[0]:,}")

    r3 = np.linalg.norm(coords, axis=1)

    flag(
        f"DM diagnostics: r_max={np.max(r3):.2f} kpc | "
        f"r95={np.percentile(r3, 95):.2f} kpc | "
        f"N(r<R200c)={np.sum(r3 < R200c_kpc):,}"
    )

    # Simulation-axis projections plus stellar-spin-defined projections.
    if Lhat_stars is None:
        Lhat_use = np.array([0.0, 0.0, 1.0])
    else:
        Lhat_use = _normalize_vector(Lhat_stars)

    edgeon_los = choose_edgeon_los_from_Lhat(Lhat_use)

    proj_defs = {
        "xy": ("axis", (0, 1), np.array([0.0, 0.0, 1.0])),
        "xz": ("axis", (0, 2), np.array([0.0, 1.0, 0.0])),
        "yz": ("axis", (1, 2), np.array([1.0, 0.0, 0.0])),
        "faceon": ("los", None, Lhat_use),
        "edgeon": ("los", None, edgeon_los),
    }

    with h5py.File(out_h5, "a") as f:

        if "meta" in f:
            del f["meta"]

        for k in ["xy", "xz", "yz", "faceon", "edgeon", "ac_plane"]:
            if k in f:
                del f[k]

        meta = f.create_group("meta")

        meta.attrs.update({
            "subhaloID": int(subhaloID) if subhaloID is not None else -1,
            "snapNum": int(snapNum) if snapNum is not None else -1,
            "R200c_kpc": float(R200c_kpc),
            "extent_factor": float(extent_factor),
            "extent_kpc": float(extent_kpc),
            "nbins": int(nbins),
            "use_fof": bool(use_fof),
            "dm_sph_smooth": True,
            "periodic": bool(periodic),
            "N_dm_total": int(coords.shape[0]),
            "m_dm_msun": float(m_dm_msun),
            "rmax_kpc_all": float(np.max(r3)),
            "r95_kpc_all": float(np.percentile(r3, 95)),
            "N_r_lt_R200c": int(np.sum(r3 < R200c_kpc)),
            "projections": "xy,xz,yz,faceon,edgeon",
        })

        for name, (mode, axes, los_vec) in proj_defs.items():
            flag(f"DM projection: {name}")

            if mode == "axis":
                coords2d = coords[:, list(axes)]
                proj_mode = "simulation_axis"
                los_store = _normalize_vector(los_vec)
                basis_e1_store = np.array([np.nan, np.nan, np.nan])
                basis_e2_store = np.array([np.nan, np.nan, np.nan])

            elif mode == "los":
                coords2d, e1, e2, los = project_coords_to_los(coords, los_vec)
                proj_mode = "stellar_spin_defined"
                los_store = los
                basis_e1_store = e1
                basis_e2_store = e2

            else:
                raise ValueError(f"Unknown projection mode: {mode}")

            with Timer(f"DM SPH map {name}"):

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

            g.attrs["projection_mode"] = proj_mode
            g.attrs["los_vector"] = los_store
            g.attrs["basis_e1"] = basis_e1_store
            g.attrs["basis_e2"] = basis_e2_store

            g.create_dataset("x_centers_kpc", data=xcent)
            g.create_dataset("y_centers_kpc", data=ycent)
            g.create_dataset("SigmaDM_Msun_kpc2", data=SigmaDM)

            del SigmaDM, coords2d
            gc.collect()

    flag("DM projections from loaded arrays DONE")

    return out_h5


# ============================================================
# ANALYSIS FUNCTIONS
# ============================================================

def shape_profiles_from_linear_map(
    weight_map,
    x_centers,
    y_centers,
    rbins,
    Nmin_pixels=300,
    cumulative=False,
    q_mask=0.98
):
    X, Y = np.meshgrid(x_centers, y_centers, indexing="ij")
    R = np.sqrt(X**2 + Y**2)

    W = np.array(weight_map, dtype=float)

    valid = np.isfinite(W) & (W > 0)

    Xf = X[valid]
    Yf = Y[valid]
    Rf = R[valid]
    Wf = W[valid]

    rmid = 0.5 * (rbins[:-1] + rbins[1:])

    q = np.full(len(rmid), np.nan)
    pa = np.full(len(rmid), np.nan)

    for i in range(len(rmid)):

        if cumulative:
            m = Rf < rbins[i+1]
        else:
            m = (Rf >= rbins[i]) & (Rf < rbins[i+1])

        if np.count_nonzero(m) < Nmin_pixels:
            continue

        x = Xf[m]
        y = Yf[m]
        w = Wf[m]

        wsum = np.sum(w)

        x0 = np.sum(w * x) / wsum
        y0 = np.sum(w * y) / wsum

        xc = x - x0
        yc = y - y0

        Cxx = np.sum(w * xc * xc) / wsum
        Cyy = np.sum(w * yc * yc) / wsum
        Cxy = np.sum(w * xc * yc) / wsum

        cov = np.array([
            [Cxx, Cxy],
            [Cxy, Cyy]
        ])

        eigvals, eigvecs = np.linalg.eigh(cov)

        smin = np.sqrt(eigvals[0])
        smaj = np.sqrt(eigvals[1])

        q_here = smin / smaj

        q[i] = q_here

        vx, vy = eigvecs[:, 1]

        ang = np.degrees(np.arctan2(vy, vx))

        # wrap to [-90, 90]
        if ang > 90:
            ang -= 180
        if ang < -90:
            ang += 180

        # PA undefined if too round
        if q_here > q_mask:
            pa[i] = np.nan
            continue

        # continuity to avoid 180-degree flips
        if i > 0 and np.isfinite(pa[i-1]):
            candidates = np.array([ang, ang + 180, ang - 180])
            ang = candidates[np.argmin(np.abs(candidates - pa[i-1]))]

        pa[i] = ang

    return rmid, q, pa


def radial_profile_linear(Sigma_map, x_centers, y_centers, rbins, statistic="mean"):
    X, Y = np.meshgrid(x_centers, y_centers, indexing="ij")
    R = np.sqrt(X**2 + Y**2)

    S = np.array(Sigma_map, dtype=float)

    S[~np.isfinite(S)] = np.nan
    S[S <= 0] = np.nan

    rmid = 0.5 * (rbins[:-1] + rbins[1:])

    prof = np.full(len(rmid), np.nan)

    for i in range(len(rmid)):
        m = (R >= rbins[i]) & (R < rbins[i+1]) & np.isfinite(S)

        if np.any(m):
            if statistic == "median":
                prof[i] = np.nanmedian(S[m])
            else:
                prof[i] = np.nanmean(S[m])

    return rmid, prof


def radial_profile_mu_from_SigmaL(
    SigmaL_map,
    x_centers,
    y_centers,
    rbins,
    M_sun_r=4.65,
    statistic="mean"
):
    rmid, SigmaL_prof = radial_profile_linear(
        SigmaL_map,
        x_centers,
        y_centers,
        rbins,
        statistic=statistic
    )

    mu_prof = SigmaL_to_mu(
        SigmaL_prof,
        M_sun_r=M_sun_r
    )

    return rmid, mu_prof, SigmaL_prof


def interp_nan_1d(y):

    y = np.array(y, dtype=float)

    x = np.arange(len(y))

    ok = np.isfinite(y)

    if ok.sum() < 2:
        return y

    y2 = y.copy()

    y2[~ok] = np.interp(x[~ok], x[ok], y[ok])

    return y2


def unwrap_pa_180(pa_deg):

    pa = np.array(pa_deg, dtype=float)
    out = np.full_like(pa, np.nan)

    ok = np.isfinite(pa)

    if np.count_nonzero(ok) == 0:
        return out

    idx = np.where(ok)[0]

    out[idx[0]] = pa[idx[0]]

    for j in range(1, len(idx)):
        i_prev = idx[j - 1]
        i_now = idx[j]

        prev = out[i_prev]
        val = pa[i_now]

        candidates = np.array([val - 180.0, val, val + 180.0])
        out[i_now] = candidates[np.argmin(np.abs(candidates - prev))]

    return out


def clean_pa_for_plot(pa_deg):
    
    pa_interp = interp_nan_1d(pa_deg)
    return unwrap_pa_180(pa_interp)

def safe_tight_layout():
      
   return


def plot_log_map_percentile(
    logmap,
    extent,
    cmap_name,
    outpath,
    title,
    cbar_label,
    p_lo=5,
    p_hi=99
):
    vmin = np.nanpercentile(logmap, p_lo)
    vmax = np.nanpercentile(logmap, p_hi)

    cmap = copy(getattr(plt.cm, cmap_name))
    cmap.set_bad(color=cmap(0.0))

    plt.figure(figsize=(7, 6))

    im = plt.imshow(
        logmap.T,
        origin="lower",
        extent=[-extent, extent, -extent, extent],
        cmap=cmap,
        vmin=vmin,
        vmax=vmax
    )

    plt.colorbar(im, label=cbar_label)

    plt.title(title)
    plt.xlabel("x [kpc]")
    plt.ylabel("y [kpc]")

    safe_tight_layout()
    plt.savefig(outpath, dpi=200, bbox_inches='tight')
    plt.close()


def run_analysis_and_compare(
    stars_h5,
    dm_h5,
    outdir=".",
    rbinstep_kpc=5.0,
    vmin_mu=18,
    vmax_mu=32,
    vmin_logDM=4,
    vmax_logDM=8,
    Nmin_pixels=300,
    save_results_back=True
):
    

    cumulative_compare = True
    q_mask_value = 0.98

    with h5py.File(stars_h5, "r+") as fs, h5py.File(dm_h5, "r") as fd:

        meta = fs["meta"].attrs

        subID = int(meta["subhaloID"])
        extent = float(meta["extent_kpc"])
        R200c_kpc = float(meta.get("R200c_kpc", np.nan))

        rbins = np.arange(0, extent + 1e-6, rbinstep_kpc)

        # ==========================================================
        # 3D DM SHAPE: CUMULATIVE
        # ==========================================================

        if "shape3d" in fd:
            g3 = fd["shape3d"]

            rmid3d = g3["rmid_kpc"][:]
            ba3d = g3["b_over_a"][:]
            ca3d = g3["c_over_a"][:]

            plt.figure(figsize=(6, 4))

            plt.plot(rmid3d, ba3d, label="b/a cumulative")
            plt.plot(rmid3d, ca3d, label="c/a cumulative")

            if np.isfinite(R200c_kpc):
                plt.axvline(R200c_kpc, ls="--", label="R200c")

            plt.xlabel("R [kpc]")
            plt.ylabel("axis ratio")
            plt.title(f"Subhalo {subID} | 3D DM cumulative shape")
            plt.grid(True)
            plt.legend()
            safe_tight_layout()

            plt.savefig(f"{outdir}/DM3D_ba_ca_{subID}.png", dpi=200, bbox_inches='tight')
            plt.close()

        # ==========================================================
        # 3D DM SHAPE: SHELLS
        # ==========================================================

        if "shape3d_shells" in fd:
            g3s = fd["shape3d_shells"]

            rmid3d_s = g3s["rmid_kpc"][:]
            ba3d_s = g3s["b_over_a"][:]
            ca3d_s = g3s["c_over_a"][:]

            plt.figure(figsize=(6, 4))

            plt.plot(rmid3d_s, ba3d_s, marker="o", label="b/a shell")
            plt.plot(rmid3d_s, ca3d_s, marker="s", label="c/a shell")

            if np.isfinite(R200c_kpc):
                plt.axvline(R200c_kpc, ls="--", label="R200c")

            plt.xlabel("R [kpc]")
            plt.ylabel("axis ratio")
            plt.title(f"Subhalo {subID} | 3D DM shell shape")
            plt.grid(True)
            plt.legend()
            safe_tight_layout()

            plt.savefig(f"{outdir}/DM3D_shell_ba_ca_{subID}.png", dpi=200, bbox_inches='tight')
            plt.close()

        # ==========================================================
        # LOOP OVER xy/xz/yz
        # ==========================================================

        for proj in projections:
            flag(f"ANALYSIS projection: {proj}")

            gs = fs[proj]
            gd = fd[proj]

            x = gs["x_centers_kpc"][:]
            y = gs["y_centers_kpc"][:]

            mu = gs["mu_r_mag_arcsec2"][:]
            SigmaL = gs["SigmaL_Lsun_kpc2"][:]
            SigmaStar = gs["SigmaStar_Msun_kpc2"][:]
            SigmaDM = gd["SigmaDM_Msun_kpc2"][:]

            # ---------------- SB map ----------------

            plt.figure(figsize=(6.2, 5.4))

            mu_plot = np.array(mu, dtype=float)
            mu_plot[~np.isfinite(mu_plot)] = np.nan
            mu_plot[mu_plot > vmax_mu] = np.nan

            cmap_mu = copy(plt.cm.magma)
            cmap_mu.set_bad(color="white")

            im = plt.imshow(
                  mu_plot.T,
                  origin="lower",
                  extent=[-extent, extent, -extent, extent],
                  cmap=cmap_mu,
                  vmin=vmin_mu,
                  vmax=vmax_mu
             )

            plt.colorbar(im, label=r"$\mu_r$ [mag arcsec$^{-2}$]")

            plt.title(f"Subhalo {subID} | {proj} | SB")
            plt.xlabel("x [kpc]")
            plt.ylabel("y [kpc]")

            safe_tight_layout()
            plt.savefig(f"{outdir}/SB_{subID}_{proj}.png", dpi=200, bbox_inches="tight")
            plt.close()
            # ---------------- Stellar surface density map ----------------

            Sstar = np.array(SigmaStar, dtype=float)

            Sstar[~np.isfinite(Sstar)] = np.nan
            Sstar[Sstar <= 0] = np.nan

            logStar = np.log10(Sstar)

            plot_log_map_percentile(
                logStar,
                extent,
                cmap_name="inferno",
                outpath=f"{outdir}/StarSigma_{subID}_{proj}.png",
                title=f"Subhalo {subID} | {proj} | Stars surface density",
                cbar_label=r"log $\Sigma_\star$ [Msun/kpc$^2$]",
                p_lo=5,
                p_hi=99
            )

            # ---------------- DM surface density map ----------------

            Sdm = np.array(SigmaDM, dtype=float)

            Sdm[~np.isfinite(Sdm)] = np.nan
            Sdm[Sdm <= 0] = np.nan

            logDM = np.log10(Sdm)

            plot_log_map_percentile(
                logDM,
                extent,
                cmap_name="inferno",
                outpath=f"{outdir}/DM_{subID}_{proj}.png",
                title=f"Subhalo {subID} | {proj} | DM surface density",
                cbar_label=r"log $\Sigma_{DM}$ [Msun/kpc$^2$]",
                p_lo=5,
                p_hi=99
            )

            # ---------------- Stars radial profile ----------------

            rs, star_prof_linear = radial_profile_linear(
                SigmaStar,
                x,
                y,
                rbins,
                statistic="mean"
            )

            star_prof = np.full_like(star_prof_linear, np.nan, dtype=float)

            ok = star_prof_linear > 0
            star_prof[ok] = np.log10(star_prof_linear[ok])

            plt.figure(figsize=(6, 4))

            plt.plot(rs, star_prof, marker="o")

            plt.xlabel("R [kpc]")
            plt.ylabel(
                r"$\log_{10}\langle \Sigma_{\star} \rangle$ [Msun/kpc$^2$]"
            )
            plt.title(f"Subhalo {subID} | {proj} | Stars surface density profile")
            plt.grid(True)
            safe_tight_layout()

            plt.savefig(f"{outdir}/SigmaStarProf_{subID}_{proj}.png", dpi=200, bbox_inches='tight')
            plt.close()

            # ---------------- DM radial profile ----------------

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
            plt.ylabel(
                r"$\log_{10}\langle \Sigma_{DM} \rangle$ [Msun/kpc$^2$]"
            )
            plt.title(f"Subhalo {subID} | {proj} | DM surface density profile")
            plt.grid(True)
            safe_tight_layout()

            plt.savefig(f"{outdir}/SigmaDMProf_{subID}_{proj}.png", dpi=200, bbox_inches='tight')
            plt.close()

            # ------------- SB radial profile: xy only -------------

            mu_prof = None
            SigmaL_prof = None

            if proj == SB_profile_projection:
                rmu, mu_prof, SigmaL_prof = radial_profile_mu_from_SigmaL(
                    SigmaL,
                    x,
                    y,
                    rbins,
                    M_sun_r=4.65,
                    statistic="mean"
                )

                plt.figure(figsize=(6, 4))

                plt.plot(rmu, mu_prof, marker="o")

                plt.gca().invert_yaxis()

                plt.xlabel("R [kpc]")
                plt.ylabel(r"$\mu_r$")
                plt.title(f"Subhalo {subID} | {proj} | SB radial profile")
                plt.grid(True)
                safe_tight_layout()

                plt.savefig(f"{outdir}/muProf_{subID}_{proj}.png", dpi=200, bbox_inches='tight')
                plt.close()

            # ---------------- Shape profiles: annuli ----------------

            rmid, qS, paS = shape_profiles_from_linear_map(
                SigmaStar,
                x,
                y,
                rbins,
                Nmin_pixels=Nmin_pixels,
                cumulative=False,
                q_mask=q_mask_value
            )

            _, qD, paD = shape_profiles_from_linear_map(
                SigmaDM,
                x,
                y,
                rbins,
                Nmin_pixels=Nmin_pixels,
                cumulative=False,
                q_mask=q_mask_value
            )

            # ---------------- Shape profiles: cumulative <r ----------------

            if cumulative_compare:
                rmid_c, qS_c, paS_c = shape_profiles_from_linear_map(
                    SigmaStar,
                    x,
                    y,
                    rbins,
                    Nmin_pixels=Nmin_pixels,
                    cumulative=True,
                    q_mask=q_mask_value
                )

                _, qD_c, paD_c = shape_profiles_from_linear_map(
                    SigmaDM,
                    x,
                    y,
                    rbins,
                    Nmin_pixels=Nmin_pixels,
                    cumulative=True,
                    q_mask=q_mask_value
                )

            # ---------------- q compare ----------------

            plt.figure(figsize=(6, 4))

            plt.plot(rmid, qS, marker="o", label="Stars annuli")
            plt.plot(rmid, qD, marker="s", label="DM annuli")

            if cumulative_compare:
                plt.plot(rmid_c, qS_c, ls="--", label="Stars <r")
                plt.plot(rmid_c, qD_c, ls="--", label="DM <r")

            plt.xlabel("R [kpc]")
            plt.ylabel("q=b/a")
            plt.title(f"Subhalo {subID} | {proj} | q(R): Stars vs DM")
            plt.grid(True)
            plt.legend()
            safe_tight_layout()

            plt.savefig(f"{outdir}/qCompare_{subID}_{proj}.png", dpi=200, bbox_inches='tight')
            plt.close()

            # ---------------- PA compare ----------------

            paS_plot = clean_pa_for_plot(paS)
            paD_plot = clean_pa_for_plot(paD)

            plt.figure(figsize=(6, 4))

            plt.plot(rmid, paS_plot, marker="o", markersize=3, lw=1.1, label="Stars annuli")
            plt.plot(rmid, paD_plot, marker="s", markersize=3, lw=1.1, label="DM annuli")

            if cumulative_compare:
                plt.plot(rmid_c, clean_pa_for_plot(paS_c), ls="--", label="Stars <r")
                plt.plot(rmid_c, clean_pa_for_plot(paD_c), ls="--", label="DM <r")

            plt.xlabel("R [kpc]")
            plt.ylabel("PA [deg]")
            plt.title(f"Subhalo {subID} | {proj} | PA(R): Stars vs DM")
            plt.grid(True)
            plt.legend()
            safe_tight_layout()

            plt.savefig(f"{outdir}/paCompare_{subID}_{proj}.png", dpi=200, bbox_inches='tight')
            plt.close()

            # ---------------- Save derived arrays back ----------------

            if save_results_back:
                if "derived" not in gs:
                    dg = gs.create_group("derived")
                else:
                    dg = gs["derived"]

                for name in [
                    "rbins_kpc",
                    "rmid_kpc",
                    "q_stars",
                    "pa_stars",
                    "q_dm",
                    "pa_dm",
                    "rmid_cum_kpc",
                    "q_stars_cum",
                    "pa_stars_cum",
                    "q_dm_cum",
                    "pa_dm_cum",
                    "mu_prof",
                    "SigmaL_prof",
                    "SigmaStar_prof_linear",
                    "SigmaDM_prof_linear"
                ]:
                    if name in dg:
                        del dg[name]

                dg.create_dataset("rbins_kpc", data=rbins)
                dg.create_dataset("rmid_kpc", data=rmid)
                dg.create_dataset("q_stars", data=qS)
                dg.create_dataset("pa_stars", data=paS)
                dg.create_dataset("q_dm", data=qD)
                dg.create_dataset("pa_dm", data=paD)

                if cumulative_compare:
                    dg.create_dataset("rmid_cum_kpc", data=rmid_c)
                    dg.create_dataset("q_stars_cum", data=qS_c)
                    dg.create_dataset("pa_stars_cum", data=paS_c)
                    dg.create_dataset("q_dm_cum", data=qD_c)
                    dg.create_dataset("pa_dm_cum", data=paD_c)

                if proj == SB_profile_projection and mu_prof is not None:
                    dg.create_dataset("mu_prof", data=mu_prof)

                if proj == SB_profile_projection and SigmaL_prof is not None:
                    dg.create_dataset("SigmaL_prof", data=SigmaL_prof)

                dg.create_dataset("SigmaStar_prof_linear", data=star_prof_linear)
                dg.create_dataset("SigmaDM_prof_linear", data=dm_prof_linear)

            del mu, SigmaL, SigmaStar, SigmaDM
            del Sstar, Sdm, logStar, logDM
            gc.collect()

    return True


# ============================================================
# 3D SHAPE AND DENSITY FUNCTIONS
# ============================================================

def compute_3D_shape(coords_kpc, rbins, Nmin=1000):
    
    r = np.linalg.norm(coords_kpc, axis=1)

    rmid = 0.5 * (rbins[:-1] + rbins[1:])

    a_arr = np.full(len(rmid), np.nan)
    b_arr = np.full(len(rmid), np.nan)
    c_arr = np.full(len(rmid), np.nan)

    ba_arr = np.full(len(rmid), np.nan)
    ca_arr = np.full(len(rmid), np.nan)

    N_arr = np.zeros(len(rmid), dtype=int)

    for i in range(len(rmid)):
        mask = r < rbins[i+1]

        n = np.count_nonzero(mask)
        N_arr[i] = n

        if n < Nmin:
            continue

        xyz = coords_kpc[mask]

        x = xyz[:, 0]
        y = xyz[:, 1]
        z = xyz[:, 2]

        I = np.zeros((3, 3), dtype=float)

        I[0, 0] = np.sum(x*x)
        I[1, 1] = np.sum(y*y)
        I[2, 2] = np.sum(z*z)

        I[0, 1] = I[1, 0] = np.sum(x*y)
        I[0, 2] = I[2, 0] = np.sum(x*z)
        I[1, 2] = I[2, 1] = np.sum(y*z)

        eigvals, eigvecs = np.linalg.eigh(I)

        idx = np.argsort(eigvals)[::-1]

        eigvals = eigvals[idx]

        a = np.sqrt(eigvals[0])
        b = np.sqrt(eigvals[1])
        c = np.sqrt(eigvals[2])

        a_arr[i] = a
        b_arr[i] = b
        c_arr[i] = c

        ba_arr[i] = b / a
        ca_arr[i] = c / a

    return rmid, a_arr, b_arr, c_arr, ba_arr, ca_arr, N_arr


def compute_3D_shape_shells(coords_kpc, rbins, Nmin=1000):

    r = np.linalg.norm(coords_kpc, axis=1)

    rmid = 0.5 * (rbins[:-1] + rbins[1:])

    a_arr = np.full(len(rmid), np.nan)
    b_arr = np.full(len(rmid), np.nan)
    c_arr = np.full(len(rmid), np.nan)

    ba_arr = np.full(len(rmid), np.nan)
    ca_arr = np.full(len(rmid), np.nan)

    N_arr = np.zeros(len(rmid), dtype=int)

    for i in range(len(rmid)):
        mask = (r >= rbins[i]) & (r < rbins[i+1])

        n = np.count_nonzero(mask)
        N_arr[i] = n

        if n < Nmin:
            continue

        xyz = coords_kpc[mask]

        x = xyz[:, 0]
        y = xyz[:, 1]
        z = xyz[:, 2]

        I = np.zeros((3, 3), dtype=float)

        I[0, 0] = np.sum(x*x)
        I[1, 1] = np.sum(y*y)
        I[2, 2] = np.sum(z*z)

        I[0, 1] = I[1, 0] = np.sum(x*y)
        I[0, 2] = I[2, 0] = np.sum(x*z)
        I[1, 2] = I[2, 1] = np.sum(y*z)

        eigvals, eigvecs = np.linalg.eigh(I)

        idx = np.argsort(eigvals)[::-1]

        eigvals = eigvals[idx]

        a = np.sqrt(eigvals[0])
        b = np.sqrt(eigvals[1])
        c = np.sqrt(eigvals[2])

        a_arr[i] = a
        b_arr[i] = b
        c_arr[i] = c

        ba_arr[i] = b / a
        ca_arr[i] = c / a

    return rmid, a_arr, b_arr, c_arr, ba_arr, ca_arr, N_arr


def compute_rho_spherical(coords_kpc, mpart_msun, rbins):
   

    r = np.linalg.norm(coords_kpc, axis=1)

    rmid = 0.5 * (rbins[:-1] + rbins[1:])

    rho = np.full(len(rmid), np.nan)

    Nshell = np.zeros(len(rmid), dtype=int)

    for i in range(len(rmid)):
        m = (r >= rbins[i]) & (r < rbins[i+1])

        n = np.count_nonzero(m)

        Nshell[i] = n

        if n == 0:
            continue

        mass = n * mpart_msun

        vol = (4.0/3.0) * np.pi * (rbins[i+1]**3 - rbins[i]**3)

        rho[i] = mass / vol

    return rmid, rho, Nshell


def principal_axes_within_R(coords_kpc, R_kpc, Nmin=2000):
   

    r = np.linalg.norm(coords_kpc, axis=1)

    m = r < R_kpc

    n = np.count_nonzero(m)

    if n < Nmin:
        raise RuntimeError(f"Too few DM particles within R={R_kpc} kpc: N={n}")

    xyz = coords_kpc[m]

    x = xyz[:, 0]
    y = xyz[:, 1]
    z = xyz[:, 2]

    I = np.zeros((3, 3), dtype=float)

    I[0, 0] = np.sum(x*x)
    I[1, 1] = np.sum(y*y)
    I[2, 2] = np.sum(z*z)

    I[0, 1] = I[1, 0] = np.sum(x*y)
    I[0, 2] = I[2, 0] = np.sum(x*z)
    I[1, 2] = I[2, 1] = np.sum(y*z)

    eigvals, eigvecs = np.linalg.eigh(I)

    idx = np.argsort(eigvals)[::-1]

    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]

    ea = eigvecs[:, 0]
    eb = eigvecs[:, 1]
    ec = eigvecs[:, 2]

    ea = ea / np.linalg.norm(ea)
    eb = eb / np.linalg.norm(eb)
    ec = ec / np.linalg.norm(ec)

    # ensure right-handed basis
    if np.dot(np.cross(ea, eb), ec) < 0:
        ec = -ec

    return ea, eb, ec, eigvals


def add_dm_ac_projection_to_h5(
    dm_h5,
    coords_dm_kpc,
    hsml_kpc,
    m_dm_msun,
    R200c_kpc,
    extent_kpc,
    nbins=512,
    dm_sph_smooth=True
):
    """
    Optional a-c projection.

    This function is kept for reference, but it is not called in run_analysis().
    """

    ea, eb, ec, eigvals = principal_axes_within_R(
        coords_dm_kpc,
        R200c_kpc
    )

    basis = np.column_stack([ea, eb, ec])

    coords_abc = coords_dm_kpc @ basis

    coords2d_ac = coords_abc[:, [0, 2]]

    if dm_sph_smooth:
        SigmaAC, xcent, ycent = make_DM_sigma_fast_sph(
            coords2d_ac,
            hsml_kpc,
            m_dm_msun,
            extent_kpc,
            nbins
        )
    else:
        SigmaAC, xcent, ycent, _ = make_DM_sigma_hist_counts(
            coords2d_ac,
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
# MAIN HYDRO PIPELINE
# ============================================================

def run_analysis(
    subhaloID,
    snapNum,
    basePath,
    R200c_kpc,
    outdir="outputs_one",
    extent_factor=0.5,
    nbins=512,
    **kwargs
):
    os.makedirs(outdir, exist_ok=True)

    stars_h5 = os.path.join(outdir, f"stars_sub{subhaloID}.hdf5")
    dm_h5 = os.path.join(outdir, f"dm_sub{subhaloID}.hdf5")

    # ============================================================
    # 1) STAR PROJECTIONS
    # ============================================================

    with Timer("STAR projections"):
        run_star_projections(
            basePath=basePath,
            snapNum=snapNum,
            subhaloID=subhaloID,
            R200c_kpc=R200c_kpc,
            out_h5=stars_h5,
            extent_factor=extent_factor,
            nbins=nbins,
            r_band_index=kwargs.get("r_band_index", 4),
            M_sun_r=kwargs.get("M_sun_r", 4.65),
            Rcut_L_kpc=kwargs.get("Rcut_L_kpc", 20.0),
        )

    # Load stellar angular momentum for face-on and edge-on projections.
    with h5py.File(stars_h5, "r") as f:
        Lhat = f["meta"]["Lhat_stars"][:]

    R_face = rotation_matrix_from_vectors(
        Lhat,
        np.array([0.0, 0.0, 1.0])
    )

    R_edge = np.array([
        [1, 0, 0],
        [0, 0, -1],
        [0, 1, 0]
    ])

    # ============================================================
    # 2) LOAD DM ONCE ONLY
    # ============================================================

    with Timer("DM: load header/subhalo"):
        header = il.groupcat.loadHeader(basePath, snapNum)

        h = header["HubbleParam"]
        z = header["Redshift"]
        a = 1.0 / (1.0 + z)

        boxsize_kpc = header["BoxSize"] * a / h

        sub = il.groupcat.loadSingle(
            basePath,
            snapNum,
            subhaloID=subhaloID
        )

        subpos_kpc = sub["SubhaloPos"] * a / h

        use_fof = kwargs.get("use_fof", False)

    with Timer("DM: load particles ONCE"):
        dm_fields = ["Coordinates", "SubfindHsml"]

        if use_fof:
            groupnum = int(sub["SubhaloGrNr"])

            dm = il.snapshot.loadHalo(
                basePath,
                snapNum,
                groupnum,
                "dm",
                fields=dm_fields
            )

        else:
            dm = il.snapshot.loadSubhalo(
                basePath,
                snapNum,
                subhaloID,
                "dm",
                fields=dm_fields
            )

    with Timer("DM: center/wrap coords"):
        coords_ckpch = dm["Coordinates"]
        hsml_ckpch = dm["SubfindHsml"]

        coords_dm = coords_ckpch * a / h - subpos_kpc
        coords_dm = wrap_relative_periodic(coords_dm, boxsize_kpc)

        hsml_kpc = hsml_ckpch * a / h

    flag(f"DM loaded once: N = {coords_dm.shape[0]:,}")

    # DM particle mass
    try:
        m_dm_code = header["MassTable"][1]
    except KeyError:
        with h5py.File(
            f"{basePath}/snapdir_{snapNum:03d}/snap_{snapNum:03d}.0.hdf5",
            "r"
        ) as f0:
            m_dm_code = f0["Header"].attrs["MassTable"][1]

    m_dm_msun = m_dm_code * 1e10 / h

    # ============================================================
    # 3) COMPUTE 3D SHAPE: CUMULATIVE + SHELLS
    # ============================================================

    with Timer("DM: 3D shape cumulative + shells"):
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

    # ============================================================
    # 3.5) COMPUTE rho_DM(r)
    # ============================================================

    with Timer("DM: rho profile"):
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

    # ============================================================
    # 3.6) SAVE 3D RESULTS
    # ============================================================

    with Timer("DM: save 3D results"):
        with h5py.File(dm_h5, "w") as f:

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

            g3d.attrs["use_fof"] = bool(use_fof)
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

            g3s.attrs["use_fof"] = bool(use_fof)
            g3s.attrs["shape_type"] = "shell"

            # spherical density profile
            gr = f.create_group("rho3d")

            gr.create_dataset("rbins_kpc", data=rbins_rho)
            gr.create_dataset("rmid_kpc", data=rmid_rho)
            gr.create_dataset("rho_msun_kpc3", data=rho_dm)
            gr.create_dataset("N_shell", data=Nshell_dm)

            gr.attrs["use_fof"] = bool(use_fof)

    # ============================================================
    # 4) DM SPH PROJECTIONS FROM SAME LOADED ARRAYS
    # ============================================================

    with Timer("DM: SPH projections from loaded arrays"):
        run_dm_projections_from_loaded(
            coords_dm_kpc=coords_dm,
            hsml_kpc=hsml_kpc,
            m_dm_msun=m_dm_msun,
            R200c_kpc=R200c_kpc,
            out_h5=dm_h5,
            extent_factor=extent_factor,
            nbins=nbins,
            R_face=R_face,
            R_edge=R_edge,
            Lhat_stars=Lhat,
            subhaloID=subhaloID,
            snapNum=snapNum,
            use_fof=use_fof,
            periodic=True
        )

    # ============================================================
    # 4.5) OPTIONAL DM a-c PROJECTION
    # ============================================================
    # 
    # with Timer("DM: a-c projection from loaded arrays"):
    #     extent_kpc = extent_factor * R200c_kpc
    #
    #     add_dm_ac_projection_to_h5(
    #         dm_h5=dm_h5,
    #         coords_dm_kpc=coords_dm,
    #         hsml_kpc=hsml_kpc,
    #         m_dm_msun=m_dm_msun,
    #         R200c_kpc=R200c_kpc,
    #         extent_kpc=extent_kpc,
    #         nbins=nbins,
    #         dm_sph_smooth=True
    #     )

    # ============================================================
    # MEMORY CLEANUP BEFORE ANALYSIS/PLOTTING
    # ============================================================

    del dm
    del coords_ckpch
    del hsml_ckpch
    del coords_dm
    del hsml_kpc

    gc.collect()

    # ============================================================
    # 5) ANALYSIS + COMPARISON
    # ============================================================

    with Timer("analysis and plotting"):
        run_analysis_and_compare(
            stars_h5=stars_h5,
            dm_h5=dm_h5,
            outdir=outdir,
            rbinstep_kpc=kwargs.get("rbinstep_kpc", 5.0),
            vmin_mu=kwargs.get("vmin_mu", 18),
            vmax_mu=kwargs.get("vmax_mu", 32),
            vmin_logDM=kwargs.get("vmin_logDM", 4),
            vmax_logDM=kwargs.get("vmax_logDM", 8),
            Nmin_pixels=kwargs.get("Nmin_pixels", 300),
            save_results_back=True
        )

    return {
        "stars_h5": stars_h5,
        "dm_h5": dm_h5,
        "outdir": outdir
    }


# ============================================================
# MULTIPROCESSING RUN BLOCK
# ============================================================

def run_analysis_star(
    subhaloID,
    snapNum,
    basePath,
    R200c_kpc,
    outdir,
    extent_factor,
    nbins
):
    return run_analysis(
        subhaloID=subhaloID,
        snapNum=snapNum,
        basePath=basePath,
        R200c_kpc=R200c_kpc,
        outdir=outdir,
        extent_factor=extent_factor,
        nbins=nbins,
        use_fof=True,
        rbinstep_kpc=5.0,
        Nmin_pixels=300,
        dm_sph_smooth=True
    )


if __name__ == "__main__":

    jobs = [
    # subhaloID, snapNum, basePath, R200c_kpc, outdir, extent_factor, nbins

    # logM200c ~ 12.5
    #(435752, snapNum, basePath, 291.565674,
    ## "outputs_435752", 1.0, 512),

    # logM200c ~ 13.0
    #(329508, snapNum, basePath, 466.948669,
    # "outputs_329508", 1.0, 512),

    # logM200c ~ 13.5
    #(198182, snapNum, basePath, 680.303406,
     #"outputs_198182", 1.0, 512),

    # logM200c ~ 14.0
    (63864, snapNum, basePath, 959.041809,
     "outputs_63864", 1.0, 512),
    ]

    # DELETE OLD OUTPUTS FIRST
    for job in jobs:
        outdir = job[4]
        shutil.rmtree(outdir, ignore_errors=True)

    nproc = 1

    print("Starting multiprocessing now...")
    print(f"Number of jobs = {len(jobs)}")
    print(f"Number of processes = {nproc}")
    sys.stdout.flush()

    with Pool(processes=nproc, maxtasksperchild=1) as pool:
        results = pool.starmap(run_analysis_star, jobs)

    print("Finished all jobs.")
    print(results)
