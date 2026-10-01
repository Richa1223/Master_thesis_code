import os
import numpy as np
import pandas as pd
import h5py

# ============================================================
# CONFIG
# ============================================================

data_root = "mass_bin_12_spinproj"
sample_csv = "TNG_work/stack_sample_logM12_pm0p25_ALL.csv"

out_csv = os.path.join(data_root, "fixed_radius_metrics_logM12_spinproj_ALL.csv")

projs = ["xy", "xz", "yz", "faceon", "edgeon"]

target_Rfrac = 0.1
target_Rkpc = 50.0
target_mu = 30.0

M_sun_r = 4.65


# ============================================================
# CONVERSIONS
# ============================================================

def SigmaL_to_mu(Sigma_L_kpc2, M_sun_r=4.65):
    Sigma_L_kpc2 = np.asarray(Sigma_L_kpc2, dtype=float)
    I_pc2 = Sigma_L_kpc2 / 1e6

    mu = np.full_like(I_pc2, np.nan, dtype=float)
    ok = I_pc2 > 0

    mu[ok] = M_sun_r + 21.572 - 2.5 * np.log10(I_pc2[ok])
    return mu


def mu_to_SigmaL(mu, M_sun_r=4.65):
    mu = np.asarray(mu, dtype=float)
    SigmaL = np.full_like(mu, np.nan, dtype=float)

    ok = np.isfinite(mu)
    I_pc2 = 10.0 ** (-0.4 * (mu[ok] - M_sun_r - 21.572))
    SigmaL[ok] = I_pc2 * 1e6

    return SigmaL


def interp_value(x, y, x0):
    """
    Interpolate y(x) at one target x0.
    Returns NaN if x0 is outside the valid range.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    good = np.isfinite(x) & np.isfinite(y)

    if np.count_nonzero(good) < 2:
        return np.nan

    xg = x[good]
    yg = y[good]

    order = np.argsort(xg)
    xg = xg[order]
    yg = yg[order]

    if x0 < np.nanmin(xg) or x0 > np.nanmax(xg):
        return np.nan

    return float(np.interp(x0, xg, yg))


def find_radius_at_mu(R_kpc, mu, target_mu=30.0):
   
    R_kpc = np.asarray(R_kpc, dtype=float)
    mu = np.asarray(mu, dtype=float)

    good = np.isfinite(R_kpc) & np.isfinite(mu)

    if np.count_nonzero(good) < 2:
        return np.nan

    R = R_kpc[good]
    m = mu[good]

    order = np.argsort(R)
    R = R[order]
    m = m[order]

    # Look for first outward crossing of target_mu.
    diff = m - target_mu

    for i in range(len(R) - 1):
        if diff[i] == 0:
            return float(R[i])

        if diff[i] * diff[i + 1] < 0:
            return float(np.interp(
                target_mu,
                [m[i], m[i + 1]],
                [R[i], R[i + 1]]
            ))

    return np.nan


def radial_profile_from_map_linear(img, xcent, ycent, rbins):
    img = np.asarray(img, dtype=float)
    xcent = np.asarray(xcent, dtype=float)
    ycent = np.asarray(ycent, dtype=float)

    X, Y = np.meshgrid(xcent, ycent, indexing="ij")
    R = np.sqrt(X**2 + Y**2)

    rmid = 0.5 * (rbins[:-1] + rbins[1:])
    prof = np.full_like(rmid, np.nan, dtype=float)

    for i in range(len(rmid)):
        mask = (
            (R >= rbins[i])
            & (R < rbins[i + 1])
            & np.isfinite(img)
            & (img > 0)
        )

        if np.count_nonzero(mask) > 0:
            prof[i] = np.nanmean(img[mask])

    return rmid, prof


# ============================================================
# HDF5 READING
# ============================================================

def get_stars_h5(subhaloID):
    return os.path.join(
        data_root,
        f"outputs_{subhaloID}",
        f"stars_sub{subhaloID}.hdf5"
    )


def read_projection_data(subhaloID, proj, R200c):
    """
    Reads q_star, q_dm, and stellar surface-brightness profile
    from stars_sub<ID>.hdf5.
    """
    fname = get_stars_h5(subhaloID)

    if not os.path.exists(fname):
        raise FileNotFoundError(fname)

    data = {}

    with h5py.File(fname, "r") as f:
        if proj not in f:
            raise KeyError(f"{proj} not found in {fname}")

        g = f[proj]

        # Read derived profiles if they exist.
        if "derived" in g:
            dg = g["derived"]
            for key in dg.keys():
                data[key] = dg[key][:]

        # Fallback maps for surface brightness.
        for key in [
            "x_centers_kpc",
            "y_centers_kpc",
            "SigmaL_Lsun_kpc2",
            "mu_r_mag_arcsec2",
        ]:
            if key in g:
                data[key] = g[key][:]

    # Build stellar surface-brightness profile.
    if "rmid_kpc" in data and "SigmaL_prof_linear" in data:
        R_mu = data["rmid_kpc"]
        mu_prof = SigmaL_to_mu(data["SigmaL_prof_linear"], M_sun_r=M_sun_r)

    elif "rmid_kpc" in data and "SigmaL_prof" in data:
        R_mu = data["rmid_kpc"]
        mu_prof = SigmaL_to_mu(data["SigmaL_prof"], M_sun_r=M_sun_r)

    elif "rmid_kpc" in data and "mu_prof" in data:
        R_mu = data["rmid_kpc"]
        mu_prof = data["mu_prof"]

    elif (
        "x_centers_kpc" in data
        and "y_centers_kpc" in data
        and "SigmaL_Lsun_kpc2" in data
    ):
        rbins = np.linspace(0.0, R200c, 51)
        R_mu, SigmaL_prof = radial_profile_from_map_linear(
            data["SigmaL_Lsun_kpc2"],
            data["x_centers_kpc"],
            data["y_centers_kpc"],
            rbins
        )
        mu_prof = SigmaL_to_mu(SigmaL_prof, M_sun_r=M_sun_r)

    elif (
        "x_centers_kpc" in data
        and "y_centers_kpc" in data
        and "mu_r_mag_arcsec2" in data
    ):
        rbins = np.linspace(0.0, R200c, 51)
        SigmaL_map = mu_to_SigmaL(data["mu_r_mag_arcsec2"], M_sun_r=M_sun_r)
        R_mu, SigmaL_prof = radial_profile_from_map_linear(
            SigmaL_map,
            data["x_centers_kpc"],
            data["y_centers_kpc"],
            rbins
        )
        mu_prof = SigmaL_to_mu(SigmaL_prof, M_sun_r=M_sun_r)

    else:
        R_mu = np.array([])
        mu_prof = np.array([])

    data["R_mu_kpc"] = R_mu
    data["mu_prof"] = mu_prof

    return data


# ============================================================
# MAIN
# ============================================================

sample = pd.read_csv(sample_csv)

rows = []

for _, row in sample.iterrows():

    sid = int(row["hydro_subhaloID"])
    R200c = float(row["R200c_kpc"])
    M200c = float(row["M200c_Msun"])
    logM = float(row["logM200c"])

    print(f"Processing {sid}")

    out = {
        "hydro_subhaloID": sid,
        "GroupID": int(row["GroupID"]),
        "logM200c": logM,
        "M200c_Msun": M200c,
        "R200c_kpc": R200c,
        "GroupNsubs": int(row["GroupNsubs"]),
    }

    for proj in projs:

        try:
            data = read_projection_data(sid, proj, R200c)
        except Exception as e:
            print(f"  {proj}: failed: {repr(e)}")
            continue

        if "rmid_kpc" not in data:
            print(f"  {proj}: missing rmid_kpc")
            continue

        R = np.asarray(data["rmid_kpc"], dtype=float)

        R_01 = target_Rfrac * R200c
        R_50 = target_Rkpc

        out[f"{proj}_R_0p1R200c_kpc"] = R_01
        out[f"{proj}_R_50kpc"] = R_50

        for qname in ["q_stars", "q_dm"]:
            if qname not in data:
                out[f"{proj}_{qname}_0p1R200c"] = np.nan
                out[f"{proj}_{qname}_50kpc"] = np.nan
                continue

            q = data[qname]

            out[f"{proj}_{qname}_0p1R200c"] = interp_value(R, q, R_01)
            out[f"{proj}_{qname}_50kpc"] = interp_value(R, q, R_50)

        # Radius where mu_g = 30
        R_mu = data.get("R_mu_kpc", np.array([]))
        mu_prof = data.get("mu_prof", np.array([]))

        R_at_mu30 = find_radius_at_mu(
            R_mu,
            mu_prof,
            target_mu=target_mu
        )

        out[f"{proj}_R_mu30_kpc"] = R_at_mu30
        out[f"{proj}_R_mu30_over_R200c"] = (
            R_at_mu30 / R200c if np.isfinite(R_at_mu30) else np.nan
        )

        # q values at R(mu=30)
        for qname in ["q_stars", "q_dm"]:
            if qname in data and np.isfinite(R_at_mu30):
                out[f"{proj}_{qname}_at_mu30"] = interp_value(
                    R,
                    data[qname],
                    R_at_mu30
                )
            else:
                out[f"{proj}_{qname}_at_mu30"] = np.nan

    rows.append(out)

df = pd.DataFrame(rows)

df.to_csv(out_csv, index=False)

print("")
print("Saved:")
print(out_csv)
print("")
print(df)
