import csv
from pathlib import Path

# -----------------------------
# Input files
# -----------------------------
centers_file = Path("final12_centers.csv")
sbl_file = Path("final_observation_sample_sbl_table.csv")
profile_file = Path("final_profiles_all12/final12_profile_summary.csv")
quality_file = Path("final12_profile_quality_notes.csv")

# If quality notes do not exist yet, create them
if not quality_file.exists():
    quality_file.write_text("""galaxy,profile_quality,keep,comment
454,good,yes,smooth profile; reliable until it reaches the empirical SBL
284,good,yes,clean profile; outer points are background-limited
22,caution,yes but flag,central residual and large masked region; outer profile is near noise floor
250,usable,yes,good inner profile; outer points noisy after SBL
356,usable,yes,shorter radial range but acceptable
332,usable/caution,yes but flag,reaches SBL early; outer points noisy
201,caution,yes but flag,spiral structure; not a clean halo case
140,caution,yes but flag,noisy after SBL; weaker candidate
145,usable/good,yes,reasonable profile; limited but stable
75,good,yes,extended profile; good candidate
70,usable/good,yes,smooth profile; outer plateau is background-limited
""")

# -----------------------------
# Helper reader
# -----------------------------
def read_csv_by_galaxy(path):
    data = {}
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            g = row["galaxy"].strip()
            data[g] = {k.strip(): v.strip() for k, v in row.items()}
    return data

centers = read_csv_by_galaxy(centers_file)
sbl = read_csv_by_galaxy(sbl_file)
profiles = read_csv_by_galaxy(profile_file)
quality = read_csv_by_galaxy(quality_file)

# Keep the order from final12_centers.csv
galaxy_order = list(centers.keys())

# -----------------------------
# Output table
# -----------------------------
out = Path("final12_master_observational_table.csv")

columns = [
    "galaxy",
    "x_center",
    "y_center",
    "central_radius_arcsec",
    "visual_class",
    "visual_notes",

    "nominal_noisechisel_sbl_100arcsec2",
    "empirical_5x5_mu_all",
    "empirical_5x5_sigma_all",
    "N_valid",
    "mu_valid",
    "N_corrected",
    "mu_corrected",
    "abs_mu_valid_minus_mu_corrected",
    "sbl_quality_flag",

    "axis_ratio_q",
    "PA_deg",
    "target_keep_arcsec",
    "last_reliable_mean_radius_arcsec",
    "last_reliable_mean_mu",
    "last_reliable_median_radius_arcsec",
    "last_reliable_median_mu",
    "reaches_sbl_mean",
    "reaches_sbl_median",

    "profile_quality",
    "keep",
    "profile_comment"
]

with out.open("w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(columns)

    for g in galaxy_order:
        c = centers.get(g, {})
        s = sbl.get(g, {})
        p = profiles.get(g, {})
        q = quality.get(g, {})

        writer.writerow([
            g,
            c.get("x_center", ""),
            c.get("y_center", ""),
            c.get("r_arcsec", ""),
            c.get("class", ""),
            c.get("notes", ""),

            s.get("nominal_noisechisel_sbl_100arcsec2", ""),
            s.get("empirical_5x5_mu_all", ""),
            s.get("empirical_5x5_sigma_all", ""),
            s.get("N_valid", ""),
            s.get("mu_valid", ""),
            s.get("N_corrected", ""),
            s.get("mu_corrected", ""),
            s.get("abs_mu_valid_minus_mu_corrected", ""),
            s.get("quality_flag", ""),

            p.get("axis_ratio_q", ""),
            p.get("PA_deg", ""),
            p.get("target_keep_arcsec", ""),
            p.get("last_reliable_mean_radius_arcsec", ""),
            p.get("last_reliable_mean_mu", ""),
            p.get("last_reliable_median_radius_arcsec", ""),
            p.get("last_reliable_median_mu", ""),
            p.get("reaches_sbl_mean", ""),
            p.get("reaches_sbl_median", ""),

            q.get("profile_quality", ""),
            q.get("keep", ""),
            q.get("comment", "")
        ])

print("Saved:", out)
print()
print(out.read_text())
