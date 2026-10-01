import csv

def read_by_galaxy(path):
    out = {}
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            out[row["galaxy"].strip()] = row
    return out

master = read_by_galaxy("final12_master_observational_table.csv")
kpc = read_by_galaxy("final12_kpc_scale.csv")

order = ["454","284","22","250","356","332","201","140","145","75","70"]

with open("final12_master_observational_table_with_redshift.csv", "w", newline="") as f:
    base_cols = list(next(iter(master.values())).keys())

    extra_cols = [
        "ra_deg",
        "dec_deg",
        "redshift",
        "kpc_per_arcsec",
        "redshift_type",
        "redshift_source",
        "object_name",
        "independent_sample",
        "redshift_notes"
    ]

    writer = csv.DictWriter(f, fieldnames=base_cols + extra_cols)
    writer.writeheader()

    for g in order:
        row = dict(master[g])
        kk = kpc[g]

        row["ra_deg"] = kk.get("ra_deg", "")
        row["dec_deg"] = kk.get("dec_deg", "")
        row["redshift"] = kk.get("redshift", "")
        row["kpc_per_arcsec"] = kk.get("kpc_per_arcsec", "")
        row["redshift_type"] = kk.get("redshift_type", "")
        row["redshift_source"] = kk.get("redshift_source", "")
        row["object_name"] = kk.get("object_name", "")
        row["independent_sample"] = kk.get("independent_sample", "")
        row["redshift_notes"] = kk.get("notes", "")

        writer.writerow(row)

print("Saved: final12_master_observational_table_with_redshift.csv")
print()
print(open("final12_master_observational_table_with_redshift.csv").read())
