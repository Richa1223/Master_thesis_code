import numpy as np
from astropy.io import fits
from photutils.aperture import RectangularAperture


# ============================================================
# SETTINGS
# ============================================================

image_file = "../nc.fits"

pixel_scale = 0.168              # arcsec per pixel
box_side_arcsec = 5.0           # 10 x 10 arcsec box
box_area_arcsec2 = 25.0

maximum_nan_fraction = 0.25      # keep boxes with <=25% masked
number_of_boxes = 1000
maximum_trials = 200000
random_seed = 1616493518

zeropoint = 27.0
n_sigma = 3.0

# Keep the known Galaxy 70 bathtub region excluded.
use_central_mask = True

galaxy_x_ds9 = 1786.0323
galaxy_y_ds9 = 1861.2717
central_mask_radius_arcsec = 58.806


# ============================================================
# CONVERT SIZES TO PIXELS
# ============================================================

box_side_pixels = box_side_arcsec / pixel_scale

central_mask_radius_pixels = (
    central_mask_radius_arcsec / pixel_scale
)

print(
    "Box side: {:.3f} arcsec = {:.3f} pixels".format(
        box_side_arcsec,
        box_side_pixels
    )
)

print(
    "Box area: {:.1f} arcsec^2".format(
        box_area_arcsec2
    )
)


# ============================================================
# LOAD IMAGE AND NOISECHISEL MASK
# ============================================================

# HDU 1 = INPUT-NO-SKY
image = fits.getdata(
    image_file,
    ext=1
).astype(float)

# HDU 2 = DETECTIONS
detection_mask = fits.getdata(
    image_file,
    ext=2
) != 0

ny, nx = image.shape


# ============================================================
# CREATE CENTRAL BATHTUB MASK
# ============================================================

galaxy_x_numpy = galaxy_x_ds9 - 1.0
galaxy_y_numpy = galaxy_y_ds9 - 1.0

yy, xx = np.ogrid[:ny, :nx]

central_mask = (
    (xx - galaxy_x_numpy) ** 2
    + (yy - galaxy_y_numpy) ** 2
    <= central_mask_radius_pixels ** 2
)


# ============================================================
# PLACE RANDOM 10 x 10 ARCSEC BOXES
# ============================================================

rng = np.random.RandomState(random_seed)

results = []

trials = 0
rejected_central = 0
rejected_above_25_percent = 0

# Half the square width plus a safety margin.
margin = box_side_pixels / 2.0 + 2.0

while (
    len(results) < number_of_boxes
    and trials < maximum_trials
):
    trials += 1

    x = rng.uniform(
        margin,
        nx - margin
    )

    y = rng.uniform(
        margin,
        ny - margin
    )

    aperture = RectangularAperture(
        (x, y),
        w=box_side_pixels,
        h=box_side_pixels,
        theta=0.0
    )

    aperture_mask = aperture.to_mask(
        method="center"
    )

    if isinstance(aperture_mask, (list, tuple)):
        aperture_mask = aperture_mask[0]

    image_cutout = aperture_mask.cutout(
        image,
        fill_value=np.nan
    )

    detection_cutout = aperture_mask.cutout(
        detection_mask.astype(np.uint8),
        fill_value=1
    )

    central_cutout = aperture_mask.cutout(
        central_mask.astype(np.uint8),
        fill_value=1
    )

    if (
        image_cutout is None
        or detection_cutout is None
    ):
        continue

    inside = aperture_mask.data > 0

    # Strictly exclude the known bathtub region.
    if use_central_mask:
        if np.any(central_cutout[inside] != 0):
            rejected_central += 1
            continue

    pixel_values = image_cutout[inside]

    masked_pixels = (
        (detection_cutout[inside] != 0)
        | ~np.isfinite(pixel_values)
    )

    number_total = len(pixel_values)
    number_masked = int(
        np.count_nonzero(masked_pixels)
    )

    nan_fraction = (
        number_masked / float(number_total)
    )

    # Keep only boxes with at most 25% masked pixels.
    if nan_fraction > maximum_nan_fraction:
        rejected_above_25_percent += 1
        continue

    valid_values = pixel_values[
        ~masked_pixels
    ]

    if len(valid_values) == 0:
        continue

    valid_sum = float(
        np.sum(valid_values)
    )

    valid_mean = float(
        np.mean(valid_values)
    )

    # Michael's correction:
    #
    # corrected sum =
    # valid sum + N_masked * mean(valid pixels)
    corrected_sum = (
        valid_sum
        + number_masked * valid_mean
    )

    results.append([
        x + 1.0,
        y + 1.0,
        nan_fraction,
        number_total,
        number_masked,
        valid_sum,
        valid_mean,
        corrected_sum
    ])


if len(results) < number_of_boxes:
    raise RuntimeError(
        "Could not find enough accepted boxes. "
        "Increase maximum_trials."
    )


# ============================================================
# SAVE EACH BOX MEASUREMENT
# ============================================================

results = np.array(results)

output_table = "box5x5_nanfrac_results.txt"

np.savetxt(
    output_table,
    results,
    header=(
        "X_DS9 Y_DS9 NAN_FRAC "
        "N_TOTAL N_MASKED VALID_SUM "
        "VALID_MEAN CORRECTED_SUM"
    )
)


# ============================================================
# DEFINE THE FOUR GROUPS
# ============================================================

nan_fraction = results[:, 2]
corrected_sum = results[:, 7]

all_selector = np.ones(
    len(results),
    dtype=bool
)

valid_selector = (
    nan_fraction == 0
)

corrected_selector = (
    nan_fraction > 0
)

strong_selector = (
    nan_fraction > 0.1
)


# ============================================================
# STANDARD DEVIATION AND SURFACE-BRIGHTNESS LIMIT
# ============================================================

def calculate_group(label, selector):
    values = corrected_sum[selector]
    number = len(values)

    if number < 2:
        return {
            "label": label,
            "number": number,
            "std": np.nan,
            "mu_limit": np.nan
        }

    standard_deviation = np.std(
        values,
        ddof=1
    )

    mu_limit = (
        -2.5
        * np.log10(
            n_sigma
            * standard_deviation
            / box_area_arcsec2
        )
        + zeropoint
    )

    return {
        "label": label,
        "number": number,
        "std": standard_deviation,
        "mu_limit": mu_limit
    }


groups = [
    calculate_group(
        "ALL APERTURES",
        all_selector
    ),
    calculate_group(
        "VALID ONLY: NAN_FRAC = 0",
        valid_selector
    ),
    calculate_group(
        "CORRECTED: NAN_FRAC > 0",
        corrected_selector
    ),
    calculate_group(
        "STRONGLY CORRECTED: NAN_FRAC > 0.1",
        strong_selector
    )
]


# ============================================================
# SAVE SUMMARY
# ============================================================

summary_file = "box5x5_nanfrac_summary.txt"

with open(summary_file, "w") as output:
    output.write(
        "Galaxy 70: 5 x 5 arcsec box experiment\n"
    )
    output.write(
        "========================================\n\n"
    )

    output.write(
        "Box side: {:.3f} arcsec\n".format(
            box_side_arcsec
        )
    )

    output.write(
        "Box side: {:.6f} pixels\n".format(
            box_side_pixels
        )
    )

    output.write(
        "Box area: {:.1f} arcsec^2\n".format(
            box_area_arcsec2
        )
    )

    output.write(
        "Maximum accepted NAN_FRAC: {:.2f}\n".format(
            maximum_nan_fraction
        )
    )

    output.write(
        "Accepted boxes: {}\n".format(
            len(results)
        )
    )

    output.write(
        "Total random trials: {}\n".format(
            trials
        )
    )

    output.write(
        "Rejected by central mask: {}\n".format(
            rejected_central
        )
    )

    output.write(
        "Rejected above NAN_FRAC=0.25: {}\n\n".format(
            rejected_above_25_percent
        )
    )

    output.write(
        "Formula:\n"
    )

    output.write(
        "mu_limit = "
        "-2.5 log10(n * sigma / area) + ZP\n"
    )

    output.write(
        "n = {:.1f}, area = {:.1f}, ZP = {:.1f}\n\n".format(
            n_sigma,
            box_area_arcsec2,
            zeropoint
        )
    )

    for group in groups:
        output.write(
            "{}\n".format(
                group["label"]
            )
        )

        output.write(
            "N = {}\n".format(
                group["number"]
            )
        )

        output.write(
            "Standard deviation = {:.8g}\n".format(
                group["std"]
            )
        )

        output.write(
            "Surface-brightness limit = "
            "{:.8f} mag/arcsec^2\n\n".format(
                group["mu_limit"]
            )
        )


# ============================================================
# PRINT RESULTS
# ============================================================

print("")
print("Finished successfully.")
print("Accepted boxes:", len(results))
print("Total trials:", trials)
print("")

for group in groups:
    print(group["label"])
    print("N =", group["number"])
    print(
        "Standard deviation =",
        group["std"]
    )
    print(
        "Surface-brightness limit =",
        group["mu_limit"],
        "mag/arcsec^2"
    )
    print("")

print("Saved:", output_table)
print("Saved:", summary_file)
