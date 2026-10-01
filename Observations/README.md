# HSC Observational Analysis

This directory contains the observational-analysis scripts used for the
Hyper Suprime-Cam component of my Master's thesis.

The observational analysis measures faint stellar-light distributions around
the final HSC galaxy sample, estimates empirical low-surface-brightness limits,
and determines the radial range over which the observed profiles remain
reliable.

## Observational Data

The analysis uses imaging and catalogue products from the
**Hyper Suprime-Cam Subaru Strategic Program (HSC-SSP) Public Data Release 3
(PDR3)**.

The final science analysis uses:

- HSC PDR3 \(r\)-band imaging
- approximately \(10'\times10'\) science cutouts
- HSC pixel scale of approximately \(0.168''\) per pixel
- AB photometric zero point \(ZP=27\)

The original HSC FITS cutouts and other large survey products are not included
in this repository.

## Final HSC Sample

The final observational science sample contains eleven independent galaxies:

- HSC-22
- HSC-70
- HSC-75
- HSC-140
- HSC-145
- HSC-201
- HSC-250
- HSC-284
- HSC-332
- HSC-356
- HSC-454

HSC-608 was identified as a duplicate observation of HSC-454 and is therefore
not treated as an independent science galaxy.

All eleven final galaxies lie in the HSC Deep layer.

## NoiseChisel Processing and Masking

The HSC images were processed using
**GNU Astronomy Utilities (Gnuastro) / NoiseChisel version 0.24**.

NoiseChisel was used for:

- astronomical-source detection
- local sky estimation
- construction of sky-subtracted `INPUT-NO-SKY` images
- creation of detection maps used in contaminant masking

The target galaxy was protected from masking using a galaxy-centred elliptical
region. Detected sources outside this protected region were treated as
contaminants.

## Surface-Brightness Profiles

Radial surface-brightness profiles were measured using concentric elliptical
annuli.

For each galaxy:

- the centre was fixed
- the axis ratio and orientation were determined from the central galaxy light
- annuli had a fixed semi-major-axis width of \(5''\)
- masked and non-finite pixels were excluded
- iterative sigma clipping was applied
- the sigma-clipped mean was used as the primary profile estimator

Flux measurements were converted to \(r\)-band surface brightness using the
HSC pixel scale and \(ZP=27\).

## Conversion to Physical Radius

Angular radii were converted to projected physical radii in kiloparsecs using
the adopted galaxy redshifts and angular-diameter distances.

The final profile tables retain both:

- angular radius in arcseconds
- physical radius in kiloparsecs

## Empirical Surface-Brightness Limit

The practical low-surface-brightness depth was estimated independently for
each galaxy using random blank-sky apertures.

The final empirical measurement uses:

- \(5''\times5''\) apertures
- \(25~\mathrm{arcsec}^2\) aperture area
- 1000 accepted apertures per field
- rejection of apertures with more than 25 per cent masked pixels
- a \(3\sigma\) surface-brightness threshold

The resulting empirical limits for the final sample span approximately

\[
28.25 \lesssim \mu_{\rm lim,emp}
\lesssim 29.18~\mathrm{mag\,arcsec^{-2}}.
\]

## Reliable Radius

For each galaxy, the reliable radial extent was defined from the first outward
crossing of the measured surface-brightness profile through the empirical
surface-brightness limit.

The crossing radius was estimated by interpolation between the last profile
point brighter than the limit and the first point fainter than the limit.

The resulting reliable radii span approximately

\[
12.5 \text{--} 69.4~\mathrm{kpc}.
\]

This radius is used as an observational reliability threshold rather than as a
physical edge of the stellar distribution.

## Main Scripts

### `Noisechisel_mask_statistics.py`

Processes NoiseChisel-related masking and background information used in the
observational analysis.

### `all11_mask_check_allellipse.py`

Produces mask and elliptical-annulus diagnostic checks for the final HSC
sample.

### `box5x5_final_for_example_galaxy70.py`

Implements the \(5''\times5''\) random-aperture background measurement used to
estimate the empirical surface-brightness depth.

### `final11_master_table.py`

Constructs the final observational master table by combining the derived HSC
measurements.

### `final_11_profiles_kpc.py`

Processes the final surface-brightness profiles and converts their radial
coordinates to physical kiloparsecs.

### `table_with_redshift.py`

Combines the observational measurements with the adopted galaxy redshift
information.

## Validation

Additional validation tests performed during the thesis include:

- comparison of nominal and empirical surface-brightness limits
- tests of partially masked random apertures
- tests of aperture-size dependence
- comparison of formal annular uncertainties with empirical background
  fluctuations
- inspection of masking and elliptical-annulus geometry for the final sample

## Notes on Reproducibility

Some scripts require local paths to the original HSC FITS images and
intermediate analysis products. These paths may need to be changed before the
scripts are run on another system.

The repository does not include:

- original HSC FITS cutouts
- large intermediate NoiseChisel products
- large derived image products
- external survey catalogues

The required survey data should be obtained directly from the HSC-SSP public
data release.

## Related Thesis Chapters

The observational workflow is described primarily in:

- **Data, Simulations, and Sample Selection**
- **Observational Methodology**
- **Method Validation and Systematic Uncertainties**
- **Observational Results**
- **Connecting Stellar Light to Dark-Matter Halo Shape**
