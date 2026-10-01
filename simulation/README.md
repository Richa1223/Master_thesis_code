# TNG50 Simulation Analysis

This directory contains the simulation-analysis scripts used for the TNG50
component of my Master's thesis.

The simulation analysis investigates the relation between the projected stellar
distribution of central galaxies and the structure of their surrounding
dark-matter haloes. It also includes measurements of intrinsic three-dimensional
dark-matter halo shape and matched comparisons between the full-physics TNG50
simulation and its dark-matter-only counterpart.

## Simulation Data

The analysis uses:

- **TNG50-1** — highest-resolution full-physics realization of TNG50
- **TNG50-1-Dark** — corresponding dark-matter-only realization
- **Snapshot 99** — corresponding to \(z=0\)

The original IllustrisTNG snapshot and group-catalogue files are not included in
this repository.

The scripts therefore require access to a local copy of the relevant TNG50 data
or corresponding downloaded cutouts/catalogues.

## Simulated Galaxy Sample

The main TNG50 population consists of central galaxies selected from the
friends-of-friends (FoF) halo catalogue at snapshot 99.

The sample is divided into four bins in

\[
\log_{10}(M_{200c}/M_\odot)
\]

with nominal bin centres:

- 11.5
- 12.0
- 12.5
- 13.0

The corresponding 0.5-dex intervals contain:

| Nominal halo-mass bin | Number of central galaxies |
|---|---:|
| 11.5 | 645 |
| 12.0 | 236 |
| 12.5 | 89 |
| 13.0 | 21 |
| **Total** | **991** |

The central galaxy of each FoF halo is identified through
`GroupFirstSub`.

## Particle Definitions

The stellar and dark-matter components are defined differently in the main
analysis:

- **Stars:** stellar particles gravitationally associated with the central
  Subfind subhalo
- **Dark matter:** the complete dark-matter particle population of the
  corresponding FoF host halo

Both components are centred on the position of the central Subfind galaxy.

Coordinates, particle masses, smoothing lengths, and other TNG catalogue
quantities are converted from their native IllustrisTNG units before analysis.

## Projection Geometry

Each simulated galaxy is analysed in five viewing directions:

- `xy`
- `xz`
- `yz`
- face-on
- edge-on

The face-on direction is defined using the angular-momentum axis of the central
stellar component.

The same viewing geometry is applied to both the stellar and dark-matter
components so that their projected shapes can be compared consistently.

## Projected Maps

The particle distributions are deposited onto two-dimensional maps using an
adaptive SPH-style projection.

The main projected quantities are:

- stellar mass surface density, \(\Sigma_\star\)
- dark-matter surface density, \(\Sigma_{\rm DM}\)
- stellar luminosity surface density
- simulated stellar surface brightness

The production maps use a \(512\times512\) grid extending from
\(-R_{200c}\) to \(+R_{200c}\) in both projected directions.

For the main TNG50 population analysis, stellar surface brightness is measured
in the synthetic SDSS \(g\) band using index 4 of
`GFM_StellarPhotometrics`.

The later direct HSC--TNG comparison uses separate \(r\)-band maps.

## Projected Shape Measurements

Projected stellar and dark-matter shapes are measured using weighted
two-dimensional second moments in concentric circular annuli.

The projected axis ratio is

\[
q = \frac{b}{a}
  = \sqrt{\frac{\lambda_{\rm min}}{\lambda_{\rm max}}},
\]

where \(\lambda_{\rm min}\) and \(\lambda_{\rm max}\) are the eigenvalues of
the projected second-moment tensor.

The main population analysis uses:

- annular radial measurements
- \(5\,{\rm kpc}\) radial bins
- stellar-mass weighting for \(q_\star\)
- dark-matter-mass weighting for \(q_{\rm DM}\)
- a minimum of 300 valid map pixels per annulus

Both annular and cumulative profiles were explored during method development,
but the final population analysis uses the annular estimator.

## Radial Profiles

The simulation analysis also measures radial profiles of:

- stellar surface density
- dark-matter surface density
- simulated stellar surface brightness
- projected stellar axis ratio
- projected dark-matter axis ratio

Profiles from different haloes are compared using the normalized radial
coordinate

\[
R/R_{200c}.
\]

Population statistics are calculated separately for each halo-mass bin and each
projection using the median and 16th--84th percentile range.

## Characteristic Radii

Several characteristic measurements are extracted from the projected profiles:

- \(q_\star\) and \(q_{\rm DM}\) at \(0.1R_{200c}\)
- \(q_\star\) and \(q_{\rm DM}\) at \(50\,{\rm kpc}\)
- \(q_\star\) and \(q_{\rm DM}\) at the radius where
  \(\mu_g = 30\,{\rm mag\,arcsec^{-2}}\)
- \(R_{\mu_g=30}/R_{200c}\)

These measurements are used to compare stellar and dark-matter flattening at
common physical and surface-brightness-defined scales.

## Intrinsic Three-Dimensional Dark-Matter Shape

The intrinsic dark-matter halo structure is measured directly from the
three-dimensional FoF dark-matter particle distribution.

The analysis calculates:

- cumulative \(b/a\)
- cumulative \(c/a\)
- shell-based \(b/a\)
- shell-based \(c/a\)

using the eigenvalues of the three-dimensional second-moment tensor.

No reduced \(1/r^2\) weighting or iterative ellipsoidal re-selection is used.

A minimum of 1000 dark-matter particles is required for an intrinsic shape
measurement.

## TNG50 / TNG50-Dark Comparison

Matched TNG50-1 and TNG50-1-Dark systems are used to examine the effect of
baryonic galaxy formation on dark-matter halo structure.

The matched analysis compares:

- cumulative intrinsic dark-matter axis-ratio profiles
- shell-based intrinsic dark-matter axis-ratio profiles
- spherically averaged dark-matter density profiles

The full-physics and dark-matter-only systems are analysed using the same shape
definitions and radial measurements.

## Main Scripts

The directory currently contains scripts including:

### `select_mass_bin_halos.py`

Selects TNG50 central galaxies within the adopted halo-mass bins.

### `check_mass_bin_counts.py`

Checks the number of central galaxies contained in each final halo-mass bin.

### `example_for_one_mass_bin_compare_projected_medians_11p5.py`

Produces projected population summaries for an example halo-mass bin and is
used to compare stellar and dark-matter median projected profiles.

### `example_for_one_mass_bin_extract_fixed_radius_metrics_12.py`

Extracts characteristic projected-shape measurements at fixed physical,
normalized, and surface-brightness-defined radii.

### `final_compare_mass_bins.py`

Compares the main projected stellar and dark-matter measurements across the
four halo-mass bins.

### `final_paired_qdiff_massbins.py`

Calculates the halo-by-halo paired projected-shape difference

\[
\Delta q_i = q_{\star,i} - q_{{\rm DM},i}
\]

before computing population statistics.

### `plot_fixed_radius_example_mass_bin_13.py`

Produces diagnostic plots of the characteristic-radius measurements for the
high-mass sample.

### `compare_hydro_dm.py`

Compares matched full-physics and dark-matter-only halo properties.

### `stack_hydro_dm_shapes.py`

Combines the intrinsic-shape measurements of matched TNG50-1 and
TNG50-1-Dark systems for comparison.

## Connection to the HSC Analysis

Outputs from this directory are used by the scripts in the
`Comparison/` directory for the direct HSC--TNG analysis.

The direct comparison uses stellar-mass-selected TNG50 systems and synthetic
\(r\)-band stellar-light maps so that the simulated photometry is compared with
the HSC \(r\)-band observations in the same photometric band.

## Notes on Reproducibility

The scripts were developed for the analysis environment used during the thesis
and may contain paths that need to be changed before being run on another
system.

The repository does not include:

- raw TNG50 snapshots
- raw TNG50-Dark snapshots
- large particle cutouts
- large intermediate map products
- API credentials or authentication information

Users should obtain the required simulation data directly from the
IllustrisTNG public data release.

## Related Thesis Chapters

The simulation workflow is described primarily in:

- **Data, Simulations, and Sample Selection**
- **Simulation Methodology**
- **Method Validation and Systematic Uncertainties**
- **Simulation Results**
- **Connecting Stellar Light to Dark-Matter Halo Shape**
