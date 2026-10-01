# Master Thesis Analysis Code

This repository contains the analysis code developed for my Master's thesis at
Ruhr University Bochum.

The thesis investigates whether faint stellar light in the outskirts of
galaxies can provide information about the structure of their surrounding
dark-matter haloes.

The project combines deep observational imaging from the
**Hyper Suprime-Cam Subaru Strategic Program (HSC-SSP)** with galaxies from the
**TNG50 cosmological hydrodynamical simulation** of the IllustrisTNG project.

The analysis is divided into three main components:

1. observational low-surface-brightness measurements using HSC;
2. stellar and dark-matter structure measurements using TNG50;
3. direct comparison between the observed HSC galaxies and
   stellar-mass-selected TNG50 systems.

---

## Repository Structure

### `Observations/`

Contains the scripts used for the HSC observational analysis.

The main tasks include:

- processing HSC PDR3 r-band galaxy images;
- NoiseChisel-based source detection and background processing;
- contaminant masking;
- elliptical-annulus surface-brightness measurements;
- empirical random-aperture surface-brightness limits;
- conversion of angular radii to physical radii;
- construction of the final observational tables.

See [`Observations/README.md`](Observations/README.md) for details.

---

### `simulation/`

Contains the scripts used for the TNG50 simulation analysis.

The main tasks include:

- selection of TNG50 central galaxies;
- halo-mass binning;
- loading stellar and dark-matter particle data;
- coordinate and unit conversion;
- galaxy orientation and projection;
- construction of projected stellar and dark-matter maps;
- projected stellar and dark-matter axis-ratio measurements;
- radial surface-density and surface-brightness profiles;
- population-level stacking;
- intrinsic three-dimensional dark-matter shape measurements;
- matched TNG50 / TNG50-Dark halo comparisons.

See [`simulation/README.md`](simulation/README.md) for details.

---

### `Comparison/`

Contains the scripts used for the direct HSC--TNG50 comparison.

The main tasks include:

- selection of TNG50 analogues based on stellar mass;
- construction of synthetic TNG50 r-band stellar-light maps;
- comparison of HSC and TNG50 surface-brightness profiles;
- comparison of observed and simulated projected stellar shapes;
- comparison of simulated stellar-light and dark-matter shapes.

See [`Comparison/README.md`](Comparison/README.md) for details.

---

## Observational Data

The observational analysis uses data from the

**Hyper Suprime-Cam Subaru Strategic Program Public Data Release 3
(HSC-SSP PDR3).**

The final observational sample contains eleven independent galaxies in the
HSC Deep layer.

The main analysis uses HSC r-band images and empirical
low-surface-brightness limits measured separately for each galaxy field.

The original HSC FITS images and catalogue products are not redistributed in
this repository.

---

## Simulation Data

The simulation analysis uses:

- **TNG50-1**
- **TNG50-1-Dark**
- **snapshot 99 ($z=0$)**

from the IllustrisTNG project.

The main TNG50 population contains 991 central galaxies divided into four
halo-mass bins centred at

$$
\log_{10}(M_{200c}/M_\odot)
=
11.5,\ 12.0,\ 12.5,\ 13.0
$$

The original TNG50 snapshots, group catalogues, and large particle products are
not distributed with this repository.

---

## Main Scientific Measurements

The observational analysis measures:

- r-band surface-brightness profiles;
- empirical limiting surface brightness;
- reliable radial extent of the observed stellar light;
- projected stellar-light shape.

The simulation analysis measures:

- projected stellar-mass shape;
- projected dark-matter shape;
- simulated stellar surface brightness;
- stellar and dark-matter radial profiles;
- intrinsic three-dimensional dark-matter axis ratios;
- differences between full-physics and dark-matter-only haloes.

The final comparison connects the HSC measurements with TNG50 systems selected
by proximity in stellar mass.

---

## Software

The analysis was carried out primarily in Python.

Major packages used include:

- NumPy
- SciPy
- pandas
- Matplotlib
- Astropy
- h5py

The observational low-surface-brightness processing also uses

**GNU Astronomy Utilities (Gnuastro) / NoiseChisel version 0.24.**

Additional TNG50 map construction uses the SPH map-making tools employed in the
simulation-analysis scripts.

---

## Reproducibility

The scripts in this repository correspond to the analysis carried out for the
Master's thesis.

Some scripts contain paths to local data products and may require path changes
before being run on another system.

The repository does **not** include:

- raw HSC FITS images;
- raw TNG50 or TNG50-Dark snapshots;
- large intermediate image products;
- large particle datasets;
- private API keys or authentication information.

Users wishing to reproduce the analysis should obtain the original data from
the corresponding HSC-SSP and IllustrisTNG public data services.

Further details on the individual analysis steps are provided in the README
files inside each directory.

---

## Thesis

**Author:** Richa Shree  
**Institution:** Ruhr University Bochum  
**Year:** 2026

The exact thesis title and final citation information will be added here after
submission.

---

## Citation

If you use material from this repository, please cite the corresponding
Master's thesis.

A full citation will be added after the thesis has been submitted.
