LineRepository
============

This folder contains structured definitions of spectral emission lines used by the sheap fitting engine. Each YAML file describes individual spectral lines used during model construction and spectral region fitting.

## Organization

The lines are grouped into separate YAML files based on their expected behavior:

- `narrow.yaml`: contains lines that are typically observed only with narrow components.
- `broad.yaml`: contains lines that are expected to appear only with broad components (e.g., from the broad-line region).
- `broad_and_narrow.yaml`: includes lines that can exhibit both narrow and broad profiles, depending on the physical conditions of the system.

This separation allows sheap to automatically assign the appropriate modeling components during spectral fitting.

## Line Format

Each line in a YAML file is represented as a dictionary with the following possible fields:

```yaml
- amplitude: 1.0
  center: 4075.954
  element: a2D2
  line_name: a2D2e
  region: fe
```

### Field Descriptions

- `center`: Rest-frame central wavelength in Angstroms. (**Required**)
- `line_name`: Unique identifier for the emission line. (**Required**)
- `amplitude`: Initial guess or template weight for the line (optional).
- `element`: Atomic or ionic species associated with the line (optional).
- `region`: Logical or physical grouping, such as `fe`, `narrow`, etc. (optional).

Only `center` and `line_name` are strictly required by the code. Other fields are included to aid in organizing, plotting, or applying special treatment during modeling.

## Regions

Some typical values for the `region` field include:

- `feii_coronal`: Forbidden Fe II transitions from highly ionized zones.
- `feii_forbidden`: Low-ionization forbidden Fe II transitions.

These labels help group lines logically or spectrally for analysis and display.

## Fe Line Templates

The following YAML files define special sets of Fe II emission lines used for modeling and template generation:

- `feii_coronal.yaml`
- `feii_model.yaml`
- `feii_forbidden.yaml`
- `feii_uv.yaml`
- `feii_IZw1.yaml`

These files were derived and adapted from the [FANTASY-AGN project](https://github.com/yukawa1/fantasy/tree/main/fantasy_agn/input).

If you use these data or the FANTASY-AGN code, please cite:

> Ilić et al. (2023)  
> *FANTASY - Fully Automated pythoN tool for AGN Spectra analYsis*  
> [The Astrophysical Journal Supplement Series, 267(1), 26](https://iopscience.iop.org/article/10.3847/1538-4365/acd783/meta)

