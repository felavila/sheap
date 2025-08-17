# SINGLE_EPOCH_ESTIMATORS (sorted by year of original paper)

**Required per entry**
- `line`: target line (must match entries in `broad_params["lines"]`)
- `kind`: `"continuum"` (uses `L_w[wavelength]`) or `"line"` (uses line luminosity array)
- `a`, `b`: single-epoch (SE) coefficients
- `vel_exp` **or** `fwhm_factor`: velocity exponent β (defaults to 2.0 if omitted)
- `f`: virial factor (keep 1.0 unless you want to inject a scale)
- `pivots`: `{ "L": luminosity pivot (erg/s), "FWHM": velocity pivot (km/s) }`
- `wavelength`: **only for** `kind="continuum"` (Å; used to pick `L_w` and optional `L_bol`)

**Optional**
- `width_def`: `"fwhm"` | `"sigma"` (which velocity width you provide)
- `extras`: flags/params for optional corrections (e.g., `{ "le20_shape": true }`, `{ "pan25_gamma": -0.34 }`)
- `enabled`: boolean (soft-disable an entry)
- `note` / `variant`: free text