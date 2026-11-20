# Data

## Source and licence

Daily weather observations for the station **Espoo Nuuksio**, 1 January 2016 – 31 December 2025, from the Finnish Meteorological Institute (FMI) [download observations service](https://en.ilmatieteenlaitos.fi/download-observations), accessed 5 February 2026.

Source: Finnish Meteorological Institute. Licensed under [Creative Commons Attribution 4.0 International (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/); see FMI's [open data licence](https://en.ilmatieteenlaitos.fi/open-data-licence).

## Files

| File | Contents |
| --- | --- |
| `raw/Espoo Nuuksio_ 1.1.2016 - 31.12.2025_….csv` | As downloaded from FMI, unmodified: daily maximum, minimum and average temperature (°C) and precipitation amount (mm) |
| `processed/nuuksio_daily_clean.csv` | **Modified** from the raw file by `src/data_loading.py` (notebook 02) |

## Changes in the processed file

- Columns renamed to short names (`tmax`, `tmin`, `tavg`, `prcp`, `station`) and values converted to numbers.
- Precipitation: FMI's codes `-1` (no precipitation) and `0` (less than 0.1 mm) are both set to `0.0`.
- Added date columns and cyclical day-of-year features (`dayofyear_sin`, `dayofyear_cos`).

The modelling dataset (next-day target, random-noise control feature, train/validation/test split, standardization) is built in memory by `src/data_loading.py` and `src/training.py` and is not saved here.
