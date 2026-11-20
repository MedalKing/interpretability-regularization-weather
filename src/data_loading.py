from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

NUUKSIO_RAW_DEFAULT = (
    Path("data/raw")
    / "Espoo Nuuksio_ 1.1.2016 - 31.12.2025_32d4a089-370d-4793-8096-7a3779d3206c.csv"
)

RENAME_MAP = {
    "Observation station": "station",
    "Maximum temperature [°C]": "tmax",
    "Minimum temperature [°C]": "tmin",
    "Average temperature [°C]": "tavg",
    "Precipitation amount [mm]": "prcp",
}

NUMERIC_COLS = ["tmax", "tmin", "tavg", "prcp"]

# ---------------------------------------------------------------------------
# Raw loading and core preprocessing
# ---------------------------------------------------------------------------

def load_raw_nuuksio_csv(path: str | Path | None = None) -> pd.DataFrame:

    if path is None:
        path = NUUKSIO_RAW_DEFAULT
    path = Path(path)

    if not path.exists():
        # Helpful message listing what *is* in data/raw
        raw_dir = path.parent
        available = list(raw_dir.glob("*.csv")) if raw_dir.exists() else []
        msg = (
            f"Raw Nuuksio CSV not found at: {path}\n"
            f"Working directory: {Path.cwd()}\n"
            f"CSV files I do see in {raw_dir}:\n"
            + "\n".join(f"  - {p.name}" for p in available)
        )
        raise FileNotFoundError(msg)

    return pd.read_csv(path)


def add_datetime_columns(df_raw: pd.DataFrame) -> pd.DataFrame:
    df = df_raw.copy()

    datetime_str = (
        df["Year"].astype(str)
        + "-"
        + df["Month"].astype(str)
        + "-"
        + df["Day"].astype(str)
        + " "
        + df["Time [Local time]"].astype(str)
    )

    df["datetime"] = pd.to_datetime(datetime_str, format="%Y-%m-%d %H:%M")
    df["date"] = df["datetime"].dt.date

    #df["year"] = df["datetime"].dt.year
    #df["month"] = df["datetime"].dt.month
    df["dayofyear"] = df["datetime"].dt.dayofyear
    #df["weekday"] = df["datetime"].dt.weekday  # 0=Mon, 6=Sun

    return df

def add_cyclical_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Assumed conventions:
    # month in 1..12, dayofyear in 1..366
    #df["month_sin"] = np.sin(2 * np.pi * (df["month"] - 1) / 12)
    #df["month_cos"] = np.cos(2 * np.pi * (df["month"] - 1) / 12)

    # 365.25 is a simple, robust choice across leap years
    df["dayofyear_sin"] = np.sin(2 * np.pi * (df["dayofyear"] - 1) / 365.25)
    df["dayofyear_cos"] = np.cos(2 * np.pi * (df["dayofyear"] - 1) / 365.25)

    return df

def harmonize_precipitation_for_thesis(
    df: pd.DataFrame,
    prcp_col: str = "prcp",
) -> pd.DataFrame:
    """
    Thesis-specific precipitation simplification.

    FMI convention:
      - -1 : no precipitation
      -  0 : precipitation below 0.1 mm

    For this thesis, both are treated as zero precipitation.
    This yields a single nonnegative continuous precipitation feature.

    Note
    ----
    This collapses the distinction between "no rain" and "trace rain",
    which is acceptable for the present forecasting task and scope.
    """
    df = df.copy()

    if prcp_col not in df.columns:
        return df

    raw = pd.to_numeric(df[prcp_col], errors="coerce")

    # Treat both -1 and 0 as zero precipitation.
    df[prcp_col] = raw.replace({-1.0: 0.0})

    return df

def standardize_nuuksio_schema(df: pd.DataFrame) -> pd.DataFrame:
    """
    Rename key columns to short names and coerce numeric columns.

    Renames:
        'Observation station' -> 'station'
        'Maximum temperature [°C]' -> 'tmax'
        'Minimum temperature [°C]' -> 'tmin'
        'Average temperature [°C]' -> 'tavg'
        'Precipitation amount [mm]' -> 'prcp'

    Coerces tmax, tmin, tavg, prcp to numeric with invalids -> NaN.

    Precipitation handling in this thesis:
      - FMI code -1 (no precipitation) is mapped to 0.0
      - FMI code  0 (precipitation below 0.1 mm) is also treated as 0.0
    """
    df = df.rename(columns=RENAME_MAP).copy()
    for col in NUMERIC_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = harmonize_precipitation_for_thesis(df)
    return df


# ---------------------------------------------------------------------------
# Clean daily dataset I/O
# ---------------------------------------------------------------------------

def make_daily_clean_nuuksio(
    df_raw: pd.DataFrame,
) -> pd.DataFrame:
    
    df = add_datetime_columns(df_raw)
    df = add_cyclical_calendar_features(df)
    df = standardize_nuuksio_schema(df)

    cols_to_keep = [
        "datetime",
        "date",
        "station",
        "tmax",
        "tmin",
        "tavg",
        "prcp",
        #"year",
        #"month",
        "dayofyear",
        #"month_sin",
        #"month_cos",
        "dayofyear_sin",
        "dayofyear_cos",
    ]

    cols_to_keep = [c for c in cols_to_keep if c in df.columns]
    df_clean = df[cols_to_keep].copy()

    if "datetime" in df_clean.columns:
        df_clean = df_clean.sort_values("datetime").reset_index(drop=True)

    return df_clean


def save_daily_clean_nuuksio(
    df: pd.DataFrame,
    path: str | Path | None = None,
) -> Path:
    """
    Save the clean Nuuksio daily dataset to CSV.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned dataframe returned by make_daily_clean_nuuksio.
    path : str or Path, optional
        Output path. Defaults to 'data/processed/nuuksio_daily_clean.csv'.

    Returns
    -------
    path : Path
        Path where the file was written.
    """
    if path is None:
        path = Path("data/processed/nuuksio_daily_clean.csv")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path


def load_daily_clean_nuuksio(path: str | Path | None = None) -> pd.DataFrame:
    """
    Convenience loader for the saved clean Nuuksio daily CSV.
    """
    if path is None:
        path = Path("data/processed/nuuksio_daily_clean.csv")
    path = Path(path)
    return pd.read_csv(path, parse_dates=["datetime"])

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def build_nextday_dataset(df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    df = df.sort_values("datetime").reset_index(drop=True).copy()
    df["tmax_next"] = df["tmax"].shift(-1)
    df = df.dropna(subset=feature_cols + ["tmax_next"])
    return df

def build_nextday_model_df(
    df_daily: pd.DataFrame,
    base_feature_cols: list[str],
    *,
    add_rand_noise: bool = False,
    rand_seed: int | None = 42,
    noise_col: str = "rand_noise",
) -> tuple[pd.DataFrame, list[str]]:
    """
    Build the modelling dataframe for next-day tmax prediction.

    Starting from a *daily* dataframe, this will:
      - (optionally) add a random noise feature at the daily level
      - call build_nextday_dataset to create tmax_next at day t+1
      - return (df_model, feature_cols_out)

    Parameters
    ----------
    df_daily : pd.DataFrame
        Clean daily dataframe (as returned by make_daily_clean_nuuksio / load_daily_clean_nuuksio).
    base_feature_cols : list[str]
        Feature columns at day t to include (excluding any random noise column).
    add_rand_noise : bool, default False
        If True, add a N(0,1) random feature at the daily level and append it to feature_cols.
    rand_seed : int or None, default 42
        Seed for the random number generator (only if add_rand_noise=True).
    noise_col : str, default "rand_noise"
        Column name for the random noise feature.

    Returns
    -------
    df_model : pd.DataFrame
        Modelling dataframe with features at day t and target 'tmax_next' at day t+1.
    feature_cols_out : list[str]
        Final list of feature columns (including noise_col if added).
    """
    df = df_daily.copy()
    feature_cols_out = list(base_feature_cols)

    if add_rand_noise:
        rng = np.random.default_rng(seed=rand_seed)
        df[noise_col] = rng.standard_normal(len(df))
        feature_cols_out.append(noise_col)

    df_model = build_nextday_dataset(df, feature_cols_out)
    return df_model, feature_cols_out