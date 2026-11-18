"""Quality flags for noisy GPS observations; preserves original observation indices."""
from __future__ import annotations
import numpy as np
import pandas as pd

EARTH_RADIUS_M = 6_371_008.8


def haversine_m(lon1, lat1, lon2, lat2):
    """Vectorised great-circle distance (metres), with coordinates ordered lon,lat."""
    lon1, lat1, lon2, lat2 = [np.deg2rad(np.asarray(v, dtype=float)) for v in
                             (lon1, lat1, lon2, lat2)]
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * EARTH_RADIUS_M * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def preprocess_track(track: pd.DataFrame, gap_seconds: float = 2.0,
                     suspicious_speed_mps: float = 55.0) -> pd.DataFrame:
    """Annotate, do not silently delete, GPS samples.

    Downstream candidate features need stable obs_idx mapping to the raw .track file.
    Suspicious speeds and repeated coordinates are diagnostic flags, not automatic removals.
    """
    if not {"longitude", "latitude", "timestamp"}.issubset(track.columns):
        raise ValueError("track must contain longitude, latitude and timestamp")
    df = track.loc[:, ["longitude", "latitude", "timestamp"]].copy().reset_index(drop=True)
    df.insert(0, "obs_idx", np.arange(len(df), dtype=np.int64))
    for col in ("longitude", "latitude", "timestamp"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    finite_coord = np.isfinite(df.longitude) & np.isfinite(df.latitude)
    df["gps_valid"] = (finite_coord & df.longitude.between(-180, 180)
                       & df.latitude.between(-90, 90) & np.isfinite(df.timestamp))
    df["dt_s"] = df.timestamp.diff()
    df["non_increasing_time"] = df.dt_s.le(0).fillna(False)
    df["large_time_gap"] = df.dt_s.gt(gap_seconds).fillna(False)
    previous_valid = df.gps_valid.shift(1, fill_value=False)
    pairs_valid = df.gps_valid & previous_valid
    distance = haversine_m(df.longitude.shift(), df.latitude.shift(),
                           df.longitude, df.latitude)
    df["step_distance_m"] = np.where(pairs_valid, distance, np.nan)
    df["speed_mps"] = np.where(pairs_valid & df.dt_s.gt(0),
                               df.step_distance_m / df.dt_s.where(df.dt_s.gt(0)), np.nan)
    df["suspicious_speed"] = df.speed_mps.gt(suspicious_speed_mps).fillna(False)
    df["repeated_position"] = (pairs_valid & df.longitude.eq(df.longitude.shift())
                               & df.latitude.eq(df.latitude.shift()))
    df["usable_coordinate"] = df.gps_valid
    return df
