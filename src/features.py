"""
Feature engineering: hourly sensor readings -> one row per unit per day.

Every feature for day D is computed from data up to the END of day D only
(rolling windows look backwards), so the model never sees the future.
The label asks: "will this unit fail in the 7 days AFTER this snapshot?"
"""
import numpy as np
import pandas as pd

from src.generate_data import EQUIPMENT_TYPES, SENSORS

WINDOWS = {"24h": "24h", "7d": "7D"}
STATS = ["mean", "std", "max"]
TYPE_COLUMNS = [f"type_{t}" for t in EQUIPMENT_TYPES]

FEATURE_COLUMNS = (
    [f"{s}_{stat}_{w}" for s in SENSORS for w in WINDOWS for stat in STATS]
    + ["vibration_slope_3d", "temperature_slope_3d",
       "operating_hours", "days_since_maintenance", "days_since_failure"]
    + TYPE_COLUMNS
)
LABEL = "failure_within_7d"


def _days_since(snapshots, event_times, data_start):
    """Days between each snapshot and the most recent event at or before it."""
    ev = np.sort(np.asarray(event_times, dtype="datetime64[ns]"))
    snaps = np.asarray(snapshots, dtype="datetime64[ns]")
    pos = np.searchsorted(ev, snaps, side="right") - 1  # last event <= snapshot
    start = np.datetime64(data_start, "ns")
    if len(ev) == 0:
        last = np.full(len(snaps), start)
    else:
        # No earlier event -> count from the start of the data
        last = np.where(pos >= 0, ev[np.clip(pos, 0, None)], start)
    return (snaps - last) / np.timedelta64(1, "D")


def _unit_features(g, unit_failures, unit_maint, data_start):
    eq_type = g["equipment_type"].iloc[0]
    g = g.set_index("timestamp").sort_index()

    # Forward-fill gaps (uses only past values). Rows still empty at the very
    # start of the series are dropped later.
    sensors = g[SENSORS].ffill()

    parts = []
    for w_name, w in WINDOWS.items():
        roll = sensors.rolling(w, min_periods=1)
        for stat in STATS:
            parts.append(getattr(roll, stat)().add_suffix(f"_{stat}_{w_name}"))
    hourly = pd.concat(parts, axis=1)
    hourly["operating_hours"] = g["operating_hours"]

    # Daily snapshot = state at the last hour of each day
    daily = hourly.resample("D").last()
    daily.index.name = "date"
    daily = daily.reset_index()
    daily["snapshot_time"] = daily["date"] + pd.Timedelta(hours=23)

    # Rate of change: average change per day over the last 3 days
    for s, name in [("vibration_mm_s", "vibration_slope_3d"), ("temperature_c", "temperature_slope_3d")]:
        col = f"{s}_mean_24h"
        daily[name] = ((daily[col] - daily[col].shift(3)) / 3).fillna(0.0)

    daily["days_since_maintenance"] = _days_since(
        daily["snapshot_time"], unit_maint["maintenance_date"], data_start)
    daily["days_since_failure"] = _days_since(
        daily["snapshot_time"], unit_failures["failure_time"], data_start)

    # Label: does a failure happen in (snapshot, snapshot + 7 days]?
    f_times = np.sort(unit_failures["failure_time"].to_numpy())
    snaps = daily["snapshot_time"].to_numpy()
    nxt = np.searchsorted(f_times, snaps, side="right")
    label = np.zeros(len(snaps), dtype=int)
    ok = nxt < len(f_times)
    label[ok] = (f_times[nxt[ok]] - snaps[ok]) <= np.timedelta64(7, "D")
    daily[LABEL] = label

    daily["equipment_id"] = g["equipment_id"].iloc[0]
    daily["equipment_type"] = eq_type
    for t in EQUIPMENT_TYPES:
        daily[f"type_{t}"] = int(t == eq_type)
    return daily


def build_features(readings, failures, maintenance):
    """Return a daily feature table for all units."""
    data_start = np.datetime64(readings["timestamp"].min())
    frames = []
    for eq_id, g in readings.groupby("equipment_id"):
        frames.append(_unit_features(
            g,
            failures[failures["equipment_id"] == eq_id],
            maintenance[maintenance["equipment_id"] == eq_id],
            data_start,
        ))
    df = pd.concat(frames, ignore_index=True)
    std_cols = [c for c in FEATURE_COLUMNS if "_std_" in c]
    df[std_cols] = df[std_cols].fillna(0.0)  # std of a single value is undefined -> 0
    df = df.dropna(subset=FEATURE_COLUMNS).reset_index(drop=True)
    cols = ["equipment_id", "equipment_type", "date", "snapshot_time"] + FEATURE_COLUMNS + [LABEL]
    return df[cols].sort_values(["date", "equipment_id"]).reset_index(drop=True)
