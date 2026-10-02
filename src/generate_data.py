"""
Synthetic sensor data generator.

Creates three CSV files in data/:
  - sensor_readings.csv   hourly readings for every equipment unit
  - failure_events.csv    when each unit failed and why
  - maintenance_log.csv   preventive maintenance dates

All data is fake and reproducible (fixed random seed).

Run:  python -m src.generate_data
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"

SEED = 42
N_DAYS = 180
START_DATE = "2025-01-01"
UNITS_PER_TYPE = 5  # 4 types x 5 = 20 units

SENSORS = ["temperature_c", "vibration_mm_s", "current_amp", "pressure_bar", "rpm"]

# Normal operating baseline per equipment type: sensor -> (mean, noise std)
EQUIPMENT_TYPES = {
    "Pump": {
        "prefix": "PMP",
        "temperature_c": (55.0, 1.5), "vibration_mm_s": (2.5, 0.25),
        "current_amp": (30.0, 1.2), "pressure_bar": (6.0, 0.2), "rpm": (1750.0, 15.0),
    },
    "Motor": {
        "prefix": "MTR",
        "temperature_c": (65.0, 2.0), "vibration_mm_s": (1.8, 0.2),
        "current_amp": (45.0, 1.8), "pressure_bar": (1.0, 0.05), "rpm": (1480.0, 10.0),
    },
    "Conveyor": {
        "prefix": "CNV",
        "temperature_c": (40.0, 1.5), "vibration_mm_s": (3.5, 0.35),
        "current_amp": (60.0, 2.5), "pressure_bar": (2.0, 0.1), "rpm": (900.0, 12.0),
    },
    "Compressor": {
        "prefix": "CMP",
        "temperature_c": (80.0, 2.5), "vibration_mm_s": (4.0, 0.4),
        "current_amp": (75.0, 3.0), "pressure_bar": (9.0, 0.3), "rpm": (2950.0, 25.0),
    },
}

# How each failure mode distorts the signals at peak degradation,
# expressed as a fraction of the baseline mean (e.g. 1.2 = +120%).
FAILURE_MODES = {
    "Bearing Wear": {"vibration_mm_s": 1.2, "temperature_c": 0.10, "current_amp": 0.05},
    "Overheating": {"temperature_c": 0.30, "vibration_mm_s": 0.4, "current_amp": 0.10},
    "Seal Leak": {"pressure_bar": -0.25, "vibration_mm_s": 0.5, "temperature_c": 0.08},
    "Misalignment": {"vibration_mm_s": 0.9, "rpm": -0.03, "current_amp": 0.08, "temperature_c": 0.06},
}


def _pick_failure_hours(rng, n_hours, n_fail, min_gap_h=20 * 24, first_h=15 * 24):
    """Pick n_fail failure hours that are at least min_gap_h apart."""
    chosen = []
    for _ in range(2000):
        if len(chosen) == n_fail:
            break
        h = int(rng.integers(first_h, n_hours - 24))
        if all(abs(h - c) >= min_gap_h for c in chosen):
            chosen.append(h)
    return sorted(chosen)


def _simulate_unit(rng, eq_id, eq_type, n_days, upcoming_failure=False):
    """Simulate hourly readings, failures and maintenance for one unit.

    upcoming_failure=True means the unit will fail 1-3 days AFTER the data ends:
    it is already degrading, but the failure is not in failure_events.csv yet
    (it hasn't happened). This gives the dashboard some "live" high-risk units.
    """
    base = EQUIPMENT_TYPES[eq_type]
    n_hours = n_days * 24
    ts = pd.date_range(START_DATE, periods=n_hours, freq="h")
    hour_of_day = ts.hour.to_numpy()

    # --- Events -----------------------------------------------------------
    fail_hours = _pick_failure_hours(rng, n_hours, int(rng.integers(2, 5)))
    fail_modes = rng.choice(list(FAILURE_MODES), size=len(fail_hours))

    # Preventive maintenance roughly every 25-35 days
    maint_hours, h = [], int(rng.integers(5, 25)) * 24
    while h < n_hours:
        maint_hours.append(h + 8)  # maintenance happens at 08:00
        h += int(rng.integers(25, 36)) * 24
    maint_hours = [m for m in maint_hours if m < n_hours]

    # --- Normal operation: baseline + noise -------------------------------
    signals = {s: rng.normal(base[s][0], base[s][1], n_hours) for s in SENSORS}
    signals["temperature_c"] += 1.5 * np.sin(2 * np.pi * hour_of_day / 24)  # day/night cycle

    # Slow wear: vibration/temperature creep up until the next maintenance or repair
    resets = set(maint_hours) | set(fail_hours)
    hours_since_reset = np.zeros(n_hours)
    last = 0
    for i in range(n_hours):
        if i in resets:
            last = i
        hours_since_reset[i] = i - last
    signals["vibration_mm_s"] += base["vibration_mm_s"][0] * 0.0002 * hours_since_reset
    signals["temperature_c"] += 0.003 * hours_since_reset

    # --- Degradation in the 3-10 days before each failure -----------------
    degrading = list(zip(fail_hours, fail_modes))
    if upcoming_failure:
        degrading.append((n_hours + int(rng.integers(24, 72)), rng.choice(list(FAILURE_MODES))))
    for f_hour, mode in degrading:
        lead = int(rng.uniform(5, 10) * 24) if f_hour >= n_hours else int(rng.uniform(3, 10) * 24)
        start = max(0, f_hour - lead)
        idx = np.arange(start, min(f_hour, n_hours - 1) + 1)
        progress = ((idx - start) / max(1, f_hour - start)) ** 1.5  # 0 -> 1, accelerating
        for sensor, effect in FAILURE_MODES[mode].items():
            signals[sensor][idx] += base[sensor][0] * effect * progress
        # Current becomes erratic as the unit degrades
        signals["current_amp"][idx] += rng.normal(0, base["current_amp"][1] * 4 * progress)
        # After f_hour the arrays continue from baseline => "repaired"

    # --- Sensor glitches: occasional spikes and ~1% missing values --------
    for s in ["vibration_mm_s", "temperature_c", "current_amp"]:
        spikes = rng.random(n_hours) < 0.002
        signals[s][spikes] *= rng.uniform(1.5, 2.5, spikes.sum())
    for s in SENSORS:
        signals[s][rng.random(n_hours) < 0.01] = np.nan

    readings = pd.DataFrame({"timestamp": ts, "equipment_id": eq_id, "equipment_type": eq_type})
    for s in SENSORS:
        readings[s] = np.round(signals[s], 3)
    readings["operating_hours"] = int(rng.integers(2_000, 20_000)) + np.arange(n_hours)

    failures = pd.DataFrame({
        "equipment_id": eq_id,
        "failure_time": ts[fail_hours],
        "failure_mode": fail_modes,
    })
    maintenance = pd.DataFrame({"equipment_id": eq_id, "maintenance_date": ts[maint_hours]})
    return readings, failures, maintenance


def add_label(readings, failures, horizon_days=7):
    """failure_within_7d = 1 if this unit fails in the next `horizon_days` days."""
    readings = readings.copy()
    readings["failure_within_7d"] = 0
    horizon = pd.Timedelta(days=horizon_days)
    for eq_id, f in failures.groupby("equipment_id"):
        mask = readings["equipment_id"] == eq_id
        times = readings.loc[mask, "timestamp"].to_numpy()
        f_times = np.sort(f["failure_time"].to_numpy())
        nxt = np.searchsorted(f_times, times, side="right")  # first failure strictly after t
        has_next = nxt < len(f_times)
        delta = np.full(len(times), np.timedelta64(10**6, "h"))
        delta[has_next] = f_times[nxt[has_next]] - times[has_next]
        readings.loc[mask, "failure_within_7d"] = (delta <= horizon.to_timedelta64()).astype(int)
    return readings


def generate(n_days=N_DAYS, seed=SEED, out_dir=DATA_DIR):
    """Generate all datasets, write CSVs to out_dir and return them."""
    rng = np.random.default_rng(seed)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    unit_ids = [(f"{cfg['prefix']}-{n:02d}", t)
                for t, cfg in EQUIPMENT_TYPES.items() for n in range(1, UNITS_PER_TYPE + 1)]
    # A few units are about to fail right after the data ends
    upcoming = set(rng.choice(len(unit_ids), size=3, replace=False).tolist())

    all_r, all_f, all_m = [], [], []
    for i, (eq_id, eq_type) in enumerate(unit_ids):
        r, f, m = _simulate_unit(rng, eq_id, eq_type, n_days, upcoming_failure=i in upcoming)
        all_r.append(r), all_f.append(f), all_m.append(m)

    failures = pd.concat(all_f, ignore_index=True)
    maintenance = pd.concat(all_m, ignore_index=True)
    readings = add_label(pd.concat(all_r, ignore_index=True), failures)

    readings.to_csv(out_dir / "sensor_readings.csv", index=False)
    failures.to_csv(out_dir / "failure_events.csv", index=False)
    maintenance.to_csv(out_dir / "maintenance_log.csv", index=False)
    return readings, failures, maintenance


def load_raw(data_dir=DATA_DIR):
    """Load the three CSVs (with parsed dates)."""
    data_dir = Path(data_dir)
    readings = pd.read_csv(data_dir / "sensor_readings.csv", parse_dates=["timestamp"])
    failures = pd.read_csv(data_dir / "failure_events.csv", parse_dates=["failure_time"])
    maintenance = pd.read_csv(data_dir / "maintenance_log.csv", parse_dates=["maintenance_date"])
    return readings, failures, maintenance


if __name__ == "__main__":
    r, f, m = generate()
    print(f"sensor_readings.csv : {len(r):,} rows, {r['equipment_id'].nunique()} units")
    print(f"failure_events.csv  : {len(f)} failures")
    print(f"maintenance_log.csv : {len(m)} maintenance events")
    print(f"positive label rate : {r['failure_within_7d'].mean():.1%}")
