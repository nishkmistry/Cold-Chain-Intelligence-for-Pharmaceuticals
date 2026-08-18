"""
Cold Chain Intelligence for Pharmaceuticals — Data Layer
==========================================================

Generates synthetic IoT cold-chain shipment time-series data with
injected excursion (failure) events, for training a predictive
excursion model and for live dashboard demo replay.

Usage:
    python coldchain_data_layer.py

Outputs:
    data/train_shipments.parquet   -> large labeled dataset for model training
    data/demo_shipments.parquet    -> small curated set for live dashboard demo
    data/sample_plots/*.png        -> sanity-check plots of a few shipments
"""

import os
import uuid
import numpy as np
import pandas as pd

# ----------------------------------------------------------------------
# STEP 1: Schema & constants
# ----------------------------------------------------------------------

RNG_SEED = 42
rng = np.random.default_rng(RNG_SEED)

# Drug categories and their compliance tolerance bands (deg C)
# Values approximate WHO PQS / common pharma cold-chain guidance.
DRUG_PROFILES = {
    "vaccine": {"setpoint": 5.0, "tol_min": 2.0, "tol_max": 8.0},
    "biologic": {"setpoint": -15.0, "tol_min": -20.0, "tol_max": -10.0},
    "standard": {"setpoint": 20.0, "tol_min": 15.0, "tol_max": 25.0},
}

# Excursion event types and their sampling probabilities
EVENT_TYPES = ["normal", "gradual_drift", "sudden_spike", "step_failure", "sensor_dropout"]
EVENT_WEIGHTS = [0.70, 0.10, 0.10, 0.05, 0.05]

SAMPLE_INTERVAL_MIN = 1          # 1 reading per minute for training data
SHIPMENT_DURATION_RANGE_MIN = (120, 2880)  # 2 hrs to 48 hrs

# Calibration constants derived from the Simulated Refrigerator Fault
# Diagnosis Dataset (Kaggle, samoilovmikhail) — 1300 runs x 1440 min,
# 13 fault classes, 1-min sampling. Computed from real data as follows:
#
#   noise_std_c            <- std(T_cab_meas - T_cab) on NORMAL runs
#   cycling_amplitude_c/    <- FFT peak on detrended T_cab of a NORMAL run
#   cycling_period_min
#   drift_rate_c_per_min    <- rate of the 10%->90% transient in
#                              (fault_run_T_cab - normal_baseline_T_cab),
#                              measured across COND_FOUL_MILD/SEVERE,
#                              EVAP_FAN_DEG, COMP_INEFFICIENCY,
#                              UNDERCHARGE_MILD/SEVERE runs
#   step_failure_target_    <- steady-state (fault_run - normal_baseline)
#   offset_c                   offset for EVAP_FAN_FAIL / severe faults
#
# See calibration_analysis.py for the full computation.
CALIBRATION = {
    "cycling_amplitude_c": 2.6,      # compressor on/off cycling swing (was 0.4)
    "cycling_period_min": 65,        # minutes per compressor cycle (was 18)
    "noise_std_c": 0.11,             # sensor measurement noise (deg C) (was 0.15)
    "humidity_noise_std_pct": 1.5,   # not present in reference dataset; kept as-is
    # Real drift rates are heavily right-skewed (median 0.033 C/min, p90 0.18,
    # occasional fast faults up to 3.5 C/min) rather than uniform, so we
    # sample from a fitted lognormal instead of a flat range.
    "drift_rate_lognormal_mu": -3.45,
    "drift_rate_lognormal_sigma": 1.43,
    "drift_rate_c_per_min_cap": 0.5,        # clip extreme tail for demo realism
    "spike_magnitude_c": (3.0, 8.0),        # not isolable from this dataset; kept as-is
    "step_failure_target_offset_c": (5.0, 15.0),  # was (4.0, 10.0), from EVAP_FAN_FAIL/OVERCHARGE
}


# ----------------------------------------------------------------------
# STEP 2: Baseline "healthy" curve generator
# ----------------------------------------------------------------------

def generate_baseline_curve(n_points, setpoint, route_ambient_factor=None):
    """
    Healthy shipment curve: setpoint + compressor cycling (sinusoid)
    + Gaussian sensor noise + route-linked ambient drift.
    """
    t = np.arange(n_points)

    # Compressor cycling
    cycle = CALIBRATION["cycling_amplitude_c"] * np.sin(
        2 * np.pi * t / CALIBRATION["cycling_period_min"]
    )

    # Sensor noise
    noise = rng.normal(0, CALIBRATION["noise_std_c"], n_points)

    temp = setpoint + cycle + noise

    # route_ambient_factor: array of length n_points in [0,1], where higher
    # values = hotter ambient/exposure conditions along the route. Real
    # values come from compute_route_ambient_factor() (ORS city_fraction +
    # TomTom congestion). Falls back to a mild random-walk placeholder if
    # no route_data was supplied (e.g. running this script standalone
    # without API keys).
    if route_ambient_factor is None:
        route_ambient_factor = np.clip(
            np.cumsum(rng.normal(0, 0.01, n_points)), 0, 1
        )
    ambient_load = route_ambient_factor * 0.5  # up to +0.5C heat load

    temp = temp + ambient_load

    humidity = np.clip(
        45 + rng.normal(0, CALIBRATION["humidity_noise_std_pct"], n_points),
        0, 100
    )

    return temp, humidity, route_ambient_factor


# ----------------------------------------------------------------------
# STEP 5 (real): route-linked helpers, used when route_data is supplied
# ----------------------------------------------------------------------

def interpolate_along_route(route_data, n_points):
    """
    Maps each of the shipment's n_points timestamps onto a position along
    the real route, using ORS's cumulative per-point duration (not a
    straight-line/constant-time assumption) so speed varies realistically
    by road type.

    Returns: lat (n_points,), lon (n_points,), frac_elapsed (n_points,)
    """
    coords = np.array(route_data["coordinates"])       # (M, 2) as [lon, lat]
    cum_dur = np.asarray(route_data["cumulative_duration_s"])
    total_dur = route_data["duration_s"] or cum_dur[-1]

    frac_elapsed = np.linspace(0, 1, n_points)
    target_t = frac_elapsed * total_dur

    lon = np.interp(target_t, cum_dur, coords[:, 0])
    lat = np.interp(target_t, cum_dur, coords[:, 1])
    return lat, lon, frac_elapsed


def road_class_at_times(route_data, frac_elapsed):
    """Look up road_class ('highway'/'arterial_city'/'other') for each
    elapsed-time fraction, via the coordinate index closest to that time."""
    cum_dur = np.asarray(route_data["cumulative_duration_s"])
    total_dur = route_data["duration_s"] or cum_dur[-1]
    target_t = frac_elapsed * total_dur
    idxs = np.clip(np.searchsorted(cum_dur, target_t), 0, len(cum_dur) - 1)

    segments = route_data["road_segments"]
    classes = []
    for i in idxs:
        cls = "other"
        for seg in segments:
            if seg["start_idx"] <= i <= seg["end_idx"]:
                cls = seg["road_class"]
                break
        classes.append(cls)
    return classes


def compute_route_ambient_factor(route_data, n_points):
    """
    Real ambient/exposure signal (replaces the random-walk placeholder):
    combines ORS's static city-driving fraction (more stop-and-go = more
    heat load) with TomTom's live average congestion ratio, then adds a
    mild smooth wander so it isn't perfectly flat across the shipment.
    """
    base = 0.5 * route_data.get("avg_congestion_ratio", 0.0) + \
           0.5 * route_data.get("city_fraction", 0.0)
    base = float(np.clip(base, 0, 1))

    wander = np.cumsum(rng.normal(0, 0.005, n_points))
    wander -= wander.mean()

    return np.clip(base + wander, 0, 1)


# ----------------------------------------------------------------------
# STEP 4: Failure-mode injectors
# ----------------------------------------------------------------------

def inject_gradual_drift(temp, start_idx, duration, setpoint):
    rate = min(
        rng.lognormal(CALIBRATION["drift_rate_lognormal_mu"],
                       CALIBRATION["drift_rate_lognormal_sigma"]),
        CALIBRATION["drift_rate_c_per_min_cap"],
    )
    end_idx = min(start_idx + duration, len(temp))
    ramp = rate * np.arange(end_idx - start_idx)
    temp[start_idx:end_idx] += ramp
    # after the injected window, hold at the drifted level (unit still degraded)
    if end_idx < len(temp):
        temp[end_idx:] += ramp[-1] if len(ramp) else 0
    labels = np.zeros(len(temp), dtype=object)
    labels[:] = "normal"
    labels[start_idx:end_idx] = "gradual_drift"
    return temp, labels


def inject_sudden_spike(temp, start_idx, setpoint):
    magnitude = rng.uniform(*CALIBRATION["spike_magnitude_c"])
    spike_len = rng.integers(3, 12)  # brief door-open event, few minutes
    end_idx = min(start_idx + spike_len, len(temp))
    # rise then fast exponential recovery back to baseline
    rise = np.linspace(0, magnitude, max(1, (end_idx - start_idx) // 2))
    recover_len = (end_idx - start_idx) - len(rise)
    recover = magnitude * np.exp(-np.linspace(0, 4, max(recover_len, 0)))
    profile = np.concatenate([rise, recover])[: end_idx - start_idx]
    temp[start_idx:end_idx] += profile
    labels = np.full(len(temp), "normal", dtype=object)
    labels[start_idx:end_idx] = "sudden_spike"
    return temp, labels


def inject_step_failure(temp, start_idx, setpoint):
    offset = rng.uniform(*CALIBRATION["step_failure_target_offset_c"])
    ramp_len = rng.integers(15, 40)
    end_ramp = min(start_idx + ramp_len, len(temp))
    ramp = np.linspace(0, offset, end_ramp - start_idx)
    temp[start_idx:end_ramp] += ramp
    if end_ramp < len(temp):
        temp[end_ramp:] += offset  # unit stays failed for rest of shipment
    labels = np.full(len(temp), "normal", dtype=object)
    labels[start_idx:] = "step_failure"
    return temp, labels


def inject_sensor_dropout(temp, humidity, start_idx):
    dropout_len = rng.integers(5, 20)
    end_idx = min(start_idx + dropout_len, len(temp))
    temp[start_idx:end_idx] = np.nan
    humidity[start_idx:end_idx] = np.nan
    labels = np.full(len(temp), "normal", dtype=object)
    labels[start_idx:end_idx] = "sensor_dropout"
    return temp, humidity, labels


# ----------------------------------------------------------------------
# STEP 6: Full shipment generator (ties everything together)
# ----------------------------------------------------------------------

def generate_shipment(drug_category=None, force_event=None, route_data=None):
    """
    route_data: optional dict from route_data_builder.build_route_data().
    When provided, lat/lon follow the real road polyline (time-weighted by
    actual per-segment speed), route_ambient_factor is derived from real
    city-fraction + live congestion, and traffic/road-type columns are
    populated. When omitted, falls back to the old straight-line/random-walk
    placeholders so the script still runs standalone without API keys.
    """
    if drug_category is None:
        drug_category = rng.choice(list(DRUG_PROFILES.keys()))
    profile = DRUG_PROFILES[drug_category]

    if route_data is not None:
        # Shipment duration follows the real route's duration + live delay,
        # instead of a random duration disconnected from any route.
        duration_min = max(
            int((route_data["duration_s"] + route_data.get("traffic_delay_s", 0.0)) / 60),
            30,
        )
    else:
        duration_min = int(rng.integers(*SHIPMENT_DURATION_RANGE_MIN))
    n_points = duration_min // SAMPLE_INTERVAL_MIN

    if route_data is not None:
        ambient = compute_route_ambient_factor(route_data, n_points)
    else:
        ambient = None  # generate_baseline_curve() will fall back to random walk

    temp, humidity, ambient = generate_baseline_curve(n_points, profile["setpoint"], ambient)
    labels = np.full(n_points, "normal", dtype=object)

    event = force_event if force_event else rng.choice(EVENT_TYPES, p=EVENT_WEIGHTS)

    if event != "normal" and n_points > 50:
        start_idx = int(rng.integers(int(n_points * 0.15), int(n_points * 0.75)))

        if event == "gradual_drift":
            duration = int(rng.integers(20, 60))
            temp, labels = inject_gradual_drift(temp, start_idx, duration, profile["setpoint"])
        elif event == "sudden_spike":
            temp, labels = inject_sudden_spike(temp, start_idx, profile["setpoint"])
        elif event == "step_failure":
            temp, labels = inject_step_failure(temp, start_idx, profile["setpoint"])
        elif event == "sensor_dropout":
            temp, humidity, labels = inject_sensor_dropout(temp, humidity, start_idx)

    shipment_id = str(uuid.uuid4())[:8]
    timestamps = pd.date_range("2026-01-01", periods=n_points, freq="min")

    if route_data is not None:
        # Real road-following position + road-type + live traffic context
        lat, lon, frac_elapsed = interpolate_along_route(route_data, n_points)
        road_class = road_class_at_times(route_data, frac_elapsed)
        traffic_delay_s = np.full(n_points, route_data.get("traffic_delay_s", 0.0))
        remaining_transit_s = (1 - frac_elapsed) * (
            route_data["duration_s"] + route_data.get("traffic_delay_s", 0.0)
        )
    else:
        # Placeholder: straight-line Chennai -> Bengaluru
        src = (13.0827, 80.2707)
        dst = (12.9716, 77.5946)
        lat = np.linspace(src[0], dst[0], n_points) + rng.normal(0, 0.01, n_points)
        lon = np.linspace(src[1], dst[1], n_points) + rng.normal(0, 0.01, n_points)
        road_class = ["unknown"] * n_points
        traffic_delay_s = np.zeros(n_points)
        remaining_transit_s = np.linspace(duration_min * 60, 0, n_points)

    door_open = (labels == "sudden_spike")
    vehicle_status = np.where(
        rng.random(n_points) < 0.05, "idle", "in_transit"
    )

    df = pd.DataFrame({
        "shipment_id": shipment_id,
        "timestamp": timestamps,
        "drug_category": drug_category,
        "temperature": temp,
        "humidity": humidity,
        "lat": lat,
        "lon": lon,
        "door_open": door_open,
        "vehicle_status": vehicle_status,
        "route_ambient_factor": ambient,
        "route_segment_type": road_class,
        "traffic_delay_s": traffic_delay_s,
        "remaining_transit_s": remaining_transit_s,
        "event_label": labels,
        "tol_min": profile["tol_min"],
        "tol_max": profile["tol_max"],
        "out_of_range": (temp < profile["tol_min"]) | (temp > profile["tol_max"]),
    })

    return df


# ----------------------------------------------------------------------
# Dataset builders (Steps 6 & 7)
# ----------------------------------------------------------------------

def build_training_set(n_shipments=800, route_pool=None):
    """
    route_pool: optional list of route_data dicts (from
    route_data_builder.build_route_data(), called once per real
    source/destination pair and cached — NOT called per-shipment). Each
    synthetic shipment randomly picks one route from the pool. If None,
    falls back to the placeholder straight-line route.
    """
    frames = []
    for _ in range(n_shipments):
        route_data = rng.choice(route_pool) if route_pool else None
        frames.append(generate_shipment(route_data=route_data))
    return pd.concat(frames, ignore_index=True)


def build_demo_set(route_pool=None):
    """A small curated set of shipments designed to look good live on the
    dashboard: a couple of normal ones, one clean gradual-drift-caught-early
    story, and one dramatic step failure."""
    demo_events = [
        ("vaccine", "normal"),
        ("vaccine", "gradual_drift"),
        ("biologic", "step_failure"),
        ("standard", "sudden_spike"),
        ("vaccine", "sensor_dropout"),
    ]
    frames = []
    for d, e in demo_events:
        route_data = rng.choice(route_pool) if route_pool else None
        frames.append(generate_shipment(drug_category=d, force_event=e, route_data=route_data))
    return pd.concat(frames, ignore_index=True)


# ----------------------------------------------------------------------
# STEP 8: Validation plots
# ----------------------------------------------------------------------

def save_sample_plots(df, out_dir, n_samples=5):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(out_dir, exist_ok=True)
    ids = df["shipment_id"].unique()[:n_samples]

    for sid in ids:
        sub = df[df["shipment_id"] == sid]
        fig, ax = plt.subplots(figsize=(9, 3.5))
        ax.plot(sub["timestamp"], sub["temperature"], label="Temperature (C)")
        ax.axhline(sub["tol_min"].iloc[0], color="red", linestyle="--", linewidth=0.8, label="Tolerance band")
        ax.axhline(sub["tol_max"].iloc[0], color="red", linestyle="--", linewidth=0.8)
        events = sub[sub["event_label"] != "normal"]
        if not events.empty:
            ax.scatter(events["timestamp"], events["temperature"], color="orange", s=10, label="Injected event")
        ax.set_title(f"Shipment {sid} | {sub['drug_category'].iloc[0]} | "
                     f"event: {sub['event_label'].unique().tolist()}")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, f"{sid}.png"), dpi=120)
        plt.close(fig)


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------

def build_route_pool():
    """
    Fetches (or loads from cache) a small set of representative real
    routes to assign to synthetic shipments. Requires ORS_API_KEY and
    TOMTOM_API_KEY in .env. Returns [] on any failure so dataset
    generation still works without keys (falls back to placeholder route).
    """
    try:
        from route_data_builder import build_route_data
    except ImportError:
        print("route_data_builder not available — using placeholder route.")
        return []

    # A handful of representative pharma-distribution city pairs (India).
    # (lon, lat) as required by ORS.
    city_pairs = [
        ((80.2707, 13.0827), (77.5946, 12.9716)),   # Chennai -> Bengaluru
        ((72.8777, 19.0760), (77.2090, 28.6139)),   # Mumbai -> Delhi
        ((88.3639, 22.5726), (85.8245, 20.2961)),   # Kolkata -> Bhubaneswar
        ((78.4867, 17.3850), (80.2707, 13.0827)),   # Hyderabad -> Chennai
    ]

    pool = []
    for origin, dest in city_pairs:
        try:
            route_data = build_route_data(origin, dest)
            pool.append(route_data)
        except Exception as e:
            print(f"Skipping route {origin}->{dest}: {e}")
    return pool


if __name__ == "__main__":
    os.makedirs("data", exist_ok=True)

    route_pool = build_route_pool()
    if route_pool:
        print(f"Using {len(route_pool)} real route(s) from ORS + TomTom.")
    else:
        print("No routes available — falling back to placeholder straight-line route.")

    print("Generating training set...")
    train_df = build_training_set(n_shipments=800, route_pool=route_pool)
    train_df.to_parquet("data/train_shipments.parquet", index=False)
    print(f"  -> {train_df['shipment_id'].nunique()} shipments, {len(train_df)} rows")
    print("  Class balance:")
    print(train_df.groupby("shipment_id")["event_label"].agg(
        lambda x: x[x != "normal"].mode()[0] if (x != "normal").any() else "normal"
    ).value_counts())

    print("\nGenerating demo set...")
    demo_df = build_demo_set(route_pool=route_pool)
    demo_df.to_parquet("data/demo_shipments.parquet", index=False)
    print(f"  -> {demo_df['shipment_id'].nunique()} shipments, {len(demo_df)} rows")

    print("\nSaving sample validation plots...")
    save_sample_plots(demo_df, "data/sample_plots", n_samples=5)

    print("\nDone. Files written to ./data/")
