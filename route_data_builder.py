"""
Builds the unified `route_data` dict consumed by coldchain_data_layer.py,
by combining:
  - route_client.py       (ORS: geometry, static duration, road-type segments)
  - traffic_client.py      (TomTom: live congestion / delay)

Also handles caching so re-running dataset generation doesn't burn API
quota: route geometry is cached for a day (barely changes), traffic is
cached for 10 minutes (it's live).
"""

import math
from typing import List, Optional, Tuple

import numpy as np

from cache_utils import cache_get, cache_set, _cache_key
from route_client import classify_segments, get_route, route_city_fraction
from traffic_client import get_route_traffic

ROUTE_CACHE_TTL_S = 24 * 3600   # route geometry: cache for a day
TRAFFIC_CACHE_TTL_S = 10 * 60   # live traffic: cache for 10 min


def haversine_m(lat1, lon1, lat2, lon2) -> float:
    """Great-circle distance in meters."""
    R = 6371000
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a))


def build_cumulative_duration(coords: List[List[float]], ors_segments: Optional[list],
                               total_duration_s: float) -> np.ndarray:
    """
    Distributes total route duration across each coordinate point, giving
    cum_duration[i] = seconds elapsed from route start to reach coords[i].

    Uses ORS's per-step durations when available (properties.segments[].
    steps[], each with a way_points [start_idx, end_idx] range) — this
    respects that highway stretches pass faster than city stretches.
    Falls back to distance-proportional (constant speed) if step data is
    unavailable.
    """
    n = len(coords)
    cum = np.zeros(n)

    if ors_segments:
        t = 0.0
        for seg in ors_segments:
            for step in seg.get("steps", []):
                s_idx, e_idx = step["way_points"]
                step_duration = step["duration"]
                span = max(e_idx - s_idx, 1)
                for i in range(s_idx, e_idx + 1):
                    frac = (i - s_idx) / span
                    cum[i] = t + frac * step_duration
                t += step_duration
        # Ensure monotonic (guards against overlapping way_points at joins)
        cum = np.maximum.accumulate(cum)
        return cum

    # Fallback: distance-weighted, assuming constant average speed
    dists = [0.0]
    for i in range(1, n):
        lon1, lat1 = coords[i - 1]
        lon2, lat2 = coords[i]
        dists.append(dists[-1] + haversine_m(lat1, lon1, lat2, lon2))
    total_dist = dists[-1] or 1.0
    return np.array([total_duration_s * d / total_dist for d in dists])


def build_route_data(origin_lonlat: Tuple[float, float], dest_lonlat: Tuple[float, float],
                      ors_api_key: Optional[str] = None, tomtom_api_key: Optional[str] = None,
                      use_cache: bool = True) -> dict:
    """
    Main entry point. Returns a dict shaped for coldchain_data_layer.py's
    generate_shipment(route_data=...):

        {
            "coordinates": [[lon, lat], ...],
            "cumulative_duration_s": np.ndarray,
            "distance_m": float,
            "duration_s": float,               # static ORS baseline
            "city_fraction": float,            # 0-1, arterial/city share
            "road_segments": [...],            # from classify_segments()
            "traffic_delay_s": float,          # live, from TomTom
            "avg_congestion_ratio": float,     # live, from TomTom, 0-1
        }
    """
    # --- Route geometry (ORS), cached ~1 day ---
    route_key = _cache_key("route", origin_lonlat, dest_lonlat)
    route_json = cache_get(route_key, ROUTE_CACHE_TTL_S) if use_cache else None
    if route_json is None:
        kwargs = {"api_key": ors_api_key} if ors_api_key else {}
        route_json = get_route(origin_lonlat, dest_lonlat, **kwargs)
        if use_cache:
            cache_set(route_key, route_json)

    feature = route_json["features"][0]
    coords = feature["geometry"]["coordinates"]
    summary = feature["properties"]["summary"]
    ors_segments = feature["properties"].get("segments")

    road_segments = classify_segments(route_json)
    city_fraction = route_city_fraction(road_segments)

    # --- Live traffic (TomTom), cached ~10 min ---
    traffic_key = _cache_key("traffic", origin_lonlat, dest_lonlat)
    traffic = cache_get(traffic_key, TRAFFIC_CACHE_TTL_S) if use_cache else None
    if traffic is None:
        kwargs = {"api_key": tomtom_api_key} if tomtom_api_key else {}
        traffic = get_route_traffic(coords, **kwargs)
        if use_cache:
            cache_set(traffic_key, traffic)

    cumulative_duration_s = build_cumulative_duration(coords, ors_segments, summary["duration"])

    return {
        "coordinates": coords,
        "cumulative_duration_s": cumulative_duration_s,
        "distance_m": summary["distance"],
        "duration_s": summary["duration"],
        "city_fraction": city_fraction,
        "road_segments": road_segments,
        "traffic_delay_s": traffic["traffic_delay_s"],
        "avg_congestion_ratio": traffic["avg_congestion_ratio"],
    }
