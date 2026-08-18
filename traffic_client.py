"""
TomTom Traffic API (Flow Segment Data) client.

Flow Segment Data gives live currentSpeed vs freeFlowSpeed and
currentTravelTime vs freeFlowTravelTime for the road segment closest to a
given point. We sample several points along the ORS route and aggregate
those into a route-level traffic delay + congestion signal, since ORS's
free directions endpoint only gives a static (non-live) duration.

Docs: https://developer.tomtom.com/traffic-api/documentation
"""

import os
from typing import List, Tuple

import numpy as np
import requests
from dotenv import load_dotenv

load_dotenv()
TOMTOM_API_KEY = os.getenv("TOMTOM_API_KEY")

FLOW_URL = "https://api.tomtom.com/traffic/services/4/flowSegmentData/absolute/10/json"


def get_flow_segment(lat: float, lon: float, api_key: str = TOMTOM_API_KEY, timeout: int = 10) -> dict:
    """Single Flow Segment Data lookup for one point."""
    if not api_key:
        raise RuntimeError("TOMTOM_API_KEY not set (check your .env file)")
    params = {"key": api_key, "point": f"{lat},{lon}"}
    resp = requests.get(FLOW_URL, params=params, timeout=timeout)
    resp.raise_for_status()
    return resp.json()["flowSegmentData"]


def sample_route_points(coords: List[List[float]], n_samples: int = 15) -> List[Tuple[float, float]]:
    """
    coords: list of [lon, lat] pairs from the ORS route geometry.
    Returns up to n_samples (lat, lon) pairs evenly spaced by index along
    the route (index spacing is a reasonable proxy for distance spacing on
    most routes; good enough for a traffic sample, not for precise timing).
    """
    if len(coords) <= n_samples:
        idxs = range(len(coords))
    else:
        idxs = np.linspace(0, len(coords) - 1, n_samples).astype(int)
    return [(coords[i][1], coords[i][0]) for i in idxs]  # (lat, lon)


def get_route_traffic(coords: List[List[float]], api_key: str = TOMTOM_API_KEY, n_samples: int = 15) -> dict:
    """
    Samples points along the route, queries TomTom Flow Segment Data for
    each, and aggregates into route-level traffic signals.

    Returns:
        {
            "traffic_delay_s": total added delay across sampled segments,
            "avg_congestion_ratio": 0 (free flow) to ~1 (gridlock),
            "congestion_profile": list of per-sample congestion ratios,
            "n_samples_ok": how many TomTom calls actually succeeded,
        }
    """
    sample_points = sample_route_points(coords, n_samples)
    results = []

    for lat, lon in sample_points:
        try:
            flow = get_flow_segment(lat, lon, api_key=api_key)
            results.append(flow)
        except (requests.RequestException, RuntimeError) as e:
            print(f"[traffic_client] warning: TomTom lookup failed for ({lat:.4f},{lon:.4f}): {e}")

    if not results:
        # Graceful fallback: no traffic signal available, assume free-flow.
        return {
            "traffic_delay_s": 0.0,
            "avg_congestion_ratio": 0.0,
            "congestion_profile": [],
            "n_samples_ok": 0,
        }

    delays = [max(f["currentTravelTime"] - f["freeFlowTravelTime"], 0) for f in results]
    congestion_ratios = [
        1.0 - (f["currentSpeed"] / f["freeFlowSpeed"]) if f.get("freeFlowSpeed") else 0.0
        for f in results
    ]

    return {
        "traffic_delay_s": float(sum(delays)),
        "avg_congestion_ratio": float(np.mean(congestion_ratios)),
        "congestion_profile": congestion_ratios,
        "n_samples_ok": len(results),
    }
