import os
from dotenv import load_dotenv
import requests

load_dotenv()
ORS_API_KEY = os.getenv("ORS_API_KEY")


def get_route(origin_lonlat, dest_lonlat, api_key=ORS_API_KEY):
    """
    origin_lonlat, dest_lonlat: tuples like (80.2707, 13.0827)
    Returns full ORS route JSON with waytype/surface extras.
    """
    url = "https://api.openrouteservice.org/v2/directions/driving-car/geojson"
    headers = {
        "Authorization": api_key,
        "Content-Type": "application/json"
    }
    body = {
        "coordinates": [list(origin_lonlat), list(dest_lonlat)],
        "extra_info": ["waytype", "surface"]
    }
    resp = requests.post(url, headers=headers, json=body)
    resp.raise_for_status()
    return resp.json()


WAYTYPE_CODES = {
    0: "unknown", 1: "state_road", 2: "road", 3: "street",
    4: "path", 5: "track", 6: "cycleway", 7: "footway",
    8: "steps", 9: "ferry", 10: "construction",
}


def classify_segments(route_json):
    feature = route_json["features"][0]
    coords = feature["geometry"]["coordinates"]
    waytype_extra = feature["properties"]["extras"]["waytype"]["values"]

    HIGHWAY_LIKE = {1}
    ARTERIAL_LIKE = {2, 3}

    segments = []
    for start_idx, end_idx, code in waytype_extra:
        if code in HIGHWAY_LIKE:
            road_class = "highway"
        elif code in ARTERIAL_LIKE:
            road_class = "arterial_city"
        else:
            road_class = "other"

        seg_coords = coords[start_idx:end_idx + 1]
        segments.append({
            "start_idx": start_idx,
            "end_idx": end_idx,
            "waytype_code": code,
            "road_class": road_class,
            "n_points": len(seg_coords),
        })
    return segments


def route_city_fraction(segments):
    """Fraction of route (by point count) that's arterial/city vs highway."""
    total = sum(s["n_points"] for s in segments)
    city_pts = sum(s["n_points"] for s in segments if s["road_class"] == "arterial_city")
    return city_pts / total if total else 0.0