"""
Lightweight disk cache with TTL, used to avoid re-calling ORS/TomTom on
every dataset generation run. Route geometry (ORS) barely changes, so it
gets a long TTL. Traffic (TomTom) is live, so it gets a short TTL.
"""

import hashlib
import json
import os
import time

CACHE_DIR = "route_cache"


def _cache_key(*parts) -> str:
    raw = "_".join(str(p) for p in parts)
    return hashlib.md5(raw.encode()).hexdigest()


def cache_get(key: str, ttl_s: float):
    path = os.path.join(CACHE_DIR, f"{key}.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        entry = json.load(f)
    if time.time() - entry["_cached_at"] > ttl_s:
        return None
    return entry["data"]


def cache_set(key: str, data) -> None:
    os.makedirs(CACHE_DIR, exist_ok=True)
    path = os.path.join(CACHE_DIR, f"{key}.json")
    with open(path, "w") as f:
        json.dump({"_cached_at": time.time(), "data": data}, f)
