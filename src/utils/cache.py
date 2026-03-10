"""Simple file-based cache for expensive computations."""

import hashlib
import json
import pickle
from pathlib import Path
from typing import Any

import pandas as pd

from src.config import OUTPUTS_DIR

CACHE_DIR = OUTPUTS_DIR / ".cache"
CACHE_DIR.mkdir(exist_ok=True, parents=True)


def _cache_key(name: str, params: dict[str, Any]) -> str:
    """Generate cache key from name and parameters."""
    # Sort params for consistent hashing
    sorted_params = json.dumps(params, sort_keys=True)
    hash_val = hashlib.md5(sorted_params.encode()).hexdigest()[:12]
    return f"{name}_{hash_val}"


def get_cached(name: str, params: dict[str, Any]) -> Any | None:
    """Retrieve cached result if available."""
    key = _cache_key(name, params)
    cache_file = CACHE_DIR / f"{key}.pkl"
    
    if cache_file.exists():
        try:
            with open(cache_file, "rb") as f:
                return pickle.load(f)
        except Exception:
            # Cache corrupted, ignore
            return None
    
    return None


def set_cached(name: str, params: dict[str, Any], result: Any):
    """Store result in cache."""
    key = _cache_key(name, params)
    cache_file = CACHE_DIR / f"{key}.pkl"
    
    try:
        with open(cache_file, "wb") as f:
            pickle.dump(result, f)
    except Exception:
        # Caching failed, continue without cache
        pass


def clear_cache():
    """Clear all cached results."""
    for cache_file in CACHE_DIR.glob("*.pkl"):
        cache_file.unlink()
