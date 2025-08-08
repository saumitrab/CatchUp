from __future__ import annotations
import os, yaml
from dataclasses import dataclass, field
from typing import Any, Dict

DEFAULT_CONFIG_PATH = os.environ.get("CATCHUP_CONFIG", "configs/default.yaml")

def load_yaml(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}

def env_override(cfg: dict) -> dict:
    # Flatten keys with underscores and override from env prefixed with CATCHUP_
    def walk(prefix, d, out):
        for k, v in d.items():
            key = f"{prefix}_{k}".upper() if prefix else k.upper()
            if isinstance(v, dict):
                walk(key, v, out)
            else:
                out[key] = v
        return out
    flat = walk("", cfg, {})
    for k in list(flat.keys()):
        env_key = f"CATCHUP_{k}"
        if env_key in os.environ:
            val = os.environ[env_key]
            # naive type casting
            if isinstance(flat[k], bool):
                flat[k] = val.lower() in ("1","true","yes","y")
            elif isinstance(flat[k], int):
                flat[k] = int(val)
            elif isinstance(flat[k], float):
                flat[k] = float(val)
            else:
                flat[k] = val
    # unflatten (simple)
    return cfg  # we don't deeply merge; rely on reading env directly in code where needed.

def load_config(path: str = DEFAULT_CONFIG_PATH) -> dict:
    cfg = load_yaml(path)
    env_override(cfg)
    return cfg
