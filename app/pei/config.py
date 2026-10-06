"""Loads config/pei.yaml: wOBA weights and the batted-ball and zone thresholds."""

from dataclasses import dataclass
from pathlib import Path

import yaml

CONFIG_PATH = Path(__file__).resolve().parent.parent.parent / "config" / "pei.yaml"


@dataclass(frozen=True)
class Settings:
    woba_weights: dict
    hard_hit_min_ev: float
    barrel_min_ev: float
    barrel_la: tuple
    sweet_spot_la: tuple
    zone_height: tuple
    zone_half_width: float


def load(path: Path = CONFIG_PATH) -> Settings:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    return Settings(
        woba_weights={str(k): float(v) for k, v in raw["woba_weights"].items()},
        hard_hit_min_ev=float(raw["hard_hit_min_ev"]),
        barrel_min_ev=float(raw["barrel_min_ev"]),
        barrel_la=tuple(raw["barrel_la"]),
        sweet_spot_la=tuple(raw["sweet_spot_la"]),
        zone_height=tuple(raw["zone_height"]),
        zone_half_width=float(raw["zone_half_width"]),
    )
