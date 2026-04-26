"""Synthetic dataset generation via configurable column-wise transforms.

Reads a top-level config (``preprocessing/synthesize_config.yaml``) that
selects one or more per-level configs from ``preprocessing/syn_cfgs/``
(``syn_cfg_1.yaml``, ``syn_cfg_2.yaml``, ...). Each per-level config defines,
for the ``price`` and ``weather`` domains, the per-column transform pipeline
(Gaussian noise — optionally smoothed — and constant shift) and clipping.

For each input CSV and each active syn_cfg, a sibling file
``<stem>_<syn_cfg_name>.csv`` is written next to the input.

CLI:
    python preprocessing/synthesize.py --domain weather --input data/weather/dwd/preprocessed/2023_merged_04177.csv
    python preprocessing/synthesize.py --domain price --input data/e_price/awattar/price_data_2023.csv --only syn_cfg_2
"""

from __future__ import annotations

import argparse
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

logger = logging.getLogger(__name__)

_PREPROC_DIR: Path = Path(__file__).resolve().parent
DEFAULT_TOP_CONFIG: Path = _PREPROC_DIR / "synthesize_config.yaml"
DEFAULT_SYN_CFG_DIR: Path = _PREPROC_DIR / "syn_cfgs"


# ---------------------------------------------------------------------------
# Smoothing
# ---------------------------------------------------------------------------

def _smooth_noise(noise: np.ndarray, smooth: dict[str, Any] | None) -> np.ndarray:
    """Smooth a noise series and re-normalise to its target std.

    Supported kinds:
      - ``moving_average`` with ``window`` (samples).
      - ``lowpass`` Butterworth (``order``, ``cutoff`` ∈ (0, 1) of Nyquist).

    Re-normalising back to the original sample std keeps the user-specified
    ``std`` meaningful regardless of the smoothing strength.
    """
    if smooth is None:
        return noise

    kind = smooth["kind"]
    target_std = float(noise.std())

    if kind == "moving_average":
        window = int(smooth["window"])
        smoothed = (
            pd.Series(noise)
            .rolling(window=window, center=True, min_periods=1)
            .mean()
            .to_numpy()
        )
    elif kind == "lowpass":
        # Butterworth low-pass via filtfilt (zero-phase)
        # Link: https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.butter.html
        from scipy.signal import butter, filtfilt
        order = int(smooth.get("order", 4))
        cutoff = float(smooth["cutoff"])
        b, a = butter(order, cutoff, btype="low")
        smoothed = filtfilt(b, a, noise)
    else:
        raise ValueError(f"Unknown smooth kind: {kind!r}")

    cur_std = float(smoothed.std())
    if cur_std > 0 and target_std > 0:
        smoothed = smoothed * (target_std / cur_std)
    return smoothed


# ---------------------------------------------------------------------------
# Transforms
# ---------------------------------------------------------------------------

class Transform(ABC):
    @abstractmethod
    def apply(self, series: np.ndarray, rng: np.random.Generator) -> np.ndarray: ...


@dataclass
class GaussianNoise(Transform):
    std: float
    smooth: dict[str, Any] | None = None

    def apply(self, series: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        noise = rng.normal(loc=0.0, scale=self.std, size=len(series))
        noise = _smooth_noise(noise, self.smooth)
        return series + noise


@dataclass
class ConstantShift(Transform):
    value: float

    def apply(self, series: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        return series + self.value


@dataclass
class LinScaler(Transform):
    """Uniform multiplicative gain: ``value * factor``."""
    factor: float

    def apply(self, series: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        return series * self.factor


TRANSFORMS: dict[str, type[Transform]] = {
    "gaussian_noise": GaussianNoise,
    "constant_shift": ConstantShift,
    "linscaler": LinScaler,
}


def _build_transform(spec: dict[str, Any]) -> Transform:
    spec = dict(spec)
    kind = spec.pop("type")
    cls = TRANSFORMS[kind]
    return cls(**spec)


# ---------------------------------------------------------------------------
# Per-column pipeline
# ---------------------------------------------------------------------------

def apply_column_pipeline(
    series: pd.Series,
    transforms_specs: list[dict[str, Any]],
    clip_spec: dict[str, float] | None,
    rng: np.random.Generator,
) -> pd.Series:
    values = series.to_numpy(dtype=np.float64, copy=True)
    for spec in transforms_specs:
        values = _build_transform(spec).apply(values, rng)
    if clip_spec:
        values = np.clip(
            values,
            clip_spec.get("min", -np.inf),
            clip_spec.get("max", np.inf),
        )
    return pd.Series(values, index=series.index, name=series.name)


# ---------------------------------------------------------------------------
# File-level synthesis
# ---------------------------------------------------------------------------

def synthesize_file(
    input_path: Path,
    output_path: Path,
    columns_spec: dict[str, dict[str, Any]],
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Apply per-column transforms to ``input_path`` and write to ``output_path``."""
    df = pd.read_csv(input_path)
    logger.info("Loaded %d records from %s", len(df), input_path)

    present = [c for c in columns_spec if c in df.columns]
    missing = set(columns_spec) - set(present)
    if missing:
        logger.info("Columns not in CSV (skipped): %s", sorted(missing))

    for col in present:
        spec = columns_spec[col]
        before = df[col]
        before_min, before_mean, before_max = before.min(), before.mean(), before.max()
        df[col] = apply_column_pipeline(
            before,
            transforms_specs=spec.get("transforms", []),
            clip_spec=spec.get("clip"),
            rng=rng,
        )
        after = df[col]
        logger.info(
            "  %s:\n    before: min=%.4f, mean=%.4f, max=%.4f\n    after:  min=%.4f, mean=%.4f, max=%.4f",
            col,
            before_min, before_mean, before_max,
            after.min(), after.mean(), after.max(),
        )

    df.to_csv(output_path, index=False)
    logger.info("Saved %d synthesised records to %s", len(df), output_path)
    return df


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

@dataclass
class SynCfg:
    name: str
    price: dict[str, dict[str, Any]]
    weather: dict[str, dict[str, Any]]


def load_syn_cfg(name: str, syn_cfg_dir: Path = DEFAULT_SYN_CFG_DIR) -> SynCfg:
    path = syn_cfg_dir / f"{name}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"Syn config not found: {path}")
    with open(path) as f:
        cfg = yaml.safe_load(f) or {}
    return SynCfg(
        name=cfg.get("name", name),
        price=cfg.get("price", {}) or {},
        weather=cfg.get("weather", {}) or {},
    )


def load_top_config(path: Path = DEFAULT_TOP_CONFIG) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Top synthesise config not found: {path}")
    with open(path) as f:
        return yaml.safe_load(f) or {}


def _output_path(input_path: Path, syn_cfg_name: str) -> Path:
    return input_path.with_name(f"{input_path.stem}_{syn_cfg_name}.csv")


# ---------------------------------------------------------------------------
# Pipeline entry point
# ---------------------------------------------------------------------------

def run_synthesis(
    price_files: list[Path],
    weather_files: list[Path],
    top_cfg_path: Path | str = DEFAULT_TOP_CONFIG,
    only: list[str] | None = None,
) -> dict[str, int]:
    """Run synthesis for every active syn_cfg over the given files.

    Args:
        price_files: Preprocessed yearly price CSVs.
        weather_files: Preprocessed yearly weather CSVs.
        top_cfg_path: Top-level synthesise config selecting active syn_cfgs.
        only: Optional restriction of active syn_cfgs (overrides top config).

    Returns:
        Counts of files written, keyed by domain.
    """
    top = load_top_config(Path(top_cfg_path))
    base_seed = int(top.get("seed", 42))
    syn_cfg_dir = Path(top.get("syn_cfg_dir", DEFAULT_SYN_CFG_DIR))

    if not syn_cfg_dir.is_absolute():
        syn_cfg_dir = (_PREPROC_DIR.parent / syn_cfg_dir).resolve()

    active_cfgs = only if only is not None else list(top.get("active_configs", []))
    if not active_cfgs:
        logger.warning("No active syn_cfgs — nothing to do")
        return {"price": 0, "weather": 0}

    stats = {"price": 0, "weather": 0}

    for cfg_idx, cfg_name in enumerate(active_cfgs):
        syn_cfg = load_syn_cfg(cfg_name, syn_cfg_dir=syn_cfg_dir)
        logger.info("=== Synthesising with %s ===", syn_cfg.name)

        for csv_path in price_files:
            if not syn_cfg.price:
                continue
            rng = np.random.default_rng(base_seed + cfg_idx * 1000 + stats["price"])
            out = _output_path(csv_path, syn_cfg.name)
            synthesize_file(csv_path, out, syn_cfg.price, rng)
            stats["price"] += 1

        for csv_path in weather_files:
            if not syn_cfg.weather:
                continue
            rng = np.random.default_rng(base_seed + cfg_idx * 1000 + 500 + stats["weather"])
            out = _output_path(csv_path, syn_cfg.name)
            synthesize_file(csv_path, out, syn_cfg.weather, rng)
            stats["weather"] += 1

    logger.info("Synthesis complete: %d price, %d weather file(s).",
                stats["price"], stats["weather"])
    return stats


# ---------------------------------------------------------------------------
# CLI (standalone use on a single file)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    parser = argparse.ArgumentParser(description="Generate synthetic CSV via configured transforms")
    parser.add_argument("--domain", choices=["price", "weather"], required=True)
    parser.add_argument("--input", required=True, help="Path to input CSV")
    parser.add_argument("--config", default=str(DEFAULT_TOP_CONFIG), help="Top-level synthesise config")
    parser.add_argument("--only", nargs="+", default=None, help="Restrict to these syn_cfg names (e.g. syn_cfg_2)")
    args = parser.parse_args()

    input_path = Path(args.input)
    if args.domain == "price":
        run_synthesis(price_files=[input_path], weather_files=[],
                    top_cfg_path=args.config, only=args.only)
    else:
        run_synthesis(price_files=[], weather_files=[input_path],
                    top_cfg_path=args.config, only=args.only)
