"""Load DataCombinator configuration from YAML.

Separates data scheduling concerns (which CSV files, which years, augmented data)
from environment topology (infras, rewards, statesources) defined in env_config.py.
"""

import logging
from pathlib import Path

import yaml

from adv_building_gym.config.utils import discover_synthetic_scenarios
from adv_building_gym.data_combinator import DataCombinator

logger = logging.getLogger(__name__)

_DEFAULT_YAML_PATH = Path(__file__).resolve().parents[2] / "configs" / "data_scheduler" / "train_data_combinator_config.yaml"


def load_data_combinator_config(
    yaml_path: str | Path | None = _DEFAULT_YAML_PATH,
    seed_override: int | None = None,
) -> DataCombinator:
    """Build a DataCombinator from a YAML config file.

    Args:
        yaml_path: Path to the YAML config. Defaults to
            ``configs/data_scheduler/train_data_combinator_config.yaml`` in the project root.
        seed_override: If provided, overrides the seed in the YAML file.

    Returns:
        A fully constructed DataCombinator with scenarios expanded from
        the year/source templates defined in the YAML.
    """
    yaml_path = Path(yaml_path) if yaml_path is not None else _DEFAULT_YAML_PATH

    if not yaml_path.exists():
        logger.error("Data combinator YAML config not found: %s", yaml_path)
        raise FileNotFoundError(f"Data combinator YAML config not found: {yaml_path}")

    with open(yaml_path, "r") as f:
        cfg = yaml.safe_load(f)

    seed = seed_override if seed_override is not None else cfg["seed"]
    shuffle = cfg["shuffle"]
    years = cfg["years"]
    include_synthesized = cfg["include_synthesized"]

    # Build scenario list from templates x years, skipping missing files
    scenarios: list[dict[str, str]] = []
    for source_template in cfg["scenario_sources"]:
        for year in years:
            scenario = {
                name: pattern.format(year=year)
                for name, pattern in source_template.items()
            }
            if all(Path(p).exists() for p in scenario.values()):
                scenarios.append(scenario)
            else:
                missing = [p for p in scenario.values() if not Path(p).exists()]
                logger.debug("Skipping scenario for year %d: missing %s", year, missing)

    # Auto-discover synthesised scenarios
    if include_synthesized:
        syn_cfg = cfg["synthesized_paths"]
        synthesized = discover_synthetic_scenarios(
            years=range(min(years), max(years) + 1),
            weather_dir=syn_cfg["weather_dir"],
            price_dirs=syn_cfg["price_dirs"],
        )
        scenarios.extend(synthesized)

    variable = cfg["variable"]

    logger.info(
        "Loaded data combinator from %s: %d scenario templates, years %s, synthesized=%s",
        yaml_path.name, len(scenarios), years, include_synthesized,
    )

    return DataCombinator(
        scenarios=scenarios,
        variable=variable,
        swap_every_n_episodes=cfg["swap_every_n_episodes"],
        mode=cfg["mode"],
        day=cfg["day"],
        seed=seed,
        shuffle=shuffle,
    )
