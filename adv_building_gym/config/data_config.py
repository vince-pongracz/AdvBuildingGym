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
    cfg_yaml_path: str | Path | None = _DEFAULT_YAML_PATH,
    seed_override: int | None = None,
) -> DataCombinator:
    """Build a DataCombinator from a YAML config file.

    Args:
        yaml_path: Path to the YAML config. Defaults to
            ``configs/data_scheduler/train_data_combinator_config.yaml`` in the project root.
        seed_override: If provided, overrides the seed coming from the YAML file.

    Returns:
        A fully constructed DataCombinator with scenarios expanded from
        the year/source templates defined in the YAML.
    """
    cfg_yaml_path = Path(cfg_yaml_path) if cfg_yaml_path is not None else _DEFAULT_YAML_PATH

    if not cfg_yaml_path.exists():
        logger.error("Data combinator YAML config not found: %s", cfg_yaml_path)
        raise FileNotFoundError(f"Data combinator YAML config not found: {cfg_yaml_path}")

    with open(cfg_yaml_path, "r") as data_combinator_cfg_file:
        cfg = yaml.safe_load(data_combinator_cfg_file)

    seed = seed_override if seed_override is not None else cfg["seed"]
    shuffle = cfg["shuffle"]
    years = cfg["years"]
    include_synthesized = cfg["include_synthesized"]

    # Build scenario list from templates x years, skipping missing files
    scenarios: list[dict[str, str]] = []
    for source_template in cfg["scenario_sources"]:
        for year in years:
            # Dictonary comprehension: fills the year in each path template and constructs the scenario dict entry for this source and year
            # scenario = { "weather": "data/weather/weather_{year}.csv", "price": "data/prices/price_{year}.csv", ... }
            scenario = {
                ts_data_type: path_pattern.format(year=year)
                for ts_data_type, path_pattern in source_template.items()
            }
            if all(Path(ds_path).exists() for ds_path in scenario.values()):
                scenarios.append(scenario)
            else:
                missing = [ds_path for ds_path in scenario.values() if not Path(ds_path).exists()]
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
        cfg_yaml_path.name, len(scenarios), years, include_synthesized,
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
