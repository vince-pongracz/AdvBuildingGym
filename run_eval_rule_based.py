"""run_eval_rule_based.py – evaluate rule-based control strategies (no RL).

Single CLI entry point: ``--trial <trial_cfg.yaml>``, like run_eval_ray.py,
but the env is driven by heuristic Dict-action strategies from
``adv_building_gym.rbc_strats`` instead of a trained policy — no ray/rllib,
stable_baselines3, torch, or pyomo anywhere in the import chain.

The env is built like the RLlib eval path: the trial's statesources
provide weather and price data, and the DataCombinator swaps CSV variants
per episode. Strategies act on the previous step's measured
``info["power_breakdown"]`` (one-step lag; step 0 is a zero action).

When the trial declares ``infra_schedule`` / ``statesource_schedule`` with
multiple ``configs.eval`` entries, each entry is evaluated for ``--episodes``
episodes into its own per-config subdirectory (mirrors run_eval_ray.py).
"""

import argparse
import datetime
import json
import logging
import os
import sys
import time

import numpy as np
import pandas as pd

from adv_building_gym._common.constants import MAX_STEPS_PER_EPISODE, SECONDS_PER_HOUR
from adv_building_gym._common.json_encoder import CustomJSONEncoder
from adv_building_gym._common.warning_filters import setup_warning_filters
from adv_building_gym.components.rewards import SumRewardAggregator
from adv_building_gym.config.trial_config import TrialConfig
from adv_building_gym.core.env import AdvBuildingGym
from adv_building_gym.rbc_strats import STRATEGY_REGISTRY

# Ray-free import: adv_building_gym/ray/__init__.py and ray/utils/__init__.py are
# plain package markers; only ray/evaluation/__init__.py eagerly imports RLlib.
from adv_building_gym.ray.utils.trajectory_collector import TrajectoryCollector

setup_warning_filters()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    force=True,
)
logger = logging.getLogger("main")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate rule-based control strategies on AdvBuildingGym (no RL)",
    )
    parser.add_argument(
        "--trial", type=str, required=True,
        help="Path to trial config YAML (e.g. configs/trial_cfgs/<trial_name>.yaml). "
            "For eval, the trial's reward_schedule should point at the eval rewards "
            "and data_schedule (if set) at the held-out eval data.",
    )
    parser.add_argument(
        "--strategy", type=str, default="all",
        choices=[*STRATEGY_REGISTRY.keys(), "all"],
        help="Which rule-based strategy to evaluate (default: all)",
    )
    parser.add_argument(
        "--episodes", type=int, default=10,
        help="Number of evaluation episodes per strategy (and per eval config)",
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="Base seed (episode N uses seed + (N-1), 1-based like run_eval_ray.py). "
            "Defaults to the trial seed.",
    )
    parser.add_argument(
        "--output-dir", type=str, default="eval_results",
        help="Directory to save evaluation results",
    )
    parser.add_argument(
        "--no-save", action="store_true", help="Don't save results to file",
    )
    parser.add_argument(
        "--data-mode", type=str, default=None,
        choices=["cycle", "random"],
        help="Override variant selection mode (cycle=round-robin, random)",
    )
    parser.add_argument(
        "--data-day", type=str, default=None,
        help="Override day mode: 'each', 'random', or a date string like '2022-07-15'",
    )
    parser.add_argument(
        "--evening-start", type=float, default=17.0,
        help="self_coverage: evening discharge window start hour in [0, 24)",
    )
    parser.add_argument(
        "--evening-end", type=float, default=23.0,
        help="self_coverage: evening discharge window end hour in [0, 24). May be "
            "less than --evening-start to span midnight (e.g. 17 -> 5).",
    )
    parser.add_argument(
        "--preserve-start-soc", action=argparse.BooleanOptionalAction, default=True,
        help="Forbid ending an episode with less battery SoC than it started "
            "(caps discharges at the episode-start SoC floor)",
    )
    parser.add_argument(
        "--plot", action="store_true", default=False,
        help="Plot the best episode's trajectory after each strategy pass",
    )
    parser.add_argument(
        "--plot-all", action="store_true", default=True,
        help="Plot all episodes' trajectories after each strategy pass",
    )
    args = parser.parse_args()

    for label, hour in (("--evening-start", args.evening_start), ("--evening-end", args.evening_end)):
        if not 0.0 <= hour < 24.0:
            parser.error(f"{label} must be in [0, 24), got {hour}")
    if args.evening_start == args.evening_end:
        parser.error("--evening-start and --evening-end must differ")

    logger.info("CMD: %s", " ".join(sys.argv))
    return args


def build_env(env_config, data_combinator) -> AdvBuildingGym:
    """Base env with native Dict actions — same wiring as the RLlib eval path
    (statesources provide weather/price data; DataCombinator swaps CSVs per
    episode) but WITHOUT FlattenAction/RescaleAction (a flat 0 would map to
    mid-range for asymmetric Boxes) and without History/Forecast wrappers
    (policy-only concerns)."""
    env = AdvBuildingGym(
        env_config=env_config,
        statesources=env_config.statesources,
        infras=env_config.infras,
        rewards=env_config.reward_config.rewards,
        data_combinator=data_combinator,
        reward_aggregator=SumRewardAggregator(),
    )
    # populate info["state"] so trajectory HDF5 gets the state columns for plotting
    env.log_full_info = True
    return env


def make_strategy(strategy_name: str, env: AdvBuildingGym, args: argparse.Namespace):
    """Instantiate a strategy with its CLI-configured parameters."""
    kwargs: dict = {"preserve_start_soc": args.preserve_start_soc}
    if strategy_name == "self_coverage":
        kwargs["evening_start"] = args.evening_start
        kwargs["evening_end"] = args.evening_end
    return STRATEGY_REGISTRY[strategy_name](env, **kwargs)


def run_episode(env: AdvBuildingGym, strategy, collector: TrajectoryCollector | None,
                episode_seed: int) -> dict:
    """Run one episode; returns the per-episode stats dict."""
    obs, reset_info = env.reset(seed=episode_seed)
    strategy.reset(obs)
    if collector is not None:
        collector.reset()
        collector.on_reset(reset_info)

    # cumulative battery throughput (energy moved in/out, kWh) from power_breakdown;
    # complements the env's net-grid cum_E_kWh
    battery_name = strategy.battery.name if strategy.battery is not None else None
    step_hours = env.env_config.CONTROL_STEP / SECONDS_PER_HOUR
    battery_charged_kWh = 0.0
    battery_discharged_kWh = 0.0

    last_info: dict | None = None
    info: dict = {}
    total_reward = 0.0
    length = 0
    done = False
    while not done and length < MAX_STEPS_PER_EPISODE:
        action = strategy.act(obs, last_info)
        obs, reward, terminated, truncated, info = env.step(action)
        if collector is not None:
            collector.on_step(length, obs, action, reward, info)
        total_reward += float(reward)
        if battery_name is not None:
            production_kW, consumption_kW = info["power_breakdown"][battery_name]
            battery_charged_kWh += consumption_kW * step_hours
            battery_discharged_kWh += production_kW * step_hours
        last_info = info
        length += 1
        done = terminated or truncated

    final_soc = float(strategy.battery.soc) if strategy.battery is not None else None
    return {
        "length": length,
        "total_reward": total_reward,
        "cum_E_kWh": float(info.get("cum_E_kWh", 0.0)),
        "cum_price_EUR": float(info.get("cum_price_EUR", 0.0)),
        "seed": episode_seed,
        "data_variant": reset_info.get("data_variant"),
        "episode_date": reset_info.get("episode_date"),
        "start_battery_soc": strategy.start_soc if strategy.battery is not None else None,
        "final_battery_soc": final_soc,
        "battery_charged_kWh": battery_charged_kWh if battery_name is not None else None,
        "battery_discharged_kWh": battery_discharged_kWh if battery_name is not None else None,
    }


def evaluate_strategy(strategy, env: AdvBuildingGym, args: argparse.Namespace,
                    trial_name: str, seed: int, out_dir: str | None) -> dict:
    """Run --episodes episodes for one strategy; write summary/CSV/trajectories."""
    save = out_dir is not None
    collector = TrajectoryCollector(env) if save else None

    episode_stats: list[dict] = []
    start_time = time.time()
    for ep in range(args.episodes):
        # 1-based episode label to match run_eval_ray.py: episode N uses seed+ (N-1),
        # so RBC episode K and RL episode K share the same seed -> same data variant/date.
        episode_num = ep + 1
        episode_seed = seed + ep
        stat = run_episode(env, strategy, collector, episode_seed)
        stat = {"episode": episode_num, **stat}
        episode_stats.append(stat)
        logger.info(
            "[%s] episode %d/%d: reward=%.4f length=%d cum_E_kWh=%.3f cum_price_EUR=%.4f "
            "battery_charged_kWh=%s battery_discharged_kWh=%s",
            strategy.name, episode_num, args.episodes, stat["total_reward"],
            stat["length"], stat["cum_E_kWh"], stat["cum_price_EUR"],
            f"{stat['battery_charged_kWh']:.3f}" if stat["battery_charged_kWh"] is not None else "n/a",
            f"{stat['battery_discharged_kWh']:.3f}" if stat["battery_discharged_kWh"] is not None else "n/a",
        )
        if collector is not None:
            collector.on_episode_end(
                episode_id=episode_num, seed=episode_seed,
                metadata={"strategy": strategy.name, "trial_name": trial_name},
            )
            collector.save_json(os.path.join(out_dir, "trajectories", f"{episode_num}_trajectory.json"))
            collector.save_hdf5(os.path.join(out_dir, "trajectories.hdf5"), episode_id=str(episode_num))

    rewards = [s["total_reward"] for s in episode_stats]
    strategy_params: dict = {"preserve_start_soc": args.preserve_start_soc}
    if strategy.name == "self_coverage":
        strategy_params["evening_start"] = args.evening_start
        strategy_params["evening_end"] = args.evening_end
    if strategy.name == "price_median":
        # median of the LAST episode (per-episode values vary with the data variant)
        strategy_params["last_episode_median_price_ct_per_kWh"] = strategy.median_price

    summary = {
        "trial_name": trial_name,
        "strategy": strategy.name,
        "strategy_params": strategy_params,
        "seed": seed,
        "num_episodes": len(episode_stats),
        "eval_time_seconds": time.time() - start_time,
        "mean_reward": float(np.mean(rewards)),
        "std_reward": float(np.std(rewards)),
        "min_reward": float(np.min(rewards)),
        "max_reward": float(np.max(rewards)),
        "mean_cum_E_kWh": float(np.mean([s["cum_E_kWh"] for s in episode_stats])),
        "std_cum_E_kWh": float(np.std([s["cum_E_kWh"] for s in episode_stats])),
        "mean_cum_price_EUR": float(np.mean([s["cum_price_EUR"] for s in episode_stats])),
        "std_cum_price_EUR": float(np.std([s["cum_price_EUR"] for s in episode_stats])),
        "episodes": episode_stats,
    }
    if strategy.battery is not None:
        summary["mean_battery_charged_kWh"] = float(np.mean([s["battery_charged_kWh"] for s in episode_stats]))
        summary["mean_battery_discharged_kWh"] = float(np.mean([s["battery_discharged_kWh"] for s in episode_stats]))

    if save:
        with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as f:
            json.dump(summary, f, cls=CustomJSONEncoder, indent=4)
        pd.DataFrame(episode_stats).to_csv(os.path.join(out_dir, "episodes.csv"), index=False)
        logger.info("[%s] results saved to %s", strategy.name, out_dir)
    return summary


def _generate_plots(out_dir: str | None, args: argparse.Namespace) -> None:
    """Render trajectory plots for one strategy pass (mirrors run_eval_ray.py)."""
    if out_dir is None or not (args.plot or args.plot_all):
        return
    import h5py

    hdf5_path = os.path.join(out_dir, "trajectories.hdf5")
    if not os.path.isfile(hdf5_path):
        logger.warning("No trajectories.hdf5 found at %s — skipping plots.", hdf5_path)
        return

    from plotting.traj_plotting.trajectory_plot import generate_all_plots

    plot_dir = os.path.join(out_dir, "plots")
    if args.plot_all:
        with h5py.File(hdf5_path, "r") as hf:
            episode_ids = list(hf.keys())
        total_paths: list[str] = []
        for ep_id in episode_ids:
            ep_label = f"ep_{ep_id}"
            paths = generate_all_plots(
                hdf5_path=hdf5_path,
                episode_id=ep_id,
                output_dir=os.path.join(plot_dir, ep_label),
                file_prefix=ep_label,
            )
            total_paths.extend(paths)
        logger.info(
            "Generated %d plot files for %d episodes in %s",
            len(total_paths), len(episode_ids), plot_dir,
        )
    else:
        paths = generate_all_plots(
            hdf5_path=hdf5_path,
            output_dir=plot_dir,
            file_prefix="ep_best",
        )
        logger.info("Generated %d plot files in %s", len(paths), plot_dir)


def main() -> None:
    """Parse arguments, load the trial config, and run all strategy passes."""
    args = parse_args()

    trial = TrialConfig.load(args.trial, require_data_schedule=False, is_training=False)
    seed = args.seed if args.seed is not None else trial.seed

    # Push the eval-mode rewards onto the env config (same as run_eval_ray.py)
    trial.env_config.reward_config.rewards = trial.reward_manager.create_eval_rewards()
    logger.info(
        "Eval rewards: %s",
        [type(r).__name__ for r in trial.env_config.reward_config.rewards],
    )

    data_combinator = trial.data_combinator
    if data_combinator is not None:
        if args.data_mode is not None:
            data_combinator.mode = args.data_mode
        if args.data_day is not None:
            data_combinator.day = args.data_day
        logger.info(
            "DataCombinator: %d variants, mode=%s, day=%s",
            len(data_combinator.variants), data_combinator.mode, data_combinator.day,
        )

    strategy_names = list(STRATEGY_REGISTRY) if args.strategy == "all" else [args.strategy]
    logger.info("Strategies: %s | episodes=%d | seed=%s", strategy_names, args.episodes, seed)

    # ---- decide eval iteration axis (same logic as run_eval_ray.py) ----
    infra_eval_n = trial.infra_combinator.eval_count() if trial.infra_combinator else 0
    ss_eval_n = trial.statesource_combinator.eval_count() if trial.statesource_combinator else 0

    iteration_axis: str | None = None
    n_iters = 1
    if infra_eval_n > 1:
        iteration_axis = "infra"
        n_iters = infra_eval_n
    elif ss_eval_n > 1:
        iteration_axis = "statesource"
        n_iters = ss_eval_n
    elif trial.infra_combinator is not None:
        iteration_axis = "infra"
    elif trial.statesource_combinator is not None:
        iteration_axis = "statesource"

    run_stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S") + "_rule_based"
    run_dir = None
    if not args.no_save:
        run_dir = os.path.join(args.output_dir, run_stamp)
        os.makedirs(run_dir, exist_ok=True)
        with open(os.path.join(run_dir, "args.json"), "w", encoding="utf-8") as f:
            json.dump({**vars(args), "resolved_seed": seed}, f, cls=CustomJSONEncoder, indent=4)

    all_summaries: list[dict] = []
    for idx in range(n_iters):
        cfg_subdir: str | None = None
        if iteration_axis == "infra":
            cfg_name = trial.infra_combinator.get_eval_config_name(idx)
            cfg_subdir = cfg_name if n_iters > 1 else None
            trial.env_config.infras = trial.infra_combinator.create_infras(idx, split="eval")
            logger.info("[eval %d/%d] infra config: %s", idx + 1, n_iters, cfg_name)
        elif iteration_axis == "statesource":
            cfg_name = trial.statesource_combinator.get_eval_config_name(idx)
            cfg_subdir = cfg_name if n_iters > 1 else None
            trial.env_config.statesources = trial.statesource_combinator.create_statesources(idx, split="eval")
            logger.info("[eval %d/%d] statesource config: %s", idx + 1, n_iters, cfg_name)

        # Materialise any remaining singletons (the inline-list case, or the
        # axis we did NOT swap this iteration).
        trial.env_config.init_singletons()

        for strategy_name in strategy_names:
            # Fresh env per strategy pass so the data-variant cycling restarts
            # identically — every strategy sees the same (variant, day) sequence.
            env = build_env(trial.env_config, data_combinator)
            try:
                strategy = make_strategy(strategy_name, env, args)
            except ValueError as exc:
                if args.strategy == "all":
                    logger.warning("Skipping strategy '%s': %s", strategy_name, exc)
                    continue
                raise

            out_dir = None
            if run_dir is not None:
                out_dir = os.path.join(run_dir, strategy_name)
                if cfg_subdir is not None:
                    out_dir = os.path.join(out_dir, cfg_subdir)
                os.makedirs(out_dir, exist_ok=True)

            summary = evaluate_strategy(strategy, env, args, trial.trial_name, seed, out_dir)
            all_summaries.append({**summary, "eval_config": cfg_subdir})
            _generate_plots(out_dir, args)

    logger.info("=" * 70)
    logger.info("Rule-based eval summary (means over episodes):")
    for s in all_summaries:
        label = s["strategy"] + (f" [{s['eval_config']}]" if s["eval_config"] else "")
        logger.info(
            "  %-45s mean_reward=%10.4f  mean_cum_E_kWh=%9.3f  mean_cum_price_EUR=%9.4f",
            label, s["mean_reward"], s["mean_cum_E_kWh"], s["mean_cum_price_EUR"],
        )
    if run_dir is not None:
        logger.info("All results under %s", run_dir)


if __name__ == "__main__":
    main()
