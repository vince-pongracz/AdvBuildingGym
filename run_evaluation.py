"""
run_evaluation.py – Evaluate controllers on the LLEC-HeatPumpHouse
environments with temperature or combined reward.
"""

import sys
import os
import argparse
import time
import numpy as np
import pandas as pd
from datetime import datetime
from stable_baselines3 import PPO, SAC, DDPG, TD3, A2C
from llec_building_gym import (
    BaseBuildingGym,
    FuzzyController,
    MPCController,
    PIController,
    PIDController,
)
import logging

# Logging configuration
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)
logger.debug(f"sys.argv: {sys.argv}")


def get_model_path(
    algorithm: str,
    reward_mode: str,
    subfolder: str = None,
    prefer_best: bool = True,
    model_seed: int = 42,
) -> str:
    """
    Returns the path to the saved model file for the specified algorithm.

    Args:
        algorithm (str): Name of the RL algorithm (e.g., "ppo", "sac").
        reward_mode (str): Subdirectory based on reward mode (e.g., "default", "multi").
        subfolder (str, optional): Optional subfolder within the reward mode directory (e.g., "S01").
        prefer_best (bool): If True, prefer loading the best_model.zip if available.
        model_seed (int): Seed number used for identifying the correct model file (e.g., 42, 18).

    Returns:
        str: Path to the model file without the .zip extension.

    Raises:
        FileNotFoundError: If no suitable model file is found in the given directory.
    """
    base_path = os.path.join("models", reward_mode)
    if subfolder:
        base_path = os.path.join(base_path, subfolder)
    # Select model seed
    standard_path = os.path.join(base_path, f"{algorithm}_model_seed{model_seed}.zip")
    best_path = os.path.join(base_path, f"best_{algorithm}", "best_model.zip")

    if prefer_best and os.path.exists(best_path):
        logger.info(f"Found best model for {algorithm}: {best_path}")
        return best_path.replace(".zip", "")
    elif os.path.exists(standard_path):
        logger.info(f"Using standard model for {algorithm}: {standard_path}")
        return standard_path.replace(".zip", "")
    else:
        logger.error(
            f"No model found for {algorithm} (seed {model_seed}) in {base_path}"
        )
        raise FileNotFoundError(
            f"[ERROR] No model found for {algorithm} (seed {model_seed}) in {base_path}"
        )


def evaluate_model(
    algorithm,
    episodes=3,
    env_seed=58,
    pred_horizon=12,
    reward_mode="temperature",
    energy_price_path=None,
    outdoor_temperature_path=None,
    obs_variant=None,
    prefer_best=True,
    model_seed=42,
):
    """
    Evaluate a reinforcement learning model or a custom controller
    within the BaseBuildingGym environment.

    Args:
        algorithm (str): Identifier for the RL model or controller to evaluate.
        episodes (int): Number of episodes to simulate.
        env_seed (int): Random seed for reproducibility.
        pred_horizon (int): Prediction horizon for MPC controllers,
                            expressed in number of 5-minute control intervals
                            (e.g., 12 = 1 hour).
    Returns:
        tuple: (df_records, avg_reward)
            - df_records (pd.DataFrame): Logged simulation data for all episodes.
            - avg_reward (float): Average cumulative reward over all episodes.
    """
    # Set global seeds for reproducibility
    np.random.seed(env_seed)
    logger.info(f"Starting evaluation for: {algorithm} (mode: {reward_mode})")

    # Initialize the evaluation environment
    eval_env = BaseBuildingGym(
        mC=300,
        K=20,
        Q_HP_Max=1500,
        simulation_time=24 * 60 * 60,
        control_step=300,
        schedule_type=outdoor_temperature_path,
        training=False,
        # temperature, combined
        reward_mode=reward_mode,
        energy_price_path=energy_price_path,
        outdoor_temperature_path=outdoor_temperature_path,
        obs_variant=obs_variant,
        cop_heat=1.0,  # 3.0–3.5 typical heating
        cop_cool=1.0,  # 2.3–3.5 typical cooling
    )

    # Load pre-trained models or initialize rule-based controllers
    model_path = f"./models/{reward_mode}/{algorithm}_model_seed{model_seed}"
    logger.debug(f"model_path: {model_path}")
    if algorithm in ["ppo", "sac", "ddpg", "td3", "a2c"]:
        model_path = get_model_path(
            algorithm=algorithm,
            reward_mode=reward_mode,
            subfolder=obs_variant,
            prefer_best=prefer_best,
            model_seed=model_seed,
        )
        if algorithm == "ppo":
            model = PPO.load(model_path)
        elif algorithm == "sac":
            model = SAC.load(model_path)
        elif algorithm == "ddpg":
            model = DDPG.load(model_path)
        elif algorithm == "td3":
            model = TD3.load(model_path)
        elif algorithm == "a2c":
            model = A2C.load(model_path)
    elif algorithm == "PI Control":
        model = PIController(Kp=0.4, Ki=1.5e-4)
    elif algorithm == "PID Control":
        model = PIDController(Kp=0.4, Ki=1.5e-4, Kd=0.05)
    elif algorithm == "Fuzzy Control":
        model = FuzzyController(
            debug=False, use_gaussian=True, sigma=1.0, fine_tuning=True
        )  # gaussian membership 1: 225.69 in 0.96s
        # model = FuzzyController(debug=False, use_gaussian=False, sigma=1.0, fine_tuning=False)# triangle membership 2: 152.54 in 1.00s
        # model = FuzzyController(debug=False, use_gaussian=True,  sigma=1.0, fine_tuning=False)# gaussian membership 3: 147.83 in 0.95s
        # model = FuzzyController(debug=False, use_gaussian=False, sigma=1.0, fine_tuning=True) # triangle membership 4: 51.47 in 1.06s
    elif algorithm in ["MPC Control", "Perfect MPC Control"]:
        if pred_horizon is None:
            raise ValueError(
                "[ERROR] Please specify --mpc_horizon for MPC-based controllers."
            )
        model = MPCController(dt=300, horizon=pred_horizon)
    else:
        raise ValueError(f"[ERROR] Unknown algorithm: {algorithm}")

    if hasattr(model, "set_random_seed"):
        model.set_random_seed(env_seed)

    all_rewards = []
    rows = []
    start_time = time.time()

    for ep in range(episodes):
        logger.info(f"[Episode] {ep}")
        obs, seed_info = eval_env.reset(seed=env_seed + ep * 58)
        done = False
        episode_rewards = []
        time_step = 0
        while not done:
            # If Perfect Foresight MPC, we transmit T_out_pred:
            if algorithm in ["Perfect MPC Control", "MPC Control"]:
                horizon_start = eval_env.building.iteration
                horizon_end = horizon_start + model.horizon
                horizon_end_clipped = min(horizon_end, len(eval_env.building.T_out) - 1)
                # Setpoint-Vorhersage
                T_set_pred = eval_env.building.T_set_profile[
                    horizon_start : horizon_end_clipped + 1
                ]
                # Select forecast depending on MPC type
                if algorithm == "Perfect MPC Control":
                    T_out_pred = eval_env.building.T_out[
                        horizon_start : horizon_end_clipped + 1
                    ]
                    T_out_true = T_out_pred  # perfect forecast is ground truth
                else:  # "MPC Control"
                    T_out_pred = eval_env.building.T_out_measurement[
                        horizon_start : horizon_end_clipped + 1
                    ]
                    T_out_true = eval_env.building.T_out[
                        horizon_start : horizon_end_clipped + 1
                    ]

                # Predict action
                # action, _ = model.predict(obs, deterministic=True, T_out_pred=T_out_pred)
                action, _ = model.predict(
                    obs,
                    deterministic=True,
                    T_out_pred=T_out_pred,
                    T_set_pred=T_set_pred,
                )
                # Forecast error logging
                if len(T_out_pred) == len(T_out_true):
                    diff = np.abs(np.array(T_out_true) - np.array(T_out_pred))
                    logger.debug(
                        f"[DEBUG] Mean Forecast Error (abs): {np.mean(diff):.2f}"
                    )
                else:
                    logger.warning(
                        f"[DEBUG] Mismatch in forecast length: pred={len(T_out_pred)}, true={len(T_out_true)}"
                    )
            else:
                action, _ = model.predict(obs, deterministic=True)
            next_obs, reward, terminated, truncated, info = eval_env.step(action)
            # logger.info(f"Obs sample: {next_obs}")
            done = terminated or truncated
            episode_rewards.append(reward)
            # Log simulation step data
            rows.append(
                {
                    "algorithm": algorithm,
                    "seed": seed_info.get("seed"),
                    "episode": ep + 1,
                    "time_step": time_step,
                    "T_set": eval_env.building.T_set,
                    "T_in": eval_env.building.T_in,
                    "T_out": eval_env.building.T_out[eval_env.building.iteration - 1],
                    "T_out_measurement": eval_env.building.T_out_measurement[
                        eval_env.building.iteration - 1
                    ],
                    "wiener_noise": eval_env.building.wiener_noise[
                        eval_env.building.iteration - 1
                    ],
                    "energy_price": eval_env.building.energy_price,
                    "obs": obs,
                    "temp_deviation": info.get("temp_deviation", np.nan),
                    "action": info.get("action", np.nan),
                    "Q_HP_Max": info.get("Q_HP_Max", np.nan),
                    "controlled_Q_HP": info.get("controlled_Q_HP", np.nan),
                    "reward_temperature": info.get("reward_temperature", np.nan),
                    "reward_economic": info.get("reward_economic", np.nan),
                    "reward_temperature_norm": info.get(
                        "reward_temperature_norm", np.nan
                    ),
                    "reward_economic_norm": info.get("reward_economic_norm", np.nan),
                    "reward": reward,
                    "cumulative_reward": sum(episode_rewards),
                    "energy_price_path": energy_price_path,
                    "outdoor_temperature_path": outdoor_temperature_path,
                    "terminated": terminated,
                    "truncated": truncated,
                    "info": info,
                }
            )
            obs = next_obs
            time_step += 1
        # Log reward for the episode
        all_rewards.append(sum(episode_rewards))
    duration = time.time() - start_time
    avg_reward = np.mean(all_rewards)
    logger.info(
        f"[RESULT] Avg. reward for {algorithm} ({reward_mode}): {avg_reward:.2f} in {duration:.2f}s"
    )
    logger.info("=" * 60)

    df_records = pd.DataFrame(rows)
    df_records["evaluation_time_sec"] = duration
    return df_records, avg_reward


def get_filename_for_algorithm(
    algo: str,
    pred_horizon: int = None,
    obs_variant: str = None,
    prefer_best: bool = True,
) -> str:
    """
    Constructs the appropriate CSV filename for a given algorithm.
    Optionally includes the observation variant and prediction horizon.

    Args:
        algo (str): Name of the algorithm (e.g., "MPC Control").
        pred_horizon (int, optional): Used only for MPC-based controllers.
        obs_variant (str, optional): Observation variant to include in the filename (e.g., "S01").

    Returns:
        str: Suggested filename for saving evaluation results.
    """
    base_name = "eval"

    if obs_variant:
        base_name += f"_{obs_variant}"

    base_name += f"_{algo.replace(' ', '_')}"

    if "MPC" in algo and pred_horizon is not None:
        base_name += f"_{pred_horizon}"

    if prefer_best and algo.lower() in ["ppo", "sac", "ddpg", "td3", "a2c"]:
        base_name += "_best"
    return f"{base_name}.csv"


# Main
def main():
    """Parse CLI arguments and run evaluation for selected algorithms."""

    parser = argparse.ArgumentParser(
        description="Evaluate RL and rule-based controllers for BaseBuildingGym."
    )
    parser.add_argument(
        "--algorithms",
        nargs="+",
        default=[
            "ppo",
            "sac",
            "ddpg",
            "td3",
            "a2c",
            "PI Control",
            "PID Control",
            "Fuzzy Control",
            "MPC Control",
        ],
        help="Algorithms to evaluate.",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=10,
        help="Number of episodes to run per algorithm.",
    )
    parser.add_argument("--seed", type=int, default=58, help="Random seed.")
    parser.add_argument(
        "--model_seed",
        type=int,
        default=42,
        help="Seed number used during training for selecting the correct model file.",
    )
    parser.add_argument(
        "--mpc_horizon",
        type=int,
        default=12,
        help="prediction horizon (number of 5-minute intervals, e.g., 12 = 1 hours)",
    )
    parser.add_argument(
        "--reward_mode",
        type=str,
        default="temperature",
        choices=["temperature", "combined"],
        help="Select reward mode: 'temperature' (single-objective), or 'combined' (multi-objective).",
    )
    parser.add_argument(
        "--energy_price_path",
        type=str,
        default="data/price_data_2025.csv",
        help="Path to normalized energy price CSV.",
    )
    parser.add_argument(
        "--outdoor_temperature_path",
        type=str,
        default=None,
        help='Optional path to outdoor temperature CSV file (e.g., "data/LLEC_outdoor_temperature_5min_data.csv"). If not provided, no temperature data will be used.',
    )
    parser.add_argument(
        "--obs_variant",
        type=str,
        default="T01",
        choices=["T01", "T02", "T03", "T04", "C01", "C02", "C03", "C04"],
        help=(
            "Select observation variant:\n"
            "  T01: [noisy_temp_deviation]                        – Temperature deviation only\n"
            "  T02: [noisy_temp_deviation, time_of_day]           – Add normalized time of day\n"
            "  T03: [noisy_temp_deviation, prev_action]           – Add previous normalized action (energy usage)\n"
            "  T04: [noisy_temp_deviation, time_of_day, prev_action] – Full thermal state\n"
            "  C01: [noisy_temp_deviation, energy_price, future_prices]                – Thermal + economic\n"
            "  C02: [noisy_temp_deviation, prev_action, energy_price, future_prices]   – Add action signal\n"
            "  C03: [noisy_temp_deviation, time_of_day, energy_price, future_prices]   – Add time of day\n"
            "  C04: [noisy_temp_deviation, time_of_day, prev_action, energy_price, future_prices] – Full combined state\n"
        ),
    )
    parser.add_argument(
        "--prefer_best",
        action="store_true",
        help="Prefer loading best_model.zip if available.",
    )
    args = parser.parse_args()

    results_dir = os.path.join("results", args.reward_mode)
    if args.outdoor_temperature_path is not None:
        results_dir = os.path.join(results_dir, "outdoor_data")
    os.makedirs(results_dir, exist_ok=True)

    for algo in args.algorithms:
        # Prepare filenames
        filename = get_filename_for_algorithm(
            algo, args.mpc_horizon, args.obs_variant, args.prefer_best
        )
        csv_path = os.path.join(results_dir, filename)
        log_path = csv_path.replace(".csv", "_log.txt")

        # Setup algorithm-specific logger (complete log for this run)
        file_handler = logging.FileHandler(log_path)
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(
            logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        )
        logger.addHandler(file_handler)

        # Log parsed arguments per algorithm
        now = datetime.now()
        logger.info("[INFO] Evaluation Setup")
        logger.info("-" * 56)
        logger.info(f"Algorithm        : {algo}")
        logger.info(f"Reward Mode      : {args.reward_mode}")
        logger.info(f"Episodes         : {args.episodes}")
        logger.info(f"Seed             : {args.seed}")
        logger.info(f"Model Seed       : {args.model_seed}")
        logger.info(f"Observation      : {args.obs_variant}")
        logger.info(f"Prediction Hzn   : {args.mpc_horizon}")
        logger.info(f"Energy Price Data: {args.energy_price_path}")
        logger.info(f"Temperature Data : {args.outdoor_temperature_path}")
        logger.info(f"Prefer Best      : {args.prefer_best}")
        logger.info(f"Start Time       : {now.strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info("-" * 56)

        # Run evaluation
        df, avg_reward = evaluate_model(
            algo,
            episodes=args.episodes,
            env_seed=args.seed,
            pred_horizon=args.mpc_horizon,
            reward_mode=args.reward_mode,
            energy_price_path=args.energy_price_path,
            outdoor_temperature_path=args.outdoor_temperature_path,
            obs_variant=args.obs_variant,
            prefer_best=args.prefer_best,
            model_seed=args.model_seed,
        )
        df.to_csv(csv_path, index=False)
        end = datetime.now()

        logger.info(f"End Time        : {end.strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info(f"Duration        : {(end - now).total_seconds():.2f} seconds")
        logger.info(f"CSV file saved  : {csv_path}")
        logger.info(f"Logger file     : {log_path}")
        logger.info("Evaluation successfully completed.")
        logger.info("=" * 60)

        # Remove handler
        logger.removeHandler(file_handler)
        file_handler.close()

    logger.info("All evaluations done.")


if __name__ == "__main__":
    main()

    """
    Example Usage Scenarios
    =======================
    1. Evaluate Controller (temperature only)
    =======================
    python run_evaluation.py --algorithms "PI Control" --reward_mode temperature --obs_variant T01 --outdoor_temperature_path "data/LLEC_outdoor_temperature_5min_data.csv"           # 251.98 in 0.99s
    python run_evaluation.py --algorithms "PID Control" --reward_mode temperature --obs_variant T01 --outdoor_temperature_path "data/LLEC_outdoor_temperature_5min_data.csv"          # 251.98 in 0.88s
    python run_evaluation.py --algorithms "Fuzzy Control" --reward_mode temperature --obs_variant T01 --outdoor_temperature_path "data/LLEC_outdoor_temperature_5min_data.csv"        # 225.69 in 1.07s
    python run_evaluation.py --algorithms "Perfect MPC Control" --reward_mode temperature --obs_variant T01 --outdoor_temperature_path "data/LLEC_outdoor_temperature_5min_data.csv"  # 280.51 in 396.77s
    python run_evaluation.py --algorithms "MPC Control" --reward_mode temperature --obs_variant T01 --outdoor_temperature_path "data/LLEC_outdoor_temperature_5min_data.csv"          # 266.90 in 396.20s
    python run_evaluation.py --algorithms ppo --reward_mode temperature --obs_variant T01 --prefer_best --outdoor_temperature_path "data/LLEC_outdoor_temperature_5min_data.csv"      # 263.62 in 2.09s
    python run_evaluation.py --algorithms sac --reward_mode temperature --obs_variant T01 --prefer_best --outdoor_temperature_path "data/LLEC_outdoor_temperature_5min_data.csv"      # 262.32 in 3.05s
    """
