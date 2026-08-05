"""In-the-loop speed comparison over 3 full episodes (288 steps each).

Compares the two observation-flattening mechanisms on the SAME real observation
at every single step:
  A. gymnasium FlattenObservation wrapper  — the current production path
     (adv_building_env_creator applies it as the outermost wrapper for all algos)
  B. RLlib FlattenObservations connector   — the retired env-to-module path,
     driven on a GROWING SingleAgentEpisode exactly like RLlib's sampling loop
     (add_env_step each step; the connector rewrites the episode's last obs).

env.step() of the Dict env is timed alongside as the baseline for context.
"""
import logging
import os
import sys
import time

logging.basicConfig(level=logging.WARNING, force=True)

# Repo root = parent of perf_test/; chdir so the trial YAML's relative data paths resolve.
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
os.chdir(REPO)

import gymnasium
import numpy as np

from ray.rllib.connectors.env_to_module import FlattenObservations
from ray.rllib.env.single_agent_episode import SingleAgentEpisode

from adv_building_gym.config.trial_config import TrialConfig
from adv_building_gym.ray.env_creator import adv_building_env_creator

TRIAL = "configs/trial_cfgs/v0/STA/lin_battery_only_price.yaml"
NUM_EPISODES = 3

print("=" * 78)
print("Flattening benchmark: FlattenObservation wrapper vs FlattenObservations connector")
print("=" * 78)
print(f"trial            : {TRIAL}")
print(f"episodes         : {NUM_EPISODES}")
print(f"SLURM_CPUS_PER_TASK : {os.environ.get('SLURM_CPUS_PER_TASK', '(unset)')}")
print(f"OMP_NUM_THREADS     : {os.environ.get('OMP_NUM_THREADS', '(unset)')}")
print(f"node             : {os.uname().nodename}")

trial = TrialConfig.load(TRIAL)
# Production chain; FlattenObservation is the outermost wrapper (env_creator.py).
flat_env = adv_building_env_creator({
    "seed": 42,
    "env_config": trial.env_config,
    "data_combinator": trial.data_combinator,
    "reward_schedule_manager": trial.reward_manager,
})
assert isinstance(flat_env, gymnasium.wrappers.FlattenObservation), type(flat_env)

# One level in: the Dict-observation env that the wrapper wraps. Stepping this
# directly yields the Dict obs both mechanisms consume.
dict_env = flat_env.env
dict_space = dict_env.observation_space
assert isinstance(dict_space, gymnasium.spaces.Dict), type(dict_space)

connector = FlattenObservations(
    input_observation_space=dict_space,
    input_action_space=dict_env.action_space,
)

print(f"obs space        : Dict with {len(dict_space.spaces)} keys "
      f"-> flat {flat_env.observation_space.shape}")
print("=" * 78)

per_ep: list[tuple[int, float, float, float]] = []
for ep in range(NUM_EPISODES):
    obs, _ = dict_env.reset(seed=500 + ep)
    sa_episode = SingleAgentEpisode(observations=[obs])

    t_step = t_wrap = t_conn = 0.0
    n = 0
    terminated = truncated = False
    while not (terminated or truncated):
        # A: the real production wrapper transform on the current Dict obs
        t0 = time.perf_counter()
        flat_wrapper = flat_env.observation(obs)
        t_wrap += time.perf_counter() - t0

        # B: the real connector on the episode whose last obs is that same Dict obs
        t0 = time.perf_counter()
        connector(rl_module=None, batch={}, episodes=[sa_episode],
                  explore=False, shared_data={})
        t_conn += time.perf_counter() - t0

        # sanity: same dim, and same values under the fixed sorted-key permutation
        if n == 0 and ep == 0:
            flat_conn = np.asarray(sa_episode.get_observations(-1), dtype=np.float32)
            assert flat_conn.shape == flat_wrapper.shape, (flat_conn.shape, flat_wrapper.shape)
            print(f"first-step check : wrapper dim {flat_wrapper.shape[0]}, "
                  f"connector dim {flat_conn.shape[0]}, "
                  f"sorted(values) equal = {np.allclose(np.sort(flat_wrapper), np.sort(flat_conn))}")

        action = dict_env.action_space.sample()
        t0 = time.perf_counter()
        obs, reward, terminated, truncated, info = dict_env.step(action)
        t_step += time.perf_counter() - t0

        sa_episode.add_env_step(
            observation=obs, action=action, reward=float(reward),
            terminated=terminated, truncated=truncated,
        )
        n += 1

    per_ep.append((n, t_step / n, t_wrap / n, t_conn / n))
    print(f"episode {ep + 1} done: {n} steps")

print()
print(f"{'episode':>8} {'steps':>6} {'env.step':>10} {'wrapper':>9} {'connector':>10}   (mean us/step)")
tot_n = tot_s = tot_w = tot_c = 0.0
for i, (n, s, w, c) in enumerate(per_ep, 1):
    print(f"{i:>8} {n:>6} {1e6 * s:>10.1f} {1e6 * w:>9.1f} {1e6 * c:>10.1f}")
    tot_n += n
    tot_s += s * n
    tot_w += w * n
    tot_c += c * n
s, w, c = tot_s / tot_n, tot_w / tot_n, tot_c / tot_n
print(f"{'ALL':>8} {int(tot_n):>6} {1e6 * s:>10.1f} {1e6 * w:>9.1f} {1e6 * c:>10.1f}")
print()
print(f"connector / wrapper ratio:            {c / w:.2f}x")
print(f"wrapper overhead on top of a step:    {100 * w / (s + w):.2f} %")
print(f"connector overhead on top of a step:  {100 * c / (s + c):.2f} %")
print(f"per-episode (288 steps) extra time:   wrapper {1e3 * 288 * w:.1f} ms  "
      f"vs  connector {1e3 * 288 * c:.1f} ms")
print(f"saving by using the wrapper:          {1e6 * (c - w):.1f} us/step  "
      f"({1e3 * 288 * (c - w):.1f} ms/episode)")
