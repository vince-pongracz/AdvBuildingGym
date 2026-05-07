# Reward schedule configuration

Reward schedule files (`reward_schedule_*.yaml`, `rsch_*.yaml`) control 
reward function activation during training and how they are swapped during 
the training (gradual TL learning).

## Fields

- `rewards_file` — path to the shared reward definitions (relative to the
  schedule file).
- `mode` — reward function swap strategy.
- `swap_every_n_iterations` — frequncy of reward set change
  (in training episodes).
- `seed` — seed for the `random` mode (reproducibility).
- `random_active_count` — (random mode only) size of the active reward
  set kept stable across swaps. Defaults to `ceil(total / 2)`. Must be in
  `[1, total_rewards]`.
- `random_swap_count` — (random mode only) number of currently active
  rewards swapped out per cycle for the same number of inactive ones.
  Default `1`; clamped to `min(active, total - active)`.
- `exploration_bump` — (optional) event-driven exploration kick applied
  each time the active reward set changes. Linearly decays back to
  baseline. Useful so the policy re-tests the action space under the
  shifted objective instead of staying stuck in the previous optimum.
  Fields:
    - `enabled` (default `false`)
    - `ppo_entropy_coeff` — boosted PPO entropy coefficient (default `0.05`)
    - `ppo_entropy_baseline` — value to decay back to (default `0.0`)
    - `sac_alpha` — value forced onto SAC `log_alpha` as `log(sac_alpha)`
      (default `0.5`); SAC's own `alpha_lr` will continue to retune it.
    - `decay_iterations` — iterations to linearly ramp boost → baseline
      (default `25`).
    - `lr_multiplier` — optional optimiser LR multiplier at the bump peak,
      interpolated back to `1.0` over `decay_iterations` (default `1.0`,
      i.e. off). Helps the critic / value head recalibrate to the shifted
      reward landscape.

## Modes

- `off` — All rewards active, no swapping. Used for
  eval (or switched off reward swap scenarios) so the agent performs on the full multi-objective reward.
- `gradual_add` — Start with the first reward, add the next one every
  `swap_every_n_iterations` episodes. Once all are added they stay active
  for the rest.
- `random` — Maintain a stable active set of `random_active_count` rewards.
  Each swap, `random_swap_count` currently active rewards are swapped out
  for the same number of currently inactive ones, so the active-set size
  stays constant (at least N rewards are always live).
