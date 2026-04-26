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

## Modes

- `off` — All rewards active, no swapping. Used for
  eval (or switched off reward swap scenarios) so the agent performs on the full multi-objective reward.
- `gradual_add` — Start with the first reward, add the next one every
  `swap_every_n_iterations` episodes. Once all are added they stay active
  for the rest.
- `iterate` — One reward is active at a time; rotate to the next at evary swap (round-robin).
- `random` — Maintain a stable active set of `random_active_count` rewards.
  Each swap, `random_swap_count` currently active rewards are swapped out
  for the same number of currently inactive ones, so the active-set size
  stays constant (at least N rewards are always live).
