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

## Modes

- `off` — All rewards active, no swapping. Used for
  eval (or switched off reward swap scenarios) so the agent performs on the full multi-objective reward.
- `gradual_add` — Start with the first reward, add the next one every
  `swap_every_n_iterations` episodes. Once all are added they stay active
  for the rest.
- `iterate` — One reward is active at a time; rotate to the next at evary swap (round-robin).
- `random` — Each swap randomly selects a subset of rewards (at least 1,
  up to all) -- this needs refactor, there is a TODO about it.
