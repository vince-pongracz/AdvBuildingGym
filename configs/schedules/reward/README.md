# Reward schedule configuration

Reward schedule files control which reward functions are active during training/evaluation and how the active set or its weights change over the course of training (curriculum / gradual TL).

The trial YAML's `rewards` section is the single source of truth for reward instantiation and default weights. Schedule files only **select** from that pool and may **override** weights — they never define new rewards.

See `EXPLORATION_RESET_README.md` for the standalone exploration-reset config that pairs with reward swaps.

## Modes

- `off` — All rewards from the trial `rewards` section are active, no swapping. Used for evaluation and for any training run that wants the full multi-objective signal without a curriculum.
- `fix` — Only the rewards listed in `on_rewards` are active for the whole run, with optional `w_override` per entry. No swapping.
- `gradual_add` — Start with the first entry of `reward_order`, add the next entry every `swap_every_n_episodes`. Once all are active they stay active. Entries can be a single `name` or a `composite:` list (all members of a composite are added in the same swap step). Optional per-entry `w_override`.
- `random` — Keep a stable active set of `random_active_count` rewards drawn from `on_rewards`. Each swap, `random_swap_count` active rewards are exchanged for the same number of inactive ones (size stays constant). Optional per-entry `w_override`.
- `dirichlet` — Resample reward **weights** every `swap_every_n_episodes` from a Dirichlet distribution (alphas = 1 for each active reward). Sampled `w_i` are multiplied by the number of active rewards. With `rejection_sampling: true` (default), draws where any `w_i < w_low` (per-reward floor in `on_rewards`) are rejected and re-sampled. The active reward *set* is fixed (the entries of `on_rewards`); only the weights change.

## Common fields

- `mode` — one of the modes above.
- `swap_every_n_episodes` — frequency of the swap (in training episodes). Not used by `off` / `fix`.
- `seed` — optional seed for reproducibility (`random`, `dirichlet`).

## Reward selection

Each entry references a reward by `name` from the trial `rewards` section. Rewards in the trial section that are *not* listed here are inactive.

- `on_rewards` — used by `fix`, `random`, `dirichlet`.
  - `w_override` (optional) — overrides the trial-section weight. Default `1.0` if omitted.
  - `w_low` (dirichlet only) — minimum acceptable weight; samples below this are rejected.
- `reward_order` — used by `gradual_add`. Ordered list of entries; each is either a `name` or a `composite:` list of names added together.
  - `w_override` (optional) — same semantics as above.

## Mode-specific fields

`gradual_add`:
- `reward_order` (see above).

`random`:
- `random_active_count` — size of the active set kept stable across swaps. Defaults to `ceil(total / 2)`. Must be in `[1, total]`.
- `random_swap_count` — number of active rewards exchanged per cycle. Default `1`; clamped to `min(active, total - active)`.

`dirichlet`:
- `rejection_sampling` (default `true`) — enable the `w_low` floor check.
- `start_weights: uniform` — use uniform weights for the first window instead of an initial Dirichlet draw. If null/omitted, the first window already uses a Dirichlet sample.
- `first_swap_after_n_episodes` — episodes before the first re-sampling (length of the start window).

## Examples

See `fix_example.yaml`, `grad_add_example.yaml`, `random_example.yaml`, `dirichlet_example.yaml`, and `eval.yaml` (mode `off`) in this directory.
