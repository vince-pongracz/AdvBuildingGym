# Exploration reset configuration

Event-driven exploration kick fired on every curriculum swap that matches `trigger`
(reward set/weight change, infra swap, or statesource swap). The goal is to **temporarily
raise the policy's exploration** so it re-tests the action space under the shifted objective.

Implementation: `adv_building_gym/config/training/exploration_reset.py` (config) and the
`_exploration_reset_util.py` helpers under `adv_building_gym/ray/callbacks/` and
`adv_building_gym/sb/callbacks/`.

## Algorithm-specific behaviour

- **SAC** — the temperature is a *learned* parameter (`curr_log_alpha` on the RLlib learner;
  `log_ent_coef` in SB3), auto-tuned each gradient step toward `target_entropy`. On a swap the
  live temperature is **raised to `log(sac_alpha)`, but only where it has fallen below it**
  (a raise-only kick — it never *lowers* exploration). No manual decay: SAC's own alpha
  optimiser relaxes it back toward the target between swaps.

- **PPO** — `entropy_coeff` is a *fixed* loss coefficient (no optimiser, no `target_entropy`)
  that does not change during training; the actual exploration is the Gaussian policy's `log_std`,
  which the policy gradient narrows as it converges. On a swap, `entropy_coeff` is raised to
  `ppo_entropy_coeff` (strengthening the entropy bonus so the optimiser re-widens `log_std`),
  then **linearly decayed back to its original configured value** over `decay_iterations`
  iterations. The reset performs both the raise and the decay, because nothing else moves it.

Learning rates are **not** touched (they are fixed constants in this project); the reset only
moves the exploration parameter, not the optimisation speed.

## Fields

- `enabled` (default `false`) — master switch. When `false`, no kick is applied.
- `trigger` — which swap fires the kick: `on_reward_swap`, `on_infra_swap`,
  `on_statesource_swap`, `on_reward_and_infra`, `all`, or `off`.
- `sac_alpha` (default `0.5`) — SAC: temperature is raised to `log(sac_alpha)` where below it.
  Must be `> 0`.
- `ppo_entropy_coeff` (default `0.05`) — PPO: peak `entropy_coeff` at the kick.
- `decay_iterations` (default `25`) — PPO: iterations to linearly decay `entropy_coeff` from the
  peak back to its original configured value. Must be `>= 1`. (Unused by SAC.)

## Example

```yaml
exploration_reset:
  enabled: true
  trigger: on_infra_swap   # off | on_reward_swap | on_infra_swap | on_statesource_swap | on_reward_and_infra | all
  sac_alpha: 0.5
  # PPO-only (ignored for SAC):
  ppo_entropy_coeff: 0.05
  decay_iterations: 25
```
