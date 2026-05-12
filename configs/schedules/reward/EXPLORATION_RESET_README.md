# Exploration reset configuration

Event-driven exploration kick fired on every reward set / weight change (see `README.md` for the reward schedule modes that emit such events: `gradual_add`, `random`, `dirichlet`).

On each event, exploration is pushed up so the policy re-tests the action space under the shifted objective, then linearly decays back to baseline over `decay_iterations` training iterations. This avoids the policy staying stuck in the previous optimum and gives the critic / value head room to recalibrate.

Algorithm-specific behaviour (`adv_building_gym/config/exploration_reset.py`):
- **PPO** — `entropy_coeff` is bumped to `ppo_entropy_coeff` and decays to `ppo_entropy_baseline`. Read fresh each loss step.
- **SAC** — `log_alpha` is forced to `log(sac_alpha)`. SAC's own `alpha_lr` continues retuning toward target entropy; the decay envelope still applies.
- **Both** — optimiser LRs are multiplied by `lr_multiplier` at the bump peak and interpolated back to `1.0` over `decay_iterations`.

## Fields

- `enabled` (default `false`) — master switch. When `false`, no bump is applied even on reward swaps.
- `trigger`: which swap triggers the exploration reset: if the rewards swap/change, if the infrastructure changes or if both of them changes.
- `ppo_entropy_coeff` (default `0.05`) — boosted PPO entropy coefficient at the bump peak.
- `ppo_entropy_baseline` (default `0.0`) — value the entropy coefficient decays back to.
- `sac_alpha` (default `0.5`) — value forced onto SAC `log_alpha` as `log(sac_alpha)`. Must be `> 0`.
- `decay_iterations` (default `25`) — iterations to linearly ramp boost → baseline. Must be `>= 1`.
- `lr_multiplier` (default `1.0`, i.e. off) — optimiser LR multiplier at the bump peak; interpolated back to `1.0` over `decay_iterations`. Must be `> 0`.

## Example

```yaml
exploration_reset_schedule:
  enabled: true
  trigger: on_reward_swap # on_reward_swap | on_infra_swap | both
  ppo_entropy_coeff: 0.05
  ppo_entropy_baseline: 0.0
  sac_alpha: 0.5
  decay_iterations: 50    # iterations over which to decay the bump back to baseline
  lr_multiplier: 1.5
```
