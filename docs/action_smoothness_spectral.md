# Spectral `ActionSmoothnessReward`

## Motivation

The previous implementation counted sign reversals in the first-differences
of recent actions, weighted by squared acceleration. That heuristic fires
whenever two consecutive diffs flip sign, regardless of whether the flips
form a true high-frequency pattern or are isolated corrections following a
ramp. The README TODO at [README.md:304](../README.md#L304) called for an
explicit frequency-spectrum check on battery charge/discharge actions.

This rewrite replaces the reversal heuristic with a real-FFT analysis over
a configurable window. Energy concentrated near the Nyquist frequency is
the signature of actuator chatter; low-frequency drifts (justified ramps,
slow setpoint tracking) are left unpenalised.

## Setup

For each action key `k` with `D` action dimensions, the reward owns a
zero-padded circular buffer of length `N = n_steps` (default 16). On every
step the new action is appended; the buffer is then reordered so row 0 is
oldest and row -1 is newest. Call this `x ∈ R^{N × D}`. Buffers are zeroed
in `on_reset` so each episode starts fresh.

## Algorithm

1. **Mean-subtract** per dimension:
   `x ← x − mean(x, axis=0)`.
   Drops the DC component explicitly.
2. **Real FFT** along the time axis:
   `X = numpy.fft.rfft(x, axis=0)` → shape `(K + 1, D)` with `K = N / 2`.
   Bin `k` corresponds to frequency `k · f_s / N` Hz, where
   `f_s = 1 / control_step`. Bin 0 is DC (already 0 after mean-subtract);
   bin `K` is Nyquist (an alternation that flips every single step).
3. **Energy per bin**: `E[k, d] = |X[k, d]|²`.
4. **Bin weighting** — choose one of two modes.
5. **Normalise** by the analytical worst case `N²` (see below).
6. **Aggregate** across the `D` action dimensions (mean), then **sum**
   across all action keys. With each per-key term in `[0, 1]`, the total
   `osc ∈ [0, n_keys]`, and the raw reward `−osc ∈ [−n_keys, 0]`.

The reward is a pure penalty — no positive shift; the weighted reward
`weight · (−osc)` lies in `[−weight · n_keys, 0]`.

## Modes

### `mode='weighted'` (default)

Frequency-weighted energy:

```
w[k] = (k / K) ** freq_exponent              (shape (K+1,))
penalty_per_dim = sum_k w[k] · E[k, d]       (per dim)
```

`w[0] = 0`, `w[K] = 1`. Low-bin oscillations contribute small weight, the
Nyquist bin contributes full weight. With `freq_exponent = 1` the weight
grows linearly; `freq_exponent > 1` weighs high frequencies more
aggressively (sharper differentiation between fast and slow oscillation);
`freq_exponent < 1` softens it.

Use when training an RL policy: the penalty is a smooth function of
oscillation frequency, providing a continuous gradient signal.

### `mode='highband'`

Hard-cutoff high-band sum:

```
cutoff_bin = round(cutoff_fraction · K)
penalty_per_dim = sum_{k > cutoff_bin} E[k, d]
```

Only bins strictly above the cutoff contribute. `cutoff_fraction = 0.5`
means the upper half of the spectrum. Frequencies below the cutoff are
ignored entirely, no matter how large their amplitude.

Use when there is a known physically-motivated cutoff frequency above
which any actuation is undesirable (e.g. mechanical resonance, control
loop dynamics).

## Normalisation: worst-case derivation

The worst-case sequence within an action range of `[-1, 1]` is the
±1 square wave `a[n] = (-1)^n` of length `N` (assumed even):

- Its mean is 0, so mean-subtraction is a no-op.
- Its DFT concentrates all energy at the Nyquist bin:
  `X[K] = Σ_n (-1)^n · e^{-2πi · K · n / N} = Σ_n (-1)^n · (-1)^n = N`.
- Therefore `|X[K]|² = N²`, and all other bins are zero.

In `mode='weighted'` the worst case contributes `1 · N² = N²` to the
weighted sum (Nyquist weight is 1). In `mode='highband'` the Nyquist bin
is included for any `cutoff_fraction < 1`. So in both modes,
`worst_per_dim = N²`, and dividing by it gives a per-dim penalty in
`[0, 1]`.

## Worked example, `N = 8`

`K = 4`, worst case `N² = 64`.

| Signal `a[0..7]`                       | mode='weighted' (`p=1`) | mode='highband' (`cutoff_fraction=0.5`) |
|----------------------------------------|-------------------------|------------------------------------------|
| `[0.5, 0.5, …]` (constant)             | 0                       | 0                                        |
| linear ramp from −1 to 1               | ≈ 0.09                  | ≈ 0.04                                   |
| `[-1, 1, -1, 1, -1, 1, -1, 1]` (Nyquist square wave) | 1             | 1                                        |
| `[-1, -1, 1, 1, -1, -1, 1, 1]` (½-Nyquist, period 4) | 0.25 (energy at bin `K/2`, weight 0.5) | 0 (energy at `k = K/2`, below cutoff `k > 2`) |

Verified analytically via `numpy.fft.rfft`; reproduce by running the snippet in
the verification section below.

## Configuration

YAML entries — either mode works inside the same `rewards.yaml`:

```yaml
- class: ActionSmoothnessReward
  weight: 1.0
  params:
    n_steps: 16
    mode: weighted
    freq_exponent: 1.0
```

```yaml
- class: ActionSmoothnessReward
  weight: 1.0
  params:
    n_steps: 16
    mode: highband
    cutoff_fraction: 0.5
```

Parameters:

| Name | Default | Used by | Meaning |
|------|---------|---------|---------|
| `n_steps` | 16 | both | FFT window length (even, ≥ 4). Buffer is zero-padded until full. |
| `mode` | `weighted` | both | `'weighted'` or `'highband'`. |
| `freq_exponent` | 1.0 | `weighted` | Exponent `p` in `w[k] = (k/K)^p`. |
| `cutoff_fraction` | 0.5 | `highband` | Fraction of Nyquist above which bins are summed. |

## Implementation pointers

- [adv_building_gym/rewards/action_smoothness_reward.py](../adv_building_gym/rewards/action_smoothness_reward.py)
- Reward instances are per-env_runner — see [adv_building_gym/envs/env_creator.py:84](../adv_building_gym/envs/env_creator.py#L84). Ring buffers are scoped to each Ray env.
- Buffer is reset in `on_reset`, called by [adv_building_gym/envs/building_adv.py](../adv_building_gym/envs/building_adv.py) on each `reset()`.

## Reference

Higher-Order Action Regularisation for RL in Building Energy Management,
NeurIPS 2025 UrbanAI Workshop.
Link: https://arxiv.org/abs/2601.02061
