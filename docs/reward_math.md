# Reward Mathematics (V0 reward set)

Mathematical definition of every `*_reward_v0.py` reward function in
`adv_building_gym/components/rewards/`.

Conventions used throughout:

- Every `get_reward` returns the scalar `w · r` where `w` is the per-reward
  `weight`. The formulas below give the **raw** reward `r` (before
  multiplication by `w`).
- **Canonical power sign convention** (set by `EnergyTracker`):
  `net_power_kW > 0` ⇒ **export** (production > consumption);
  `net_power_kW < 0` ⇒ **import / consumption**.
- `clip(x, a, b) = min(max(x, a), b)`.
- `s_*` are normalised observation entries; `ctxt_*` are per-episode context
  scalars.

---

## 1. TempRewardV0 — temperature comfort

File: `temp_reward_v0.py`. Per-step, range `[-1, 1]`.

Let the temperature error in °C be

$$ d = (s_{\text{temp\_in\_norm}} - s_{\text{desired\_temp\_in\_norm}}) \cdot c_{\text{temp\_abs\_max}} $$

where `c_temp_abs_max = ctxt_temp_abs_max` (default 60 °C). With constants
`exp_scale = 2.0` (denote $E$) and `x_scale = 1.4` (denote $\alpha$):

**Precision driver** (sharp peak near setpoint, clipped to be non-negative):

$$ r_{\text{prec}} = \mathrm{clip}\!\left(E\,e^{-|\alpha d|} - 1,\; 0,\; 1\right) $$

**Slow driver** (linear far-field penalty). Let the zero-crossing of the
precision exponential be $x_0 = -\tfrac{1}{\alpha}\ln\tfrac{1}{E}$ and the
threshold $T = 40 - x_0$ (so the slow term hits $-1$ at ≈ 40 °C error):

$$
r_{\text{slow}} =
\begin{cases}
\mathrm{clip}\!\left(-\dfrac{d - x_0}{T},\,-1,\,0\right), & d > 0 \\[2mm]
\mathrm{clip}\!\left(\dfrac{d + x_0}{T},\,-1,\,0\right), & d \le 0
\end{cases}
$$

**Total:** $r = \mathrm{clip}(r_{\text{prec}} + r_{\text{slow}},\,-1,\,1)$.

Peaks at $+1$ when $d = 0$, decays towards $-1$ as $|d|$ grows.

---

## 2. EconomicRewardV0 — dense electricity cost / income

File: `economic_reward_v0.py`. Per-step, range `[-1, 1]`.

Let `p = s_E_price` (normalised price) and `P_ref` the reference power
(`ctxt_operator_max_power_kW` if positive, else constructor
`reference_power_kW`, default 15 kW). With `P_net = net_power_kW`:

$$ r = \mathrm{clip}\!\left(\frac{P_{\text{net}} \cdot p}{P_{\text{ref}}},\; -1,\; 1\right) $$

Sign matrix (canonical convention):

| | price > 0 | price < 0 |
|---|---|---|
| export ($P_{net}>0$) | $+$ income | $-$ paying to dump |
| consume ($P_{net}<0$) | $-$ cost | $+$ paid to consume |

---

## 3. LongTermEconomicRewardV0 — sparse, episode-aggregated economics

File: `long_term_economic_reward_v0.py`. Returns `0` every step until the
episode terminates, then flushes. Range at flush `[-N, +N]`
(where `N = steps_seen`).

Per-step the price is rescaled by a data-driven denominator. With
`dyn_max = ctxt_E_price_dynamic_max` and `dyn_max_ep = ctxt_E_price_dynamic_max_ep`
(both default 1.0):

$$ \tilde p = \frac{s_{\text{E\_price}}}{0.7\,\text{dyn\_max} + 0.3\,\text{dyn\_max\_ep}} $$

$$ \text{per\_step} = \mathrm{clip}\!\left(\frac{P_{\text{net}} \cdot \tilde p}{P_{\text{ref}}},\,-1,\,1\right), \qquad A \mathrel{+}= \text{per\_step},\quad n \mathrel{+}= 1 $$

**How the `dyn_max` context scalars are computed** (`EnergyPriceDataSource`).
Both default to **1.0** (no-op) unless `dynamic_max_price_calc = True`. When
enabled, a statistic of the raw baseprice slice $b$ — `mean` / `median` /
`percentile(q)` / `mean_above_median`, per `max_calc_mode` — is rescaled into
`s_E_price`'s normalised frame by `price_max` (the normalisation scale factor):

$$ \text{dyn\_max} = \max\!\left(\frac{\text{stat}(b)}{\text{price\_max}},\; 10^{-6}\right) $$

- **`ctxt_E_price_dynamic_max`**: $b$ = full CSV series, computed once per data variant.
- **`ctxt_E_price_dynamic_max_ep`**: same statistic over the episode window
  `b = baseprice[row_offset : row_offset + episode_length]`, recomputed each `reset()`.

At episode end ($n = $ `episode_length`, or `info["terminated"]`):

$$ r = \mathrm{clip}(A,\,-n,\,n) $$

`P_ref` resolves as in EconomicRewardV0 (default 15 kW).

---

## 4. MinimiseEnergyConsumptionRewardV0 — sparse energy throughput

File: `energy_consumption_reward_v0.py`. Sparse, flushed at episode end.
Range at flush `[-N, +N]`.

With `P_op = op_max_kW` (`ctxt_operator_max_power_kW` if positive, else
`reference_power_kW`, default 20 kW):

$$ \text{per\_step} = \mathrm{clip}\!\left(\frac{P_{\text{net}}}{P_{\text{op}}},\,-1,\,1\right), \qquad A \mathrel{+}= \text{per\_step},\quad n \mathrel{+}= 1 $$

Consumption ($P_{net}<0$) ⇒ negative per-step (penalty); export ⇒ positive.
At episode end:

$$ r = \mathrm{clip}(A,\,-n,\,n) $$

(Note: `export_scale` is accepted but the asymmetric scaling line is commented
out, so export and import are treated symmetrically.)

---

## 5. OperatorEnergyControlRewardV0 — grid-limit guardrail

File: `operator_energy_control_reward_v0.py`. Pure penalty, per-step range
`[-1, 0]`. Symmetric in both flow directions.

Define the load ratio (using the **absolute** net power):

$$ \rho = \frac{|P_{\text{net}}|}{\text{ctxt\_operator\_max\_power\_kW}} $$

Constants: `harsh_penalty = -1.0` ($H$), `soft_threshold_pct = 0.9` ($s$),
`recovery_steps = 3` ($R$), decay scale `5.0`, recovery rate
$\lambda = \ln(100)/\max(R,1)$. Let $k$ = steps since the last over-limit
violation.

$$
r =
\begin{cases}
H, & \rho > 1 \quad (\text{over-limit; records violation step}) \\[1mm]
0, & \rho \le s \\[1mm]
H\,e^{-\lambda k}, & s < \rho \le 1 \ \text{and}\ k \le R \quad (\text{recovery override}) \\[1mm]
e^{-5t} - 1,\ \ t = \dfrac{\rho - s}{1 - s}, & s < \rho \le 1 \ \text{and}\ k > R \quad (\text{warning zone})
\end{cases}
$$

The warning-zone term decays smoothly from $0$ (at $\rho = s$) to ≈ $-1$
(at $\rho = 1$). After an over-limit hit, the next $R$ steps are pinned to the
decaying recovery curve instead of the warning curve.

---

## 6. BatteryTargetRewardV0 — SoC dead-zone

File: `battery_target_reward_v0.py`. Pure penalty, per-step range `[-1, 0]`.

With SoC `c = s_battery_soc` and band `[min_pct, max_pct]`:

$$
r =
\begin{cases}
0, & \text{min\_pct} \le c \le \text{max\_pct} \\
-1, & \text{otherwise}
\end{cases}
$$

---

## 7. BatteryMgmtRewardV0 — terminal SoC-deficit

File: `battery_mgmt_reward_v0.py`. Sparse; fires only on the terminal step
(`info["terminated"]`). Per-fire range `[-L, 0]` where `L = episode_length`.
Asymmetric: only ending below the start SoC is penalised.

Capture the start SoC at reset: `c_0 = s_battery_soc`. At the terminal step
with end SoC `c_T = s_battery_soc` and `scale = σ` (default 1.0):

$$ \text{deficit} = \max(0,\; c_0 - c_T) $$

$$ r = -\,\mathrm{clip}\!\left(\frac{\text{deficit}}{\sigma},\,0,\,1\right) \cdot L $$

All non-terminal steps return `0`.

---

## 8. EVChargingRewardV0 — dense shaping + sparse session verdicts

File: `ev_charging_reward_v0.py`. Mixed dense/sparse. Maintains a counter
`m` = number of connected steps in the current session.

**Disconnected** (and not the just-disconnected step): `0`.

**Disconnect step** (`info["ev_just_disconnected"]`): let session magnitude
$M = \max(m, 1)$ and the session target $\tau = $ `info["ev_session_target_soc"]`
(note: from `info`, not the live `s_evc_target_soc`). Success
$= |s_{\text{ev\_soc}} - \tau| \le \text{disconnect\_soc\_tolerance}$:

$$ r = \begin{cases} +M & \text{success} \\ -M & \text{failure} \end{cases} $$

then `m ← 0`. (Constructor overrides `success_reward` / `failure_penalty`
replace $\pm M$ when provided.)

**Min-curve violation** (session active and `s_evc_soc < s_evc_soc_min`):

$$ r = -M $$

(or `min_curve_violation_penalty` override).

**Regular connected step** (dense, `[0, 1]`): increment `m`, with
`soc_diff = |s_evc_soc - s_evc_target_soc|`, `diff_threshold = δ` (0.02),
`soc_diff_multiplier = μ` (5.0):

$$
r =
\begin{cases}
1, & \text{soc\_diff} < \delta \\
e^{-\mu \cdot \text{soc\_diff}}, & \text{otherwise}
\end{cases}
$$

---

## 9. EVChargingOnTimeRewardV0 — on-time charging feasibility

File: `ev_charging_ontime_reward_v0.py`. Per-step range `[-1, 1]`.

- EV not connected (`s_evc_connected < 0.5`): `0`.
- Target met (`s_evc_soc ≥ s_evc_target_soc`): `r = +1`.

Otherwise compute the energy needed vs. the energy still achievable in the
remaining window. With remaining time
`t_rem = s_evc_charge_to_target_hrs_norm · ctxt_evc_max_charge_time_hrs`:

$$ E_{\text{need}} = (\text{target} - \text{soc}) \cdot \text{ctxt\_ev\_max\_cap\_kWh} $$

$$ E_{\text{able}} = \text{ctxt\_ev\_max\_charging\_kW} \cdot \text{ctxt\_ev\_charger\_efficiency} \cdot t_{\text{rem}} $$

- If $E_{\text{able}} \le 0$ (no time left, target unmet):
  $r = \text{harsh\_penalty}$ (default $-1$).
- Otherwise, with ratio $= E_{\text{need}}/E_{\text{able}}$:

$$
r =
\begin{cases}
\max(0,\; 1 - \text{ratio}), & \text{if charging } (a_{\text{lin\_ev\_charger}} > 0) \\
0, & \text{otherwise}
\end{cases}
$$

So the agent is rewarded for staying feasible (need ≪ achievable) **only when
it is actually drawing charge**.

---

## 10. ActionSmoothnessRewardV0 — spectral oscillation penalty

File: `action_smoothness_reward_v0.py` — a thin alias of
`ActionSmoothnessReward` (`action_smoothness_reward.py`). Pure penalty,
range `[-n_keys, 0]` (n_keys = number of action keys).

For each action key, a length-`N` ring buffer (`n_steps`, default 16) holds the
recent actions. Per dimension $d$, mean-subtract the window $x$, take the real
FFT $X = \mathrm{rfft}(x)$ giving bins $k = 0\ldots K$ with $K = N/2$, and form
the per-bin energy $E_k = |X_k|^2$.

**Weighted mode** (default), with frequency weights
$w_k = (k/K)^{p}$ (`freq_exponent` $p$, default 1; $w_0 = 0$, $w_K = 1$):

$$ \text{energy}_d = \sum_{k=0}^{K} w_k \, E_{k,d} $$

**Highband mode**, cutoff bin $c = \text{round}(\text{cutoff\_fraction}\cdot K)$:

$$ \text{energy}_d = \sum_{k=c+1}^{K} E_{k,d} $$

Normalise by the analytical worst case $N^2$ (the $\pm1$ square wave puts all
energy at Nyquist), average over dimensions, and clip:

$$ \text{penalty}_{\text{key}} = \mathrm{clip}\!\left(\frac{1}{D}\sum_{d=1}^{D} \frac{\text{energy}_d}{N^2},\; 0,\; 1\right) $$

Sum the per-key penalties and negate:

$$ r = -\sum_{\text{keys}} \text{penalty}_{\text{key}} \;\in\; [-n_{\text{keys}},\, 0] $$

Reference: NumPy `rfft` and Parseval's theorem; see
`docs/action_smoothness_spectral.md` for the full derivation.
