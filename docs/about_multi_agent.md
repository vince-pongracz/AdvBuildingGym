data_schedule_callback.py#15-16
Analyse the selected todo lines
Read data_schedule_callback.py
Read data_combinator.py
Bash find day offset usage
IN
grep -rn "get_day_offset\|apply_data_variant\|row_offset" /home/iai/dj0397/AdvBuildingGym/adv_building_gym --include="*.py" | head -50

OUT
/home/iai/dj0397/AdvBuildingGym/adv_building_gym/callbacks/data_schedule_callback.py:79:                unwrapped.apply_data_variant(variant)
/home/iai/dj0397/AdvBuildingGym/adv_building_gym/envs/building_adv.py:337:    def apply_data_variant(self, variant: dict[str, str]) -> None:
/home/iai/dj0397/AdvBuildingGym/adv_building_gym/envs/building_adv.py:371:            self.apply_data_variant(variant)
/home/iai/dj0397/AdvBuildingGym/adv_building_gym/envs/building_adv.py:375:            self.apply_data_variant(variant)
/home/iai/dj0397/AdvBuildingGym/adv_building_gym/envs/building_adv.py:379:        row_offset = 0
/home/iai/dj0397/AdvBuildingGym/adv_building_gym/envs/building_adv.py:392:        row_offset, self._episode_day_mode = self.data_combinator.get_day_offset(
/home/iai/dj0397/AdvBuildingGym/adv_building_gym/envs/building_adv.py:395:        self._episode_date = resolve_episode_date(self.statesources, row_offset, self.control_step)
/home/iai/dj0397/AdvBuildingGym/adv_building_gym/envs/building_adv.py:401:        if options and "row_offset" in options:
/home/iai/dj0397/AdvBuildingGym/adv_building_gym/envs/building_adv.py:402:            row_offset = int(options["row_offset"])
/home/iai/dj0397/AdvBuildingGym/adv_building_gym/envs/building_adv.py:403:            self._episode_date = resolve_episode_date(self.statesources, row_offset, self.control_step)
/home/iai/dj0397/AdvBuildingGym/adv_building_gym/envs/building_adv.py:411:            sync.synchronise(self.iteration, row_offset)
/home/iai/dj0397/AdvBuildingGym/adv_building_gym/envs/data_variant.py:8:    def apply_data_variant(self, variant: dict[str, str]) -> None:
/home/iai/dj0397/AdvBuildingGym/adv_building_gym/utils/env_sync.py:13:        self.row_offset = 0
/home/iai/dj0397/AdvBuildingGym/adv_building_gym/utils/env_sync.py:15:    def synchronise(self, iteration: int, row_offset: int | None = None) -> None:
/home/iai/dj0397/AdvBuildingGym/adv_building_gym/utils/env_sync.py:18:        if row_offset is not None:
/home/iai/dj0397/AdvBuildingGym/adv_building_gym/utils/env_sync.py:19:            self.row_offset = row_offset
/home/iai/dj0397/AdvBuildingGym/adv_building_gym/utils/env_sync.py:23:        """Row index into time-series data: row_offset + iteration."""
/home/iai/dj0397/AdvBuildingGym/adv_building_gym/utils/env_sync.py:24:        return self.row_offset + self.iteration
/home/iai/dj0397/AdvBuildingGym/adv_building_gym/utils/episode_date.py:15:    row_offset: int,
/home/iai/dj0397/AdvBuildingGym/adv_building_gym/utils/episode_date.py:18:    """Derive a date string from *row_offset* using the first statesource with a ``start`` column.
/home/iai/dj0397/AdvBuildingGym/adv_building_gym/utils/episode_date.py:22:        row_offset: Row index into the underlying timeseries.
/home/iai/dj0397/AdvBuildingGym/adv_building_gym/utils/episode_date.py:29:        if src.ts is not None and "start" in src.ts.columns and row_offset < len(src.ts):
/home/iai/dj0397/AdvBuildingGym/adv_building_gym/utils/episode_date.py:30:            return str(pd.to_datetime(src.ts.iloc[row_offset]["start"]).date())
/home/iai/dj0397/AdvBuildingGym/adv_building_gym/utils/episode_date.py:33:    return f"day-{row_offset // steps_per_day}"
/home/iai/dj0397/AdvBuildingGym/adv_building_gym/devices/infrastructure/base.py:22:    _exclude_params: ClassVar[Set[str]] = {'iteration', 'row_offset'}
/home/iai/dj0397/AdvBuildingGym/adv_building_gym/devices/statesources/outer/inside_temperature.py:76:            # the actual simulation date or row_offset.
/home/iai/dj0397/AdvBuildingGym/adv_building_gym/devices/statesources/base.py:25:    _exclude_params: ClassVar[Set[str]] = {'iteration', 'row_offset', 'ts', '_last_processed_ds_path'}
/home/iai/dj0397/AdvBuildingGym/adv_building_gym/data_combinator/data_combinator.py:122:    def get_day_offset(
/home/iai/dj0397/AdvBuildingGym/adv_building_gym/data_combinator/data_combinator.py:148:            (row_offset, day_mode) where row_offset is the starting row and
/home/iai/dj0397/AdvBuildingGym/adv_building_gym/data_combinator/data_combinator.py:166:        row_offset = day_index * steps_per_day
/home/iai/dj0397/AdvBuildingGym/adv_building_gym/data_combinator/data_combinator.py:167:        return row_offset, self.day
Read building_adv.py (lines 331-420)
Analysis of the TODO at data_schedule_callback.py:15-16
The TODO asks two questions. Both have answers in the existing code — the day pick is not part of the variant push; it lives entirely on the env side at reset() time.

Q1: How is the day picked within a chosen variant?
The callback only picks a variant (year-long CSV bundle) and pushes it to all runners — it never touches the day. Day selection happens locally in each env's reset():

building_adv.py:392-394 calls self.data_combinator.get_day_offset(self.episode_count, max_days, steps_per_day, data_start_year, self._rng) every reset.
data_combinator.py:151-167 implements three modes driven by the day: field in the data YAML:
"random" → rng.integers(0, max_days) per episode (uses the env's self._rng, so each runner picks a different day even with the same variant).
"each" → episode_count % max_days (sequential walk through the year).
date string (e.g. "2025-03-15") → fixed day index every episode.
The result is a row_offset = day_index * steps_per_day (line 166) which gets fed into every component via sync.synchronise(self.iteration, row_offset) at building_adv.py:411. EnvSync.effective_index = row_offset + iteration indexes into the CSV.
So: variant = which CSV; day = where in that CSV the episode starts. Independent axes, separately scheduled.

Q2: What if the episode is longer than a day?
It works automatically — effective_index just keeps incrementing past the day boundary into the next day's rows. The CSVs are full-year time series, so reading past steps_per_day simply continues into the next calendar day.

The caveats are:

steps_per_day is hardcoded to env_config.EPISODE_LENGTH at building_adv.py:378 (with a self-acknowledging comment "Assuming 1 day per episode; adjust if multiple days per episode"). For a 2-day episode at 5-min steps, EPISODE_LENGTH=576 would make steps_per_day=576 and max_days = len(ts)//576 — so day_index would actually mean "2-day-block index", not "calendar day". The semantics still work (no overrun), but the variable name is misleading and get_day_date() would label the start incorrectly because data_combinator.py:163-164 computes jan1 + day_index days.
For day="random" with multi-day episodes, the random sampling is over max_days = len(ts) // EPISODE_LENGTH blocks rather than over individual calendar days — episodes never straddle block boundaries, so you can't randomly land mid-week.
The _date_to_day_index path (line 176-179) computes day_of_year - 1 and then multiplies by steps_per_day=EPISODE_LENGTH. For a fixed-date episode longer than 1 day, that offset is wrong (it treats day-of-year as block index).
Suggested resolution of the TODO
Replace the two TODO lines with a docstring note such as:

Day selection within a variant is performed independently by each env's reset() via DataCombinator.get_day_offset() — this callback only pushes the variant (CSV bundle). Episodes longer than one day read consecutively past the day boundary; if EPISODE_LENGTH exceeds one calendar day, steps_per_day in building_adv.reset() should be decoupled from EPISODE_LENGTH so day indexing remains calendar-aligned.

data_schedule_callback.py#25-28
Analyse and eval this next todo:
Bash check weather statesource context publishing
IN
grep -n "ctxt_\|reload\|_post_load\|abs_max\|scale" /home/iai/dj0397/AdvBuildingGym/adv_building_gym/devices/statesources/outer/weather.py | head -40

OUT
11:from adv_building_gym.utils.normalisation import Normalisation, normalise_with_scale_factor, normalise_series
23:        'temp_abs_max', 'temp_out_raw',
24:        'wind_speed_abs_max', 'wind_speed_raw',
35:        # Scale factors derived from data in _post_load_data_processing
36:        self.temp_abs_max: float = 1.0
37:        self.wind_speed_abs_max: float = 1.0
41:            self._run_post_load()
45:    def _post_load_data_processing(self) -> None:
46:        """Normalise weather columns after CSV load / reload.
53:        # on every reload of the same file, which happens each episode reset).
76:        # Normalise raw columns and derive scale factors so downstream
79:            "temp_amb": ("s_temp_out_norm", "temp_abs_max"),
81:            "avg_wind_speed": ("s_avg_wind_speed_norm", "wind_speed_abs_max"),
84:        for raw_col, (norm_col, scale_attr) in cols.items():
86:                normalised, scale_factor = normalise_with_scale_factor(self.ts[raw_col], self.normalise)
88:                if scale_attr is not None:
89:                    setattr(self, scale_attr, scale_factor)
104:        # Raw scale factors — set once when data is loaded, not every step.
106:        # normalised observations (e.g. temp_out_raw = temp_out_norm * temp_abs_max).
107:        if "ctxt_temp_abs_max" not in state_spaces.keys():
108:            state_spaces["ctxt_temp_abs_max"] = Box(
111:        if "ctxt_wind_speed_abs_max" not in state_spaces.keys():
112:            state_spaces["ctxt_wind_speed_abs_max"] = Box(
158:            self.temp_out_raw = temp_out_norm * self.temp_abs_max
163:            self.wind_speed_raw = avg_wind_speed_norm * self.wind_speed_abs_max
170:        # Raw scale factors — constant within an episode, change only when
171:        # a new data variant is loaded (via _post_load_data_processing).
172:        states["ctxt_temp_abs_max"][0] = np.float32(self.temp_abs_max)
173:        states["ctxt_wind_speed_abs_max"][0] = np.float32(self.wind_speed_abs_max)
Analysis of the TODO at data_schedule_callback.py:25-28
The user's hypotheses are largely right. Walking through each question:

1. "Are the eval workers updated when training workers are updated?"
Yes. data_schedule_callback.py:81-84 iterates both groups:


algorithm.env_runner_group.foreach_env_runner(apply, ...)
if algorithm.eval_env_runner_group is not None:
    algorithm.eval_env_runner_group.foreach_env_runner(apply, ...)
So eval and training share the variant pool advance at every iteration boundary.

2. "Is it the normalisation that requires synchronous swap?"
No — the user's reasoning here is correct. Per-variant scale factors are published as ctxt_* observation keys (weather.py:172-173: ctxt_temp_abs_max, ctxt_wind_speed_abs_max), so the policy is conditioned on the variant's scale and can in principle handle mixed-variant batches. The "simultaneous" wording is not enforced by normalisation.

3. "So is it only the eval call that needs alignment?"
That is the strongest remaining reason, but it's not the only one. Plausible motivations to keep the swap iteration-aligned:

Eval semantics during training. With evaluation_interval=4 and evaluation_duration=2, eval episodes report performance on whatever variant happens to be loaded. If training and eval workers ran independent schedules, the eval metric would reflect a different distribution than the iteration's training data — making per-iteration TensorBoard curves much harder to read.
Curriculum coupling. The reward (reward_switch_callback) and infra (infra_schedule_callback) schedules also fire on on_train_result. Aligning data swaps with them keeps experiment logs interpretable as "iteration N had reward set X, infra Y, data variant Z" rather than a moving cocktail across runners.
Per-iteration metric averaging. metrics_num_episodes_for_smoothing=25 and keep_per_episode_custom_metrics=True aggregate across runners. Mixed variants within one iteration just average them away; aligned variants give clean per-variant curves.
PPO/SAC math: neither algorithm requires synchrony — PPO's on-policy batch and SAC's replay both accept mixed distributions fine. So this is a logging/interpretability constraint, not a correctness constraint.
4. "What if eval picks a random year/day?"
This is the case that motivated the user's question and it has a real subtlety:

Year (variant) is synchronised between training and eval by the callback's dual push.
Day is not synchronised. Day selection happens independently in each env's reset() via DataCombinator.get_day_offset() using each env's local self._rng (building_adv.py:392-394). With day: "random", training runners and eval runners pick different days within the same variant. So "simultaneous" really means "same year-long CSV", not "same episode".
This is probably fine — you usually want eval to sample diverse days within a held-out scenario. But the wording in the docstring at line 4 ("see the same dataset at the same time") is misleading: workers see the same CSV bundle, not the same day-window.

5. A separate concern worth flagging
The callback pushes the training combinator's variant to the eval group as well. If a user passes --data-config configs/data_scheduler/eval_data_combinator_config.yaml-style held-out data for eval, that intent is not honoured during in-training eval — the eval workers get re-pointed at the training combinator's pool every iteration. The held-out config is only respected by the standalone run_eval_ray.py path, not by evaluation_interval evals.

Suggested resolution
Replace the TODO with something like:

Iteration-aligned swap is required for interpretability, not correctness: it keeps in-training eval, reward/infra curricula, and per-iteration TensorBoard curves aligned to the same variant. The policy is conditioned on per-variant scale factors via ctxt_* keys, so PPO/SAC themselves do not require synchronous variants. Day selection within a variant remains independent per env — workers share the CSV bundle, not the calendar day. In-training eval workers are pushed the training variant, not a held-out one; held-out eval lives in run_eval_ray.py.

data_schedule_callback.py
Replace the todo
Read data_schedule_callback.py (lines 2-36)
Edit data_schedule_callback.py
Added 6 lines
Replaced the TODO at lines 25-31 with a NOTE block summarizing the analysis: synchronous swap is for interpretability (eval/curriculum/metric alignment), not algorithm correctness; eval env_runners are updated alongside training; only the variant is synchronised (not the day); in-training eval uses the training combinator, not a held-out one.

data_schedule_callback.py
Analyse the scenario: what if the env would be recreated at each change/swap? -- analyse benefits and drawbacks. Wouldn't it be easier?
Recreate the env on each swap — would it be easier?
Short answer: Cleaner conceptually, but markedly worse in practice for this codebase. The current reload() pattern is the right default; full recreation would buy a small simplicity win and pay several large operational costs.

Benefits of recreation
No stale state. Every cached attribute (scale factors, normalised columns, EV deques, _last_processed_ds_path) is freshly constructed. reload() has to carefully re-run _post_load_data_processing() to avoid cross-variant bleed; recreation makes that bug class structurally impossible.
One construction path. EnvConfig.create_infras() / create_statesources() already exist and are the canonical "build a worker env" code. Recreation collapses "init" and "swap" into the same path — no DataVariantProvider interface, no statesource-by-name matching, no introspection walking DictInfoToList → SyncVectorEnv → sub_envs in _publish_variant_to_runners.
Symmetric with the infra swap mental model. The infra schedule already does a partial "recreate the components, keep the shell" via env.set_infras(...). Going the rest of the way (recreate the shell too) is conceptually uniform.
Drawbacks of recreation
RLlib bindings break. In the new API stack the EnvRunner owns SyncVectorEnv, and the connector pipeline + RLModule are bound to the env's observation_space / action_space at build time. Replacing env_runner.env from a callback is poking at private state with much higher blast radius than the current attribute-level mutation, and any drift in spaces (different infra set, different obs keys) silently mismatches the policy network.
Episode boundary handling. Env can only be replaced safely between episodes. The current path lands the swap inside reset() naturally. A recreate-on-swap callback would either need a wait-until-episode-end queue or would force a truncation — adding a non-Markovian seam right where the policy tries to learn.
Cost per swap. CSV parsing, statesource __init__, connector spec inference — paid by every runner on every swap. reload() is essentially path swap + one DataFrame reprocess; recreation re-does the whole pipeline for what is functionally a column update.
Episode counter / RNG resets. episode_count, action_history, _rng, the RngService per-caller streams — all reset on recreation. The data combinator uses episode_count // swap_every_n_episodes and the day-selection rng; resetting those defeats the schedule that triggered the recreation in the first place. You'd have to manually re-seed and re-thread counters across the rebuild.
Composition with the other curricula. reward_switch_callback and infra_schedule_callback mutate the same env via env.set_reward_funcs(...) / env.set_infras(...). If the data callback also recreated the env, ordering becomes load-bearing: you'd need a strict on_train_result sequence (recreate → re-set rewards → re-set infras) or you'd lose the other two schedules' state on every data swap.
Approach A coexistence is gone. The docstring's "Approach A" (env owns its own combinator and may swap on its own resets) only works because the env survives the swap. Forced recreation outside the env makes env-side swapping impossible.
Race surface in Ray. foreach_env_runner running a function that tears down env_runner.env races with the runner's sample loop. The current callback only mutates leaf attributes — structurally inert.
Where recreation would actually help
Eval / standalone runs (run_eval_ray.py) — recreation is fine there because runner ownership is shallow and there's no concurrent training. In fact, the standalone eval path effectively does this already by building the env from scratch per checkpoint.
Debugging variant-leak bugs — if you ever suspect that a _post_load_data_processing cache is bleeding across variants, building one env per variant is a clean isolation harness for a smoke test.
If spaces actually do change with the variant (different infra set, different obs keys) — then reload() cannot do it and you must recreate. Today nothing in the data scheduler implies this; if a future variant carries an infra delta, push that through the infra schedule, not the data schedule.
Recommendation
Keep the current reload-on-reset model for the data schedule. The "easier" feeling of recreation comes from imagining a single env in isolation; once you account for the new-API-stack runner/connector binding, the curriculum callback composition, and the episode-boundary constraint, in-place reload is the simpler mechanism in context.

data_schedule_callback.py
So the same AdvBuildingEnv, but with different underlying data source csv-s, rewards, etc..
Exactly — that's the design. One persistent AdvBuildingGym instance per worker, with three independent mutation paths firing on on_train_result:

Schedule	Mutation point	What survives	What changes
Data	state_src.reload(new_csv) via apply_data_variant	env, infras, statesource objects, spaces, RLModule, connectors	CSV-backed ts DataFrame inside each statesource + cached scale factors (temp_abs_max, …) re-derived in _post_load_data_processing
Reward	env.set_reward_funcs(new_list) from reward_switch_callback	env, infras, statesources, spaces	the reward_funcs list (RewardFunction instances rebuilt by RewardScheduleManager); reward scaling but not observation/action spaces
Infra	env.set_infras(new_list, building_props=…) from infra_schedule_callback	env shell, statesources	infra component instances + BuildingProps (so K, mC); action keys may change
The invariant that makes this safe: the policy network is bound to the env's observation_space and action_space at build time, so any mutation that would change those spaces is risky. Today:

Data swap — guaranteed space-preserving (only ts mutates).
Reward swap — space-preserving (rewards don't contribute to obs/action spaces).
Infra swap — can change action keys. This is why the YAML-driven infra schedule is constrained to topologies the policy was trained against; otherwise the connector pipeline + RLModule would silently mismatch.
So the rule of thumb in this codebase: mutate leaves, never the shell. The shell (env object, spaces, RLModule binding, connector pipeline) is the contract with RLlib — once built, it stays. Variant/reward/infra schedules all respect that.

data_schedule_callback.py
To the drawbacks section: 1. The infra set and observation keys would remain the same. 
2. Env replace would be initiated within an episode_end callback.. 
4. Episode counter is per env, right? How could a ray actor mitigate this problem? 
5. those switches and schedulings should be also refactored to the recreate env based approach. 
7. I do not understand this. 
If I keep the current reload on reset based approach, how can I implement later such 
training cases where for example multiple PPO-s, SAC-s train on the same environment 
but for example with different reward sets? -- so they all produce an action, which is
then somehow aggregated and applied to the env, 
which then produces a single next state and reward, 
which is fed back to all the PPO-s and SAC-s?
At the created table, reward schedule, what survives: RLModule and connectors survive? Table indicates not: Why not?
I completely aggree with the last conclusion, the shell must survive, subcomponents can change -- but still, how to resolve such a scenario, where multiple agents (multiple PPOs or SACs with different rewards -- so with different goal) train together on the same env?
Going point by point, then the multi-agent question (which is the interesting one):

Drawback rebuttals
Infra/obs keys stable. Granted — under the assumption "the variant carries no obs-space delta," the recreation path is safe. That's a real precondition though, not a free property: EnvConfig.create_statesources() would need a contract test asserting observation_space equality before/after, otherwise you get silent connector mismatches at runtime instead of a build-time error.

Recreate at episode_end, not on_train_result. Better, yes — that fully removes the mid-episode tear-down concern. The natural hook would be inside reset() itself (when a "recreate me" flag is set by a callback), but at that point you're really asking the env to rebuild its own components — which is what set_infras() / set_reward_funcs() / apply_data_variant() already do. The hook exists; "recreate" mostly means "extend it to recreate statesources too." That's a small, safe diff, not a structural redesign.

Episode counter is per env. Correct — each AdvBuildingGym instance owns self.episode_count, incremented at reset(). A Ray Named Actor mitigates two things:

Global counter for the combinator (DataCombinator.get_variant(global_episode_count)) — but the data callback already does this implicitly: it advances by iteration not episode, and pushes one variant to all runners, so global progress is already centralised at the callback layer. You'd only need a Named Actor if you wanted env-driven (Approach A) global episode counting.
RNG continuity across recreations — RngService already exists and survives env recreation. The thing that doesn't survive is self._rng inside the env (used for day selection); reseeding it on rebuild from RngService.get().get_random("data_combinator_day") would close that gap.
So: the actor pattern handles it, but only if you commit to routing every stateful counter through it. That's a bigger refactor than reload.

Refactor all schedules to recreate-based. Agreed in principle — partial recreation (data only) is the worst of both. If you go recreate, go fully: data + reward + infra all go through one rebuild_env(new_data, new_rewards, new_infras) path. The benefit is one mutation primitive instead of three; the cost is that every iteration where any schedule fires pays the full reconstruction cost, even when only one axis changed.

Race surface — concede. I overstated this. on_train_result fires between iterations, so the sample loop is paused. The real residual risk is just structural mutation of env_runner.env reaching into RLlib internals — not a Ray-level race.

The table — do RLModule and connectors survive? Yes.
I should have listed them explicitly. Corrected version:

What	Data swap	Reward swap	Infra swap
EnvRunner, RLModule, connector pipeline	✅ survives	✅ survives	✅ survives
observation_space, action_space	✅ unchanged	✅ unchanged	⚠️ may change if action keys differ
AdvBuildingGym instance	✅ same object	✅ same object	✅ same object
infras list	unchanged	unchanged	replaced
reward_funcs list	unchanged	replaced	unchanged
statesources[i].ts (DataFrame)	replaced	unchanged	unchanged
The shell (top three rows) is the RLlib contract; everything below is mutable per worker.

The interesting question: multiple PPO/SAC on one env, different reward sets
This is orthogonal to reload-vs-recreate — both architectures support it equally. The right framing is a multi-agent setup, and there are two flavours; pick the one that matches your scientific question.

Flavour A — Multi-agent with disjoint action partitions (RLlib-native):
Each policy owns a subset of a_* keys, env returns a dict of rewards (one per agent). Implementation:

Convert AdvBuildingGym to MultiAgentEnv. Action space becomes {"agent_hp": Box(...), "agent_storage": Box(...)} etc.
RewardConfig becomes a dict of reward sets, one per agent. The env already computes a reward_breakdown dict in info["_component_info"] — that infrastructure is 80% of what you need.
Each agent gets its own RLModule (different algos: PPO for one, SAC for another, fine in MARL).
Single env step → state shared, per-agent rewards routed to per-agent learners.
This is the standard answer and what RLlib is designed for.

Flavour B — Same action space, multiple proposals aggregated (what you literally described):
N policies each propose a full action; an aggregator (mean / weighted vote / argmax-by-confidence) produces the executed action; env steps once; each policy receives the same observation but its own reward.

RLlib doesn't natively support "shared action, per-policy reward" because credit assignment becomes ambiguous (each policy is updated as if it produced the executed action — but it didn't, the aggregate did). Two options:

Wrap as MultiAgentEnv with a virtual aggregator agent. Each policy is an agent submitting a proposal; the env's step() aggregates and executes. The action space is the action space, replicated per agent. Per-agent reward is r_executed_under_policy_i_reward_function evaluated as if that policy's action had been taken — but you executed the aggregate, so this is off-policy for each individual learner and you need importance correction or accept the bias. SAC tolerates this better than PPO.
Centralised training, decentralised proposal (CTDE-flavoured). Train one shared backbone, multiple reward heads / Q-heads with different objectives; aggregate at action time. This is the "multi-objective RL" branch — papers like Envelope MOQ-Learning or PD-MORL. Code-wise it's a custom RLModule, not a callback change.
For your codebase specifically: Flavour A is reachable from where you are now with a moderate refactor (env → MultiAgentEnv, reward dict, action-space partition by infra owner). The _component_info["reward_breakdown"] dict and the per-infra action keys are already the right shape for it. Flavour B requires custom RLModule work regardless of how the env is built.

Conclusion on reload-vs-recreate in light of this: doesn't matter. Multi-agent-on-shared-env is a policy-side and reward-routing problem; the env-rebuild axis is independent. So the choice can be made on the simpler criterion (operational cost during training), which still favours reload.

How could be each of the current single agent, flavour A and flavour B.1 supported by the current architecture? Do I rather need different training scripts for that? I think flavour B.2 is not too flexible regarding the environment as the custom module has to track the infrastructure situation, while the current implementation easily supports such situation that the infrastructures are different -- that only means retraining, no RLModule rewrite.
Good instinct on B.2 — that's exactly the trade. Mapping each scenario onto the current code:

What stays, what changes per scenario
Layer	Single agent (today)	Flavour A (MARL, disjoint action partitions)	Flavour B.1 (MARL, aggregated proposals)	Flavour B.2 (multi-head RLModule)
AdvBuildingGym core	as-is	wrap as MultiAgentEnv; split Dict action by owner; obs shared	wrap as MultiAgentEnv; same action space per agent + aggregator inside step()	unchanged
Reward config (RewardConfig)	single list	dict agent_id → RewardConfig; env evaluates all, returns reward dict	dict agent_id → RewardConfig	single list, but consumed per head inside the module
Wrappers (FlattenAction + RescaleAction)	env-level	per-agent (RLlib applies to each policy's action space)	same	unchanged
select_model.py	one PPO/SAC config	config.multi_agent(policies={...}, policy_mapping_fn=...); each policy carries its own algo-specific config	same as A	one algo, but config.rl_module(rl_module_spec=CustomSpec)
Connectors (StridedHistoryConnector)	one pipeline, one hst_tracked_keys	per-policy pipeline, per-policy hst_tracked_keys (since connectors live on the learner side in the new API stack)	per-policy	unchanged
Callbacks (episode metrics, trajectory, eval state-action)	per-env	per-agent breakdowns; RewardScheduleManager.advance() needs to fire per agent	per-agent	unchanged, but heads must be logged separately
RewardScheduleManager	one schedule	one per agent — i.e. a dict[agent_id → RewardScheduleManager] and the reward_switch_callback becomes a fan-out	same	one schedule, but the module must support live-swap of head outputs
run_train_ray.py	as-is	needs MARL branch	needs MARL branch + aggregator hyperparameters	needs custom-module registration
One script with a flag, or sibling scripts?
Recommendation: sibling scripts — run_train_ray.py (single-agent), run_train_ray_marl.py (Flavours A and B.1), run_train_ray_morl.py (B.2). The cost of branching one script with --multi-agent and --aggregate-mode flags grows fast:

The reward-config YAML schema itself differs (single list vs. dict-of-lists) — flagging that inside one parser becomes ugly.
common_model_setup would need conditional config.multi_agent(...) calls and conditional per-policy connector building. That's where bugs hide — it's the file that's the contract with RLlib, and you don't want it to grow conditionals.
select_model.py already has algorithm branching; adding a MARL axis on top doubles the matrix.
Sibling scripts share the libraries (env, callbacks, components, data combinator, ComponentRegistry) and only the orchestration top differs. That's a clean split — same pattern as run_train_ray.py vs run_train_sb.py already in the repo. Each script is ~200 lines of orchestration; duplication cost is low.

The shared-library line is the right invariant: everything under adv_building_gym/ should be agnostic to single vs multi-agent. The MARL wrapper, the per-agent reward config, and the policy mapping live next to the new training script.

A practical staging:

A first. Refactor RewardConfig → dict[agent_id, RewardConfig] with single-agent being {"default": RewardConfig(...)}. Add a MultiAgentAdvBuildingGym thin wrapper. Sibling script. Most of the existing infrastructure (data scheduler, infra scheduler, episode metrics) generalises without rewrite.
B.1 builds on A. Same wrapper class with an extra aggregator: Callable[[dict[agent_id, action]], action] parameter. The interesting research knob (mean vs. weighted vs. learned aggregator) lives there.
B.2 is its own track. Custom MultiObjectiveRLModule + custom learner. Worth doing only if multi-objective is the actual research question; otherwise A handles "different goals, different policies" with infra-flexibility intact.
On B.2 and infra flexibility
Your concern is right and worth stating sharply: the current architecture's superpower is that the policy network is a generic MLP fed a Dict-of-keys observation, with infra topology expressed entirely through s_* / a_* / ctxt_* key sets. Swap infras → spaces change → train a fresh policy on the same backbone, no module-side code change. That property is preserved in single-agent and in Flavour A (each agent's policy is still a generic MLP over its key partition). It's preserved in B.1 too (each agent is a generic MLP). It's lost in B.2 because the multi-head module hard-codes head-to-objective mapping; if you add an EV charger and now want an EV-charging objective head, you edit the module class.

So the rule of thumb that falls out: keep the policy network generic, push specialisation into reward configs and per-agent partitioning. That's the design you already have for free; A and B.1 extend it without breaking it. B.2 trades it for tighter integration of multi-objective signals — only worth it if multi-objective optimality is the research target rather than a side-effect of having multiple controllers.