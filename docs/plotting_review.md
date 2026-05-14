# Plotting Module — Code Review

**Date:** 2026-03-14
**Scope:** `plotting/` directory (top-level package + `plotting/src/` modular subpackage)

---

## 1. Architecture Overview

```
plotting/
├── __init__.py              # Public API re-exports from src/
├── __main__.py              # CLI entry point: python -m plotting.traj_plotting
├── traj_plot_config.yaml         # Domain-specific settings (groupings, labels, skip keys)
├── PLOTTING_README.md
└── src/
    ├── __init__.py          # Modular API exports
    ├── utils.py             # EpisodeData, loading, discovery, styling, config loader
    ├── trajectory_plot.py   # Orchestration + CLI (generate_all_plots, main)
    ├── plot_states.py       # State variable plots
    ├── plot_actions.py      # Action dimension plots
    ├── plot_rewards.py      # Reward breakdown plot
    └── plot_energy.py       # Energy/power dual-axis plot
```

Single entry point: `python -m plotting.traj_plotting`. Public Python API via `from plotting import ...`.
All plot functions accept an `EpisodeData` dataclass loaded by `load_episode()`.
All plot functions return `list[go.Figure]` uniformly.
Domain-specific settings (state groupings, skip keys, action dimension labels) are
centralised in `traj_plot_config.yaml`.

---

## 2. Open Issues

### 2.1 `_REPO_ROOT` computed via relative parent traversal

**Severity: Medium**

```python
# utils.py:17
_REPO_ROOT = Path(__file__).resolve().parents[2]
```

Fragile if the module is installed as a package, moved, or symlinked — `parents[2]` silently points to the wrong directory.

**Recommendation:** Accept `metrics_root` and `output_dir` as explicit arguments (already supported). Remove the computed defaults or gate them behind an environment variable (`ADV_BUILDING_GYM_ROOT`).

---

## 3. Design Weaknesses

### 3.1 24-hour assumption baked into x-axis

`apply_day_xaxis()` hardcodes a `[0, 1440]` minute range with 5-minute ticks. Episodes shorter or longer than 24 hours render incorrectly — short episodes have mostly empty whitespace; multi-day episodes are clipped.

```python
# utils.py:247
tick_vals = list(range(0, 1441, 5))  # always 0-1440 min
```

**Recommendation:** Derive the range from `episode.time_minutes` (min/max). Keep the 5-min tick spacing but compute `tick_vals` dynamically.

### 3.2 No y-axis labels on state/action plots

State and action plots have no `yaxis_title`. The reward and energy plots do. Action plots like `HP_action [energy]` would benefit from a unit label (e.g., "normalised action").

---

## 4. Code Quality Issues

### 4.1 Bare `except Exception` in `ensure_chrome_for_kaleido`

```python
# utils.py:307
except Exception as exc:
    logger.warning(...)
```

The function silently continues, and the actual `write_image()` call later fails with a confusing Kaleido error instead of the original download error.

**Recommendation:** Let the exception propagate when the user explicitly requested static formats, and only warn-and-continue when static export is optional.

### 4.2 `customdata` as `list(zip(...))` creates large Python lists

```python
# plot_energy.py:27
net_custom = list(zip(time_hhmm, power, cum_e))
```

For 288-step episodes this is negligible, but Plotly accepts NumPy arrays directly for `customdata`. A 2D array would be more memory-efficient and consistent with the codebase's NumPy preference.

### 4.3 `time_hhmm` property recomputed on every access

```python
# utils.py:74-78
@property
def time_hhmm(self) -> list[str]:
    return [f"{int(m) // 60:02d}:{int(m) % 60:02d}" for m in self.time_minutes]
```

Called once per plot function (4 times total). A `@functools.cached_property` would be cleaner.

---

## 5. Missing Functionality

### 5.1 No multi-episode comparison

The module plots one episode at a time. No way to overlay two episodes (e.g., best vs worst, or same scenario with different seeds) for visual comparison.

### 5.2 No automated integration with evaluation pipeline

The plotting module is standalone — it must be invoked manually after evaluation. No hook in the evaluation callback triggers plot generation automatically.

### 5.3 No error handling for malformed HDF5 data

`load_episode()` assumes the HDF5 schema is correct. Missing groups (`trajectory`, `summary`) or missing datasets (`step`, `reward`) raise raw `KeyError` / `h5py` exceptions with no context.

**Recommendation:** Add a brief schema validation step and raise a descriptive error.

### 5.4 No unit tests

No tests for any plotting function. At minimum, `load_episode()` and the individual plot functions should have smoke tests with synthetic `EpisodeData` to catch regressions.

---

## 6. Summary Table

| # | Issue | Severity | Effort |
|---|-------|----------|--------|
| 2.1 | Fragile `_REPO_ROOT` via `parents[]` | Medium | Low |
| 3.1 | 24-hour x-axis assumption | Medium | Low |
| 3.2 | Missing y-axis labels | Low | Low |
| 5.1 | No multi-episode comparison | Feature gap | Medium |
| 5.3 | No HDF5 schema validation | Medium | Low |
| 5.4 | No unit tests | Medium | Medium |

**Priority order:** 3.1 → 5.3 → 5.4 → rest.
