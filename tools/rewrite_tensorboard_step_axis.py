"""Rewrite finished Tune runs' TensorBoard scalars with env steps as global_step.

Existing event files were written with global_step = training_iteration: Tune's TBX
logger uses ``timesteps_total`` as global_step and falls back to ``training_iteration``,
and the new RLlib API stack no longer fills ``timesteps_total``. That makes the STEP
axis algorithm-dependent (one SAC iteration != one PPO iteration in env steps), so
runs cannot be compared. This script replays each trial's result.json rows into a
fresh event file whose global_step is ``num_env_steps_sampled_lifetime``.
Link: ray/tune/logger/tensorboardx.py (TBXLoggerCallback.log_trial_result)

The original trial dir is left untouched; the rewritten events go to a sibling run
directory ``<trial_dir>_<suffix>/`` which TensorBoard lists next to the original.
Re-running replaces the previously generated sibling events (idempotent).

Usage:
    python tools/rewrite_tensorboard_step_axis.py snapshots/20260524_002820_sta_temp_only/
    python tools/rewrite_tensorboard_step_axis.py snapshots/<snap>/runs/train_<ts>/models/<trial_name>
    python tools/rewrite_tensorboard_step_axis.py models/<trial>/ray/<algo>/<run>/<trial_id>/
    python tools/rewrite_tensorboard_step_axis.py <dir> --suffix env_steps

Accepted paths: a Tune trial dir (contains result.json), a snapshot root (searched
under runs/*/models/), or any parent dir (searched recursively for result.json).
"""

import argparse
import json
import logging
import math
import time
from pathlib import Path

from ray.tune.result import TIMESTEPS_TOTAL, TIME_TOTAL_S, TRAINING_ITERATION
from ray.tune.utils import flatten_dict

logger = logging.getLogger(__name__)

ENV_STEP_KEY = "num_env_steps_sampled_lifetime"
# Same removal list as TBXLoggerCallback.log_trial_result (parity with live logging).
# Link: ray/tune/logger/tensorboardx.py
DROPPED_KEYS = ("config", "pid", "timestamp", TIME_TOTAL_S, TRAINING_ITERATION)
TAG_PREFIX = "ray/tune/"


def iter_result_rows(result_json: Path):
    """Yield parsed result.json rows, skipping unparsable lines (e.g. a truncated tail)."""
    with result_json.open() as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                logger.warning("%s line %d: unparsable row skipped", result_json, line_number)


def prepare_out_dir(trial_dir: Path, suffix: str) -> Path:
    """Return the sibling run dir ``<trial_dir>_<suffix>/``, dropping events from a previous run.

    The sibling dir holds nothing but generated event files, so re-running the script
    replaces them instead of appending a second event file (TensorBoard would merge both).
    """
    out_dir = trial_dir.parent / f"{trial_dir.name}_{suffix}"
    for stale_event_file in out_dir.glob("events.out.tfevents.*"):
        stale_event_file.unlink()
        logger.info("%s: replaced previously generated events", out_dir)
    return out_dir


def rewrite_trial(trial_dir: Path, suffix: str) -> int:
    """Replay one trial's result.json into the ``<trial_dir>_<suffix>/`` sibling run.

    All scalars of one result row are batched into a single multi-value TensorBoard
    event (one event per iteration instead of one per tag) — per-tag add_scalar is
    far too slow for thousands of rows x hundreds of tags on a network filesystem.
    """
    from tensorboardX.event_file_writer import EventFileWriter
    from tensorboardX.proto.event_pb2 import Event
    from tensorboardX.proto.summary_pb2 import Summary

    result_json = trial_dir / "result.json"
    out_dir = prepare_out_dir(trial_dir, suffix)
    writer = EventFileWriter(str(out_dir))
    rows_written = 0
    try:
        for row in iter_result_rows(result_json):
            env_steps = row.get(ENV_STEP_KEY)
            if env_steps is None:
                env_steps = row.get("env_runners", {}).get(ENV_STEP_KEY)
            if env_steps is None:
                logger.warning("%s: row without %s skipped", result_json, ENV_STEP_KEY)
                continue
            step = int(env_steps)
            wall_time = float(row.get("timestamp") or time.time())

            row = {k: v for k, v in row.items() if k not in DROPPED_KEYS}
            row[TIMESTEPS_TOTAL] = step  # parity with the live env-step-axis hook
            summary_values = [
                Summary.Value(tag=TAG_PREFIX + attr, simple_value=float(value))
                for attr, value in flatten_dict(row, delimiter="/").items()
                if isinstance(value, (int, float)) and not (isinstance(value, float) and math.isnan(value))
            ]
            writer.add_event(Event(wall_time=wall_time, step=step, summary=Summary(value=summary_values)))
            rows_written += 1
    finally:
        writer.close()
    logger.info("%s: %d rows → %s (last step %s)", trial_dir, rows_written, out_dir, step if rows_written else "n/a")
    return rows_written


def discover_trial_dirs(paths: list[Path]) -> list[Path]:
    """Resolve each path to Tune trial dirs (dirs containing result.json), deduplicated.

    A snapshot root (recognised by its runs/ subdir) is searched only under
    runs/*/models/ — the bundled code/ and data/ copies are never traversed.
    """
    trial_dirs: dict[Path, None] = {}
    for path in paths:
        if (path / "result.json").is_file():
            found = [path]
        elif (path / "runs").is_dir():
            found = sorted(p.parent for p in path.glob("runs/*/models/**/result.json"))
        else:
            found = sorted(p.parent for p in path.rglob("result.json"))
        if not found:
            logger.warning("%s: no result.json found underneath, skipped", path)
        trial_dirs.update(dict.fromkeys(p.resolve() for p in found))
    return list(trial_dirs)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("paths", nargs="+", type=Path,
                        help="trial dirs (containing result.json), snapshot roots, or parent dirs to search")
    parser.add_argument("--suffix", default="env_steps",
                        help="suffix for the sibling run dir created next to each trial dir")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    trial_dirs = discover_trial_dirs(args.paths)
    if not trial_dirs:
        parser.error("no result.json found under the given paths")
    for trial_dir in trial_dirs:
        rewrite_trial(trial_dir, args.suffix)


if __name__ == "__main__":
    main()
