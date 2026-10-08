"""Iteration start / end log markers via driver-side callbacks.

RLlib has no dedicated iteration-start hook (verified against ray 2.52.1:
``AlgorithmConfig.callbacks()`` accepts only ``on_algorithm_init``, ``on_train_result``
and the evaluate/episode/env hooks), so the boundary is reconstructed from the two
hooks that do exist. ``on_train_result`` fires in ``Algorithm.log_result()`` right
after ``step()`` returns, and Tune calls ``train()`` back-to-back, hence:

- "Iteration N: end"       <- on_train_result (result["training_iteration"] == N)
- "Iteration 1: start"     <- on_algorithm_init
- "Iteration N+1: start"   <- on_train_result, logged after the end marker

The final start marker may announce an iteration that never runs (Tune checks the
stop criteria only after processing the result).
"""

import logging

logger = logging.getLogger(__name__)


def create_iter_start_logging_cb():
    """Factory returning a callable that logs the start of the upcoming training iteration.

    Register it under BOTH ``on_algorithm_init`` (logs "Iteration 1: start") and
    ``on_train_result`` (logs "Iteration N+1: start"); ``result`` is only passed by the latter.
    """

    def log_iteration_start(*, algorithm, result: dict | None = None, **kwargs) -> None:
        # Trainable.train() increments _iteration and stamps result["training_iteration"]
        # BEFORE log_result(), so both hooks see the number of completed iterations here.
        # Link: ray/tune/trainable/trainable.py (Trainable.train)
        completed_iterations: int = result.get("training_iteration", algorithm.iteration) if result is not None else algorithm.iteration
        logger.info("Iteration %d: start", completed_iterations + 1)

    return log_iteration_start


def create_iter_end_logging_on_train_result_cb():
    """Factory returning an ``on_train_result`` callable that logs the end of the just-finished iteration."""

    def log_iteration_end(*, algorithm, result: dict, **kwargs) -> None:
        logger.info("Iteration %d: end", result.get("training_iteration", algorithm.iteration))

    return log_iteration_end
