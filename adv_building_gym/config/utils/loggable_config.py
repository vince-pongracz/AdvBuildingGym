"""Interface for config classes that can log their values."""

import logging
from abc import ABC, abstractmethod
from dataclasses import fields


class LoggableConfig(ABC):
    """Mixin for configuration classes that support structured logging.

    Provides a default implementation that logs all dataclass fields.
    Non-dataclass configs should override ``log_values()``.
    """

    @abstractmethod
    def _log_label(self) -> str:
        """Return the label used in log output (e.g. 'EnvConfig')."""

    def log_values(self) -> None:
        """Log all config field values at INFO level."""
        logger = logging.getLogger(type(self).__module__)
        lines = [f"  {f.name} = {getattr(self, f.name)}" for f in fields(self)]
        logger.info("%s:\n%s", self._log_label(), "\n".join(lines))
