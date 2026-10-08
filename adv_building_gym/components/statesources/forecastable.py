"""Re-export of the ``Forecastable`` interface for ``s_fc_*`` look-ahead observations.

The contract itself lives in ``_common.forecasting`` so the ``core`` ``ForecastWrapper`` can
import it without importing ``components``. Component modules keep importing it from here
(``from ..forecastable import Forecastable``); ``CsvLookahead`` (in ``csv_lookahead.py``) is
the separate concern for cached future-row reads that publish *no* forecast.
"""

from __future__ import annotations

from adv_building_gym._common.forecasting import Forecastable

__all__ = ["Forecastable"]
