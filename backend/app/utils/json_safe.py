"""Convert pandas/numpy values into strict-JSON-safe Python values (no NaN/Infinity)."""

import datetime as dt
import decimal
import math
from typing import Any

import numpy as np
import pandas as pd

MAX_TEXT_LENGTH = 200


def safe_float(value: Any, ndigits: int | None = 6) -> float | None:
    """Return a finite float (optionally rounded) or None."""
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return round(number, ndigits) if ndigits is not None else number


def to_json_safe(value: Any, max_text_length: int = MAX_TEXT_LENGTH) -> str | int | float | bool | None:
    """Convert a single cell value to a JSON-safe scalar."""
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating, decimal.Decimal)):
        return safe_float(value, ndigits=None)
    if isinstance(value, pd.Timedelta):
        return None if pd.isna(value) else str(value)
    # NaT and other pandas missing markers.
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    if isinstance(value, (pd.Timestamp, dt.datetime, dt.date, dt.time)):
        return value.isoformat()
    if isinstance(value, np.datetime64):
        return pd.Timestamp(value).isoformat()
    text = value.decode("utf-8", "replace") if isinstance(value, bytes) else str(value)
    if len(text) > max_text_length:
        return text[: max_text_length - 1] + "…"
    return text
