"""Altitude Schmitt trigger for local-depth vs legacy-height OF mode."""

from __future__ import annotations


def request_local_depth(
    altitude_m: float,
    max_altitude_m: float,
    hysteresis_m: float,
    previously_requested: bool | None,
) -> bool:
    """Schmitt-trigger local-depth mode around ``max_altitude_m``."""
    band_m = max(0.0, float(hysteresis_m))
    altitude_m = float(altitude_m)
    max_altitude_m = float(max_altitude_m)
    if previously_requested is None:
        return altitude_m <= max_altitude_m
    if previously_requested:
        return altitude_m <= max_altitude_m + band_m
    return altitude_m <= max_altitude_m - band_m
