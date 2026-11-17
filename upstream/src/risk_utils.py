# FILE: src/risk_utils.py
from __future__ import annotations

from typing import Literal


def get_rho(risk_model: str, gamma_or_eta: float) -> float:
    """
    Map risk model name to rho (risk aversion parameter in the small-w objective).

    CARA: rho = gamma
    CRRA: rho = eta
    """
    m = (risk_model or "CARA").strip().upper()
    if m == "CRRA":
        return float(gamma_or_eta)
    # default CARA
    return float(gamma_or_eta)

