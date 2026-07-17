"""Shared dataclasses and helper utilities for the ETL pipeline.

Imported by all step modules to avoid circular dependencies.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Any, Dict, List, Optional

DEFAULT_RULE_TIERS: Dict[str, Dict[str, Any]] = {
    "T1": {"rp": 2, "p_thr": 0.50, "n_req": 1, "label": "Moderate Watch"},
    "T2": {"rp": 5, "p_thr": 0.50, "n_req": 1, "label": "High Alert"},
    "T3": {"rp": 10, "p_thr": 0.35, "n_req": 1, "label": "Very High Activation"},
}


def build_unit_id(level: str, pcode: str) -> str:
    """Build a normalized ETL unit identifier like ``ADM3::PH0201501``."""
    level_norm = str(level or "").strip().upper()
    pcode_norm = str(pcode or "").strip()
    if level_norm not in {"ADM2", "ADM3"} or not pcode_norm:
        return ""
    return f"{level_norm}::{pcode_norm}"


def expand_template(
    template: str,
    issue_date: date,
    basin: Optional[str] = None,
    ens_no: Optional[str] = None,
    ens: Optional[str] = None,
) -> str:
    """Substitute date placeholders in a path template.

    Supported placeholders:
    - {date}: YYYY-MM-DD
    - {yyyymmdd}: YYYYMMDD
    - {yyyy}, {mm}, {dd}
    - {basin}: basin name (e.g., "cagayan")
    - {ens_no}, {ens}
    """
    return template.format(
        date=issue_date.isoformat(),
        yyyymmdd=issue_date.strftime("%Y%m%d"),
        yyyy=issue_date.strftime("%Y"),
        mm=issue_date.strftime("%m"),
        dd=issue_date.strftime("%d"),
        basin=basin or "",
        ens_no=ens_no if ens_no is not None else "00",
        ens=ens if ens is not None else "00",
    )


@dataclass
class TierDecision:
    """Tier result for one administrative unit."""

    tier: str
    rp: int
    p_threshold: float
    fired: bool
    fire_lead: Optional[int]
    probability_at_fire: Optional[float]
    impact_population_threshold: Optional[float]
    impact_population_at_fire: Optional[float] = None


@dataclass
class UnitDecision:
    """Per-unit trigger decision payload."""

    unit_id: str
    tiers: List[TierDecision]
    level: str = ""
    name: str = ""
    pcode: str = ""


@dataclass
class BasinRunOutput:
    """Result payload for one basin for one issue date."""

    basin_name: str
    issue_date: str
    forecast_paths: List[str]
    units: List[UnitDecision]
    metadata: Dict[str, Any]
