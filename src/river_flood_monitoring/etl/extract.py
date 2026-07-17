"""Extract stage for daily flood monitoring ETL.

This module prepares basin-level inputs for forecasting by resolving forecast
NetCDF file paths and loading OEP thresholds from configured sources.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Any, Dict, List, Optional, Set
import json
import re

from river_flood_monitoring.config import BasinConfig
from river_flood_monitoring.logging import get_logger
from .run_spec import PipelineRunSpec
from .utils import build_unit_id, expand_template

logger = get_logger(__name__)


def extract(
    config: BasinConfig,
    issue_date: date,
    run_spec: PipelineRunSpec,
) -> Dict[str, Any]:
    """
    Prepare basin inputs required for the forecast step.
    """
    basin_name = config.basin_name
    logger.info("Processing basin '%s'", basin_name)

    if run_spec.inputs is None:
        raise ValueError("Run spec must define inputs.oep_json")

    # load GloFAS file paths (single file or ensemble-member files)
    forecast_paths = _resolve_forecast_path(run_spec, issue_date)
    if not forecast_paths:
        raise FileNotFoundError(
            f"Forecast file not available for basin '{basin_name}' on {issue_date}. "
            "Supply a forecast file via ingest settings."
        )

    # load EVT parameters Parquet path
    evt_template = str(run_spec.inputs.evt_params_parquet).strip()
    evt_parquet = Path(expand_template(evt_template, issue_date, basin=basin_name))
    # load OEP file path
    oep_template = str(run_spec.inputs.oep_json).strip()
    oep_path = Path(expand_template(oep_template, issue_date, basin=basin_name))
    thresholds, unit_metadata = _load_oep_thresholds(oep_path, run_spec.decision.oep_min)

    if run_spec.output is None:
        raise ValueError(
            "Run spec must define output settings with top-level target_adm2_pcodes"
        )

    basin_key = basin_name.strip().lower()
    target_adm2 = run_spec.output.target_adm2_pcodes.get(basin_key) or []
    if not target_adm2:
        raise ValueError(
            f"No target_adm2_pcodes configured for basin '{basin_name}'. "
            "Define top-level target_adm2_pcodes in the run spec."
        )

    adm3_to_adm2 = _load_adm3_to_adm2_mapping(Path(run_spec.detection.adm3_geojson))
    thresholds, unit_metadata = _filter_units_by_target_adm2(
        thresholds=thresholds,
        unit_metadata=unit_metadata,
        target_adm2_pcodes=set(target_adm2),
        adm3_to_adm2=adm3_to_adm2,
        basin_name=basin_name,
    )

    det = run_spec.detection

    return {
        "basin_name": basin_name,
        "forecast_paths": forecast_paths,
        "oep_path": oep_path,
        "thresholds": thresholds,
        "unit_metadata": unit_metadata,
        "evt_parquet": evt_parquet,
        "det": det,
    }


def _resolve_forecast_path(
    run_spec: PipelineRunSpec,
    issue_date: date,
) -> Optional[List[str]]:
    """
    Return local path(s) to the GloFAS ensemble NetCDF forecast files.

    The path is derived from ``forecast_path_template``. If the template
    contains ``{ens}`` or ``{ens_no}``, it is treated as a glob pattern and
    all matching files are returned in sorted order.

    Returns ``None`` when no ingest settings are defined or no files are found.
    """
    if run_spec.ingest is None:
        logger.debug("No ingest settings — skipping forecast path resolution")
        return None

    template = run_spec.ingest.forecast_path_template
    has_ens_placeholder = "{ens}" in template or "{ens_no}" in template
    if has_ens_placeholder:
        template_for_lookup = template.replace("{ens}", "*").replace("{ens_no}", "*")
    else:
        # Support templates using a fixed member token like dis_00_YYYYMMDD00.nc.
        template_for_lookup = re.sub(r"([_-])00(?=[_-])", r"\1*", template, count=1)

    candidate = Path(expand_template(template_for_lookup, issue_date))

    if has_ens_placeholder or template_for_lookup != template:
        logger.info("Checking forecast file pattern: %s", candidate)
        matched = sorted(p for p in candidate.parent.glob(candidate.name) if p.is_file())
        if matched:
            logger.info("Forecast files found: %d match(es)", len(matched))
            return [str(p) for p in matched]
    else:
        logger.info("Checking forecast file: %s", candidate)
        if candidate.exists():
            logger.info("Forecast file found: %s", candidate)
            return [str(candidate)]

    logger.warning("Forecast file(s) not found: %s", candidate)
    return None


def _load_oep_thresholds(
    oep_json_path: Path,
    oep_min: float,
) -> tuple[Dict[str, Dict[int, float]], Dict[str, Dict[str, str]]]:
    """
    Load per-unit OEP impact thresholds from JSON.

    Returns
    -------
    tuple
        ``({LEVEL::pcode: {rp: threshold_people}}, {LEVEL::pcode: {level, name, pcode}})``
    """
    logger.info(
        "Loading OEP thresholds from %s (oep_min=%.0f)", oep_json_path, oep_min
    )
    if not oep_json_path.exists():
        raise FileNotFoundError(f"OEP JSON not found: {oep_json_path}")

    raw = json.loads(oep_json_path.read_text(encoding="utf-8"))
    rp_report = [int(float(x)) for x in raw.get("rp_report", [])]

    thresholds: Dict[str, Dict[int, float]] = {}
    unit_metadata: Dict[str, Dict[str, str]] = {}
    for rec in raw.get("units", []):
        pcode = rec.get("pcode")
        if not pcode:
            continue
        level = str(rec.get("level", "") or "").strip()
        unit_id = build_unit_id(level, str(pcode))
        if not unit_id:
            continue
        oep_rl = rec.get("oep_rl", [])
        rp_map: Dict[int, float] = {}
        for idx, rp in enumerate(rp_report):
            if idx >= len(oep_rl):
                continue
            try:
                rp_map[rp] = float(oep_rl[idx])
            except (TypeError, ValueError):
                continue
        if not rp_map:
            continue

        thresholds[unit_id] = rp_map
        unit_metadata[unit_id] = {
            "level": level,
            "name": str(rec.get("name", "") or "").strip(),
            "pcode": str(pcode).strip(),
        }

    logger.info(
        "OEP thresholds loaded: %d units with valid thresholds (from %d total, oep_min=%.0f)",
        len(thresholds),
        len(raw.get("units", [])),
        oep_min,
    )
    return thresholds, unit_metadata


def _load_adm3_to_adm2_mapping(adm3_geojson_path: Path) -> Dict[str, str]:
    """Load ADM3->ADM2 pcode mapping from admin areas GeoJSON."""
    if not adm3_geojson_path.exists():
        raise FileNotFoundError(f"ADM3 GeoJSON not found: {adm3_geojson_path}")

    raw = json.loads(adm3_geojson_path.read_text(encoding="utf-8"))
    features = raw.get("features") or []
    mapping: Dict[str, str] = {}

    for feature in features:
        props = feature.get("properties") or {}
        adm3_pcode = str(props.get("adm3_pcode", "") or "").strip()
        adm2_pcode = str(props.get("adm2_pcode", "") or "").strip()
        if adm3_pcode and adm2_pcode:
            mapping[adm3_pcode] = adm2_pcode

    if not mapping:
        raise ValueError(
            f"ADM3 GeoJSON has no ADM3/ADM2 mapping rows: {adm3_geojson_path}"
        )

    return mapping


def _filter_units_by_target_adm2(
    thresholds: Dict[str, Dict[int, float]],
    unit_metadata: Dict[str, Dict[str, str]],
    target_adm2_pcodes: Set[str],
    adm3_to_adm2: Dict[str, str],
    basin_name: str,
) -> tuple[Dict[str, Dict[int, float]], Dict[str, Dict[str, str]]]:
    """Filter units to selected ADM2 targets using admin-area parent mapping."""
    kept_thresholds: Dict[str, Dict[int, float]] = {}
    kept_unit_metadata: Dict[str, Dict[str, str]] = {}
    missing_parent_map: List[str] = []

    for unit_id, rp_map in thresholds.items():
        meta = unit_metadata.get(unit_id) or {}
        pcode = str(meta.get("pcode", "") or "").strip()
        level = str(meta.get("level", "") or "").strip().upper()

        if not pcode or not level:
            continue

        if level == "ADM2":
            include = pcode in target_adm2_pcodes
        elif level == "ADM3":
            parent_adm2 = adm3_to_adm2.get(pcode)
            if not parent_adm2:
                missing_parent_map.append(pcode)
                continue
            include = parent_adm2 in target_adm2_pcodes
        else:
            continue

        if include:
            kept_thresholds[unit_id] = rp_map
            kept_unit_metadata[unit_id] = meta

    if missing_parent_map:
        sample = ", ".join(sorted(set(missing_parent_map))[:10])
        raise ValueError(
            "Missing ADM3->ADM2 parent mapping for OEP units in basin "
            f"'{basin_name}': {sample}"
        )

    logger.info(
        "Target ADM2 filtering complete for basin '%s': %d -> %d units (targets=%d)",
        basin_name,
        len(thresholds),
        len(kept_thresholds),
        len(target_adm2_pcodes),
    )

    return kept_thresholds, kept_unit_metadata