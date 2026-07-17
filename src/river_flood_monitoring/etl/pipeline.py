"""Pipeline orchestration for daily flood monitoring ETL.

This module contains the high-level pipeline entry points.
"""

from __future__ import annotations

from dataclasses import asdict
from datetime import date
from datetime import datetime
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from river_flood_monitoring.config import BasinConfig, build_basin_configs
from river_flood_monitoring.logging import get_logger, setup_pipeline_file_log

from .run_spec import load_run_spec
from .extract import extract
from .forecast import forecast
from .save import save
from .run_spec import DetectionSettings
from .utils import BasinRunOutput, TierDecision, UnitDecision, expand_template

logger = get_logger(__name__)


def run_daily_monitoring(
    issue_date: date,
    basin_names: List[str],
    run_spec_path: str,
    do_extract: bool = False,
    do_forecast: bool = False,
    do_save: bool = False,
) -> Tuple[List[BasinRunOutput], Optional[Path]]:
    """Execute selected ETL steps for all requested basins.

    Attaches a dated ``logs/<run_name>_<datetime>.txt`` file handler before
    the first basin so the full run trace is persisted.

    If no step flags are enabled, extract, forecast, and save run in order.
    When an upstream step is skipped, this function attempts to load the
    required artifacts from ``data/etl_step_cache/<run_name>/<issue_date>/``.
    """
    if not any([do_extract, do_forecast, do_save]):
        do_extract = True
        do_forecast = True
        do_save = True

    run_spec = load_run_spec(run_spec_path)

    log_path_date = date.today()
    if run_spec.output.log_dir_template.strip():
        log_dir_template = run_spec.output.log_dir_template
    log_file = setup_pipeline_file_log(
        log_dir=Path(expand_template(log_dir_template, log_path_date)),
        run_name=run_spec.run_name,
    )

    logger.info(
        "run_daily_monitoring started — run='%s', issue_date=%s, basins=%d, "
        "steps=[extract=%s, forecast=%s, save=%s], log=%s",
        run_spec.run_name,
        issue_date,
        len(basin_names),
        do_extract,
        do_forecast,
        do_save,
        log_file,
    )

    artifact_root = _artifact_root(run_spec.run_name, issue_date)
    artifact_root.mkdir(parents=True, exist_ok=True)

    _write_run_manifest(
        artifact_root=artifact_root,
        run_name=run_spec.run_name,
        issue_date=issue_date,
        basin_names=basin_names,
        run_spec_path=run_spec_path,
        selected_steps={
            "extract": do_extract,
            "forecast": do_forecast,
            "save": do_save,
        },
    )

    extracted_by_basin: Dict[str, Dict[str, Any]] = {}
    forecast_by_basin: Dict[str, Dict[str, Any]] = {}
    basin_configs: List[BasinConfig] = build_basin_configs(basin_names)

    for cfg in basin_configs:
        basin_name = cfg.basin_name

        if do_extract:
            extracted = extract(
                config=cfg,
                issue_date=issue_date,
                run_spec=run_spec,
            )
            _write_extract_artifact(artifact_root, basin_name, extracted)
            extracted_by_basin[basin_name] = extracted
        elif do_forecast:
            extracted_by_basin[basin_name] = _read_extract_artifact(artifact_root, basin_name)

        if do_forecast:
            forecasted = forecast(extracted_by_basin[basin_name], issue_date, run_spec)
            _write_forecast_artifact(artifact_root, basin_name, forecasted)
            forecast_by_basin[basin_name] = forecasted
        elif do_save:
            forecast_by_basin[basin_name] = _read_forecast_artifact(artifact_root, basin_name)

    if do_save:
        basin_forecasts = [forecast_by_basin[cfg.basin_name] for cfg in basin_configs]
        save_outputs = save(run_spec, issue_date, basin_forecasts)
        basin_results = save_outputs["basin_results"]
        activation_file = save_outputs["activation_file"]
        operational_information_file = save_outputs["operational_information_file"]

        total_fired = sum(
            1
            for basin in basin_results
            for unit in basin.units
            for tier in unit.tiers
            if tier.fired
        )
        logger.info(
            "run_daily_monitoring complete — %d basins, %d tier fires, activation=%s, operational_information=%s",
            len(basin_results),
            total_fired,
            activation_file,
            operational_information_file,
        )
        return basin_results, operational_information_file

    if do_forecast:
        total_units = sum(len(v.get("units", [])) for v in forecast_by_basin.values())
        logger.info(
            "run_daily_monitoring complete (no save) — forecast artifacts=%d, units=%d",
            len(forecast_by_basin),
            total_units,
        )
    elif do_extract:
        logger.info(
            "run_daily_monitoring complete (extract only) — artifacts=%d",
            len(extracted_by_basin),
        )
    else:
        logger.info("run_daily_monitoring complete (no steps executed)")

    return [], None


def _artifact_root(run_name: str, issue_date: date) -> Path:
    return Path("data/etl_step_cache") / run_name / issue_date.isoformat()


def _run_manifest_path(artifact_root: Path) -> Path:
    return artifact_root / "run_manifest.json"


def _extract_artifact_path(artifact_root: Path, basin_name: str) -> Path:
    return artifact_root / "extract" / f"{basin_name}.json"


def _forecast_artifact_path(artifact_root: Path, basin_name: str) -> Path:
    return artifact_root / "forecast" / f"{basin_name}.json"


def _write_run_manifest(
    artifact_root: Path,
    run_name: str,
    issue_date: date,
    basin_names: List[str],
    run_spec_path: str,
    selected_steps: Dict[str, bool],
) -> None:
    payload = {
        "schema_version": 1,
        "run_name": run_name,
        "issue_date": issue_date.isoformat(),
        "run_spec_path": run_spec_path,
        "basin_names": basin_names,
        "selected_steps": selected_steps,
        "generated_at_utc": datetime.utcnow().isoformat() + "Z",
    }
    path = _run_manifest_path(artifact_root)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _write_extract_artifact(
    artifact_root: Path,
    basin_name: str,
    extracted: Dict[str, Any],
) -> None:
    path = _extract_artifact_path(artifact_root, basin_name)
    path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "schema_version": 1,
        "basin_name": str(extracted["basin_name"]),
        "forecast_paths": [str(p) for p in extracted.get("forecast_paths") or []],
        "forecast_filename_example": extracted.get("forecast_filename_example"),
        "oep_path": _to_optional_str(extracted.get("oep_path")),
        "thresholds": extracted.get("thresholds", {}),
        "unit_metadata": extracted.get("unit_metadata", {}),
        "adm3_to_adm2": extracted.get("adm3_to_adm2", {}),
        "evt_parquet": _to_optional_str(extracted.get("evt_parquet")),
        "det": asdict(extracted["det"]),
        "generated_at_utc": datetime.utcnow().isoformat() + "Z",
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _read_extract_artifact(artifact_root: Path, basin_name: str) -> Dict[str, Any]:
    path = _extract_artifact_path(artifact_root, basin_name)
    if not path.exists():
        raise FileNotFoundError(
            f"Extract artifact not found for basin '{basin_name}': {path}. "
            "Run with --extract first."
        )

    raw = json.loads(path.read_text(encoding="utf-8"))
    thresholds: Dict[str, Dict[int, float]] = {}
    for unit_id, rp_map in (raw.get("thresholds") or {}).items():
        thresholds[unit_id] = {int(k): float(v) for k, v in rp_map.items()}

    det_cfg = raw.get("det") or {}
    det = DetectionSettings(
        t0_years=float(det_cfg.get("t0_years", 2.0)),
        a_min_km2=float(det_cfg.get("a_min_km2", 100.0)),
        depth_threshold_m=float(det_cfg.get("depth_threshold_m", 0.02)),
        cc_connectivity=int(det_cfg.get("cc_connectivity", 2)),
        flood_detect_rps=[int(v) for v in det_cfg.get("flood_detect_rps", [2, 5, 10, 20])],
        evt_params_parquet=str(det_cfg.get("evt_params_parquet", "")),
        jrc_root=str(det_cfg.get("jrc_root", "")),
        worldpop_tif=str(det_cfg.get("worldpop_tif", "")),
        adm3_geojson=str(det_cfg.get("adm3_geojson", "")),
    )

    return {
        "basin_name": str(raw["basin_name"]),
        "forecast_paths": [str(p) for p in raw.get("forecast_paths") or []],
        "forecast_filename_example": raw.get("forecast_filename_example"),
        "oep_path": Path(raw["oep_path"]),
        "thresholds": thresholds,
        "unit_metadata": raw.get("unit_metadata") or {},
        "adm3_to_adm2": {
            str(adm3): str(adm2)
            for adm3, adm2 in (raw.get("adm3_to_adm2") or {}).items()
        },
        "evt_parquet": Path(raw["evt_parquet"]),
        "det": det,
    }


def _write_forecast_artifact(
    artifact_root: Path,
    basin_name: str,
    forecasted: Dict[str, Any],
) -> None:
    path = _forecast_artifact_path(artifact_root, basin_name)
    path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "schema_version": 1,
        "basin_name": str(forecasted["basin_name"]),
        "forecast_paths": [str(p) for p in forecasted.get("forecast_paths") or []],
        "oep_path": _to_optional_str(forecasted.get("oep_path")),
        "impacts_source": str(forecasted.get("impacts_source", "")),
        "units": [_serialise_unit(unit) for unit in forecasted.get("units") or []],
        "generated_at_utc": datetime.utcnow().isoformat() + "Z",
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _read_forecast_artifact(artifact_root: Path, basin_name: str) -> Dict[str, Any]:
    path = _forecast_artifact_path(artifact_root, basin_name)
    if not path.exists():
        raise FileNotFoundError(
            f"Forecast artifact not found for basin '{basin_name}': {path}. "
            "Run with --forecast first."
        )

    raw = json.loads(path.read_text(encoding="utf-8"))
    return {
        "basin_name": str(raw["basin_name"]),
        "forecast_paths": [str(p) for p in raw.get("forecast_paths") or []],
        "oep_path": Path(raw["oep_path"]),
        "impacts_source": str(raw.get("impacts_source", "")),
        "units": [_deserialise_unit(unit) for unit in raw.get("units") or []],
    }


def _serialise_unit(unit: UnitDecision) -> Dict[str, Any]:
    return {
        "unit_id": unit.unit_id,
        "level": unit.level,
        "name": unit.name,
        "pcode": unit.pcode,
        "tiers": [
            {
                "tier": tier.tier,
                "rp": tier.rp,
                "p_threshold": tier.p_threshold,
                "fired": tier.fired,
                "fire_lead": tier.fire_lead,
                "probability_at_fire": tier.probability_at_fire,
                "impact_population_threshold": tier.impact_population_threshold,
                "impact_population_at_fire": tier.impact_population_at_fire,
            }
            for tier in unit.tiers
        ],
    }


def _deserialise_unit(raw: Dict[str, Any]) -> UnitDecision:
    tiers = [
        TierDecision(
            tier=str(t["tier"]),
            rp=int(t["rp"]),
            p_threshold=float(t["p_threshold"]),
            fired=bool(t["fired"]),
            fire_lead=int(t["fire_lead"]) if t.get("fire_lead") is not None else None,
            probability_at_fire=(
                float(t["probability_at_fire"])
                if t.get("probability_at_fire") is not None
                else None
            ),
            impact_population_threshold=(
                float(t["impact_population_threshold"])
                if t.get("impact_population_threshold") is not None
                else None
            ),
            impact_population_at_fire=(
                float(t["impact_population_at_fire"])
                if t.get("impact_population_at_fire") is not None
                else None
            ),
        )
        for t in raw.get("tiers") or []
    ]
    return UnitDecision(
        unit_id=str(raw["unit_id"]),
        level=str(raw.get("level", "") or ""),
        name=str(raw.get("name", "") or ""),
        pcode=str(raw.get("pcode", "") or ""),
        tiers=tiers,
    )


def _to_optional_str(value: Any) -> Optional[str]:
    if value is None:
        return None
    return str(value)


__all__ = [
    "extract",
    "forecast",
    "save",
    "run_daily_monitoring",
]