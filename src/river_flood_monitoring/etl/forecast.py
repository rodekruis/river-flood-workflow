"""Flood monitoring forecast stage: detect, impact, evaluate, and decide.

This module implements the forecasting pipeline used by the ETL run:

Detect: read GloFAS ensemble NetCDF files, convert discharge to return periods
    with EVT1 parameters, detect spatial flood patches using connected
    components, and render per-patch depth rasters from JRC maps.
Impact: aggregate affected population per admin unit from event patch rasters
    and WorldPop, producing an ImpactCube
    (unit -> lead -> member -> rp -> affected_people).
Evaluate: compute ensemble exceedance probabilities against per-unit OEP
    impact thresholds.
Decide: apply configured tier rules with persistence and minimum-lead
    constraints to produce unit-level alert decisions.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable
import re
import tempfile
from functools import lru_cache
from contextlib import ExitStack
import numbers
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union
import numpy as np
import pandas as pd

from river_flood_monitoring.etl.run_spec import DetectionSettings, DecisionSettings, PipelineRunSpec
from .utils import TierDecision, UnitDecision

from river_flood_monitoring.logging import get_logger
logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_RP_CAP = 500.0
TOTAL_ENSEMBLE_MEMBERS = 51


# Type alias: unit → lead → member → rp → people
ImpactCube = Dict[str, Dict[int, Dict[int, Dict[int, float]]]]


def forecast(
    extracted: Dict[str, Any],
    issue_date: date,
    run_spec: PipelineRunSpec,
) -> Dict[str, Any]:
    """Run flood detection, exceedance, and alert rule evaluation."""
    basin_name = str(extracted["basin_name"])
    logger.info("Starting evaluation — basin='%s', issue_date=%s", basin_name, issue_date)

    lead_days_list = _build_lead_days_list(
        min_lead=run_spec.decision.min_lead,
        max_lead=run_spec.decision.max_lead,
    )

    impact_cube, members, _ = detect_flood_events(
        forecast_paths=extracted["forecast_paths"],
        forecast_filename_example=extracted.get("forecast_filename_example"),
        evt_params_path=extracted["evt_parquet"],
        oep_json_path=extracted["oep_path"],
        issue_date=issue_date,
        basin_name=basin_name,
        settings=extracted["det"],
        lead_days_list=lead_days_list,
    )
    impacts_source = "detect_phase:" + ",".join(str(p) for p in extracted["forecast_paths"])
    logger.info("Detection mode complete — impact cube has %d units", len(impact_cube))

    prob_exceed = _compute_prob_exceed(impact_cube, extracted["thresholds"], members)
    units: List[UnitDecision] = _apply_tier_rules(
        prob_exceed,
        extracted["thresholds"],
        extracted.get("unit_metadata", {}),
        impact_cube,
        members,
        run_spec.decision,
    )

    logger.info(
        "Evaluation complete — basin='%s', %d units evaluated",
        basin_name,
        len(units),
    )
    return {
        "basin_name": basin_name,
        "forecast_paths": extracted["forecast_paths"],
        "oep_path": extracted["oep_path"],
        "units": units,
        "impacts_source": impacts_source,
    }


def _build_lead_days_list(min_lead: int, max_lead: int) -> List[int]:
    """Build an inclusive lead-day list from decision settings."""
    min_ld = int(min_lead)
    max_ld = int(max_lead)

    if min_ld < 1:
        raise ValueError(f"Invalid lead window: min_lead={min_ld} must be >= 1")
    if max_ld < min_ld:
        raise ValueError(
            f"Invalid lead window: min_lead={min_ld} must be <= max_lead={max_ld}"
        )

    return list(range(min_ld, max_ld + 1))


def _find_latest_persistent_lead(
    firing_leads: Iterable[int],
    min_lead: int,
    persist_days: int,
) -> Optional[int]:
    """Return the latest lead day satisfying both persistence and minimum lead.

    A lead qualifies when it is part of a contiguous window of
    ``persist_days`` consecutive integer lead days all of which appear in
    ``firing_leads``, and the lead is >= ``min_lead``.

    Returns the latest (highest) such lead for maximum warning time.
    """
    if persist_days <= 1:
        eligible = [ld for ld in firing_leads if ld >= min_lead]
        return max(eligible) if eligible else None

    firing_set = set(int(ld) for ld in firing_leads)
    eligible = sorted([ld for ld in firing_set if ld >= min_lead], reverse=True)

    for lead in eligible:
        for start in range(lead - persist_days + 1, lead + 1):
            window = {start + step for step in range(persist_days)}
            if lead in window and window.issubset(firing_set):
                return lead
    return None


def _apply_tier_rules(
    prob_exceed: Dict[str, Dict[int, Dict[int, float]]],
    thresholds: Dict[str, Dict[int, float]],
    unit_metadata: Dict[str, Dict[str, str]],
    impact_cube: ImpactCube,
    members: Iterable[int],
    decision: DecisionSettings,
) -> List[UnitDecision]:
    """Evaluate all tier rules across all units.

    Parameters
    ----------
    prob_exceed:
        Exceedance probability cube — unit → lead → rp → probability.
    thresholds:
        OEP thresholds — unit → rp → impact_population_threshold.
    decision:
        Policy settings (persist_days, min_lead, rule list).

    Returns
    -------
    list of UnitDecision
    """
    logger.info(
        "Applying %d tier rules to %d units (persist_days=%d, min_lead=%d)",
        len(decision.rules),
        len(prob_exceed),
        decision.persist_days,
        decision.min_lead,
    )
    member_list = sorted(set(int(m) for m in members))
    units: List[UnitDecision] = []
    fired_count = 0
    ineligible_count = 0

    for unit_id, lead_map in prob_exceed.items():
        unit_thresholds = thresholds.get(unit_id, {})
        rp2_threshold = float(unit_thresholds.get(2, 0.0))
        meets_oep_min = rp2_threshold >= float(decision.oep_min)
        if not meets_oep_min:
            ineligible_count += 1

        tier_results: List[TierDecision] = []
        for rule in decision.rules:
            fire_lead: Optional[int] = None
            if meets_oep_min:
                firing = [
                    lead
                    for lead, rp_prob in lead_map.items()
                    if rp_prob.get(rule.rp, 0.0) >= rule.p_thr
                ]
                fire_lead = _find_latest_persistent_lead(
                    firing,
                    min_lead=decision.min_lead,
                    persist_days=decision.persist_days,
                )
            prob_at_fire: Optional[float] = None
            impact_population_at_fire: Optional[float] = None
            if fire_lead is not None:
                prob_at_fire = lead_map.get(fire_lead, {}).get(rule.rp)
                per_member_at_fire = impact_cube.get(unit_id, {}).get(fire_lead, {})
                if member_list:
                    impact_values = [
                        float(per_member_at_fire.get(member, {}).get(rule.rp, 0.0))
                        for member in member_list
                    ]
                    impact_population_at_fire = sum(impact_values) / len(impact_values)
                fired_count += 1
                logger.info(
                    "Tier %s FIRED — unit='%s', fire_lead=%d, p=%.2f",
                    rule.name,
                    unit_id,
                    fire_lead,
                    prob_at_fire or 0.0,
                )

            tier_results.append(
                TierDecision(
                    tier=rule.name,
                    rp=rule.rp,
                    p_threshold=rule.p_thr,
                    fired=fire_lead is not None,
                    fire_lead=fire_lead,
                    probability_at_fire=prob_at_fire,
                    impact_population_threshold=unit_thresholds.get(rule.rp),
                    impact_population_at_fire=impact_population_at_fire,
                )
            )
        meta = unit_metadata.get(unit_id, {})
        units.append(
            UnitDecision(
                unit_id=unit_id,
                level=str(meta.get("level", "") or ""),
                name=str(meta.get("name", "") or ""),
                pcode=str(meta.get("pcode", "") or ""),
                tiers=tier_results,
            )
        )

    logger.info(
        "Tier evaluation complete: %d units, %d tier decisions fired, %d units below oep_min=%.0f",
        len(units),
        fired_count,
        ineligible_count,
        decision.oep_min,
    )
    return units


@dataclass
class EventPatchImpactInput:
    """Input payload.

    Parameters
    ----------
    lead_day:
        Forecast lead day for this patch.
    member_id:
        Ensemble member identifier for this patch.
    rp:
        Return period bucket that this patch contributes to.
    depth_raster:
        Path to the flood depth raster for this patch.
    event_id:
        Optional patch/event identifier used for traceability.
    extra:
        Optional payload forwarded to philflood's aggregator when supported.
    """

    lead_day: int
    member_id: int
    rp: int
    depth_raster: Path
    event_id: str = ""
    extra: Dict[str, Any] = field(default_factory=dict)


def _unit_key(unit_name: str) -> str:
    unit = str(unit_name).strip()
    if not unit:
        return ""
    if "::" in unit:
        return unit
    return f"ADM3::{unit}"


def _normalise_aggregated_rows(result: Any) -> Dict[str, float]:
    """Normalise aggregator output to {unit_key: affected_people}.

    Supports dict-like and dataframe-like outputs to keep the impact phase tolerant to
    implementation differences across philflood versions.
    """
    out: Dict[str, float] = {}

    if isinstance(result, Mapping):
        for key, value in result.items():
            if isinstance(value, numbers.Number):
                unit = _unit_key(str(key))
                if unit:
                    out[unit] = out.get(unit, 0.0) + float(value)
        return out

    # DataFrame-like output handling without requiring pandas type imports.
    if hasattr(result, "iterrows"):
        for _, row in result.iterrows():
            unit = ""
            for col in ("unit_id", "unit", "adm3_name", "name"):
                if col in row and str(row[col]).strip():
                    unit = _unit_key(str(row[col]))
                    break

            pop_val = None
            for col in ("affected_pop", "affected_population", "population", "people"):
                if col in row:
                    pop_val = row[col]
                    break

            if unit and isinstance(pop_val, numbers.Number):
                out[unit] = out.get(unit, 0.0) + float(pop_val)

    return out


def _aggregate_population_from_arrays(
    pop: np.ndarray,
    depth: np.ndarray,
    admin_id_raster: np.ndarray,
    thresholds_m: Iterable[float],
    id_to_name: Dict[int, str],
) -> pd.DataFrame:
    """Aggregate affected population by admin unit at given depth thresholds."""
    ids_flat = admin_id_raster.reshape(-1)
    pop_flat = pop.reshape(-1)
    depth_flat = depth.reshape(-1)
    rows: List[Dict[str, object]] = []

    for thr in thresholds_m:
        mask = depth_flat >= float(thr)
        counts: Dict[int, float] = {}
        for id_, p in zip(ids_flat[mask], pop_flat[mask]):
            if isinstance(id_, float) and np.isnan(id_):
                continue
            try:
                key = int(id_)
            except Exception:
                continue
            if not np.isfinite(float(p)):
                continue
            counts[key] = counts.get(key, 0.0) + float(p)

        for id_, c in counts.items():
            rows.append(
                {
                    "adm_id": id_,
                    "name": id_to_name.get(id_, str(id_)),
                    "depth_thr_m": float(thr),
                    "affected_pop": float(c),
                }
            )

    return pd.DataFrame(rows)


def _read_raster_array(path: str) -> Tuple[np.ndarray, Any, Any]:
    import rasterio

    with rasterio.open(path) as src:
        arr = src.read(1).astype(np.float32)
        transform = src.transform
        crs = src.crs
        nodata = src.nodata

    if nodata is not None:
        arr[arr == nodata] = np.nan
    return arr, transform, crs


@lru_cache(maxsize=4)
def _read_static_raster_array_cached(path: str) -> Tuple[np.ndarray, Any, Any]:
    """Read and cache static rasters reused across many event patches."""
    return _read_raster_array(path)


def _reproject_depth_to_worldpop(
    src_arr: np.ndarray,
    src_transform: Any,
    src_crs: Any,
    dst_shape: Tuple[int, int],
    dst_transform: Any,
    dst_crs: Any,
) -> np.ndarray:
    import rasterio
    from rasterio.warp import reproject

    out = np.full(dst_shape, np.nan, dtype=np.float32)
    reproject(
        source=src_arr,
        destination=out,
        src_transform=src_transform,
        src_crs=src_crs,
        dst_transform=dst_transform,
        dst_crs=dst_crs,
        src_nodata=np.nan,
        dst_nodata=np.nan,
        resampling=rasterio.enums.Resampling.bilinear,
    )
    return out


def _aggregate_affected_population_(**kwargs: Any) -> pd.DataFrame:
    """Aggregate population exposure."""
    depth_raster = kwargs.get("depth_raster")
    worldpop_tif = kwargs.get("worldpop_tif")
    threshold_m = kwargs.get("depth_threshold_m", 0.02)

    if not depth_raster or not worldpop_tif:
        raise ValueError(
            "Missing raster inputs: depth_raster and worldpop_tif are required."
        )

    # WorldPop is constant across all patches in a run; cache to avoid repeated
    # full-raster disk reads that can make long runs appear hung.
    pop_grid, pop_transform, pop_crs = _read_static_raster_array_cached(
        str(Path(worldpop_tif).resolve())
    ) #TODO: check if needed
    depth_grid, depth_transform, depth_crs = _read_raster_array(str(depth_raster))

    if (
        depth_grid.shape != pop_grid.shape
        or depth_transform != pop_transform
        or depth_crs != pop_crs
    ):
        depth_grid = _reproject_depth_to_worldpop(
            src_arr=depth_grid,
            src_transform=depth_transform,
            src_crs=depth_crs,
            dst_shape=pop_grid.shape,
            dst_transform=pop_transform,
            dst_crs=pop_crs,
        )

    admin = kwargs.get("admin_id_raster")
    names = kwargs.get("id_to_name")
    if admin is None or names is None:
        raise ValueError(
            "Missing admin inputs. Provide admin_id_raster and id_to_name."
        )

    return _aggregate_population_from_arrays(
        pop=pop_grid,
        depth=depth_grid,
        admin_id_raster=np.asarray(admin),
        thresholds_m=[float(threshold_m)],
        id_to_name={int(k): str(v) for k, v in dict(names).items()},
    )


def _call_population_aggregator(
    patch: EventPatchImpactInput,
    worldpop_tif: Path,
    depth_threshold_m: float,
) -> Dict[str, float]:
    """Invoke internal population aggregation for one event patch."""

    candidates: Dict[str, Any] = {
        "depth_raster": str(patch.depth_raster),
        "worldpop_tif": str(worldpop_tif),
        "depth_threshold_m": float(depth_threshold_m),
        "event_id": patch.event_id or f"lead{patch.lead_day}_m{patch.member_id}_rp{patch.rp}",
        "rp": int(patch.rp),
    }
    candidates.update(patch.extra)

    result = _aggregate_affected_population_(**candidates)
    return _normalise_aggregated_rows(result)


def _compute_impacts_from_event_patches(
    patches: Iterable[EventPatchImpactInput],
    worldpop_tif: Path,
    depth_threshold_m: float = 0.02,
) -> Tuple[List[int], List[int], ImpactCube]:
    """Build an ``ImpactCube`` by aggregating people affected per event patch.

    Each patch is processed independently and mapped into
    ``cube[unit][lead_day][member_id][rp]``.
    """
    worldpop_tif = Path(worldpop_tif)
    if not worldpop_tif.exists():
        raise FileNotFoundError(f"WorldPop raster not found: {worldpop_tif}")

    patch_list = list(patches)
    members_seen = set()
    leads_seen = set()
    cube: ImpactCube = {}
    n_ok = 0

    for patch in patch_list:
        lead = int(patch.lead_day)
        member = int(patch.member_id)
        rp = int(patch.rp)
        if lead <= 0 or member < 0 or rp <= 0:
            logger.warning("Skipping patch with invalid lead/member/rp: %s", patch)
            continue

        if not Path(patch.depth_raster).exists():
            logger.warning("Skipping patch, depth raster not found: %s", patch.depth_raster)
            continue

        unit_impacts = _call_population_aggregator(
            patch=patch,
            worldpop_tif=worldpop_tif,
            depth_threshold_m=depth_threshold_m,
        )
        for unit, affected_people in unit_impacts.items():
            per_rp = cube.setdefault(unit, {}).setdefault(lead, {}).setdefault(member, {})
            per_rp[rp] = per_rp.get(rp, 0.0) + float(affected_people)

        members_seen.add(member)
        leads_seen.add(lead)
        n_ok += 1

    members = sorted(members_seen)
    leads = sorted(leads_seen)
    logger.info(
        "Computed impacts from %d/%d event patches: %d members, %d lead days, %d units",
        n_ok,
        len(patch_list),
        len(members),
        len(leads),
        len(cube),
    )
    return members, leads, cube


def _load_precomputed_impacts(
    impacts_path: Path,
) -> Tuple[List[int], List[int], ImpactCube]:
    """Load an ensemble impact cube from a JSON file.

    Expected schema::

        {
          "ensemble_members": [1, 2, ...],
          "lead_days": [1, 2, ...],
          "records": [
            {
              "unit_id": "ADM3::Name",
              "lead_day": 5,
              "member_id": 1,
              "rp_affected_people": {"2": 123.0, "5": 77.0, "10": 20.0}
            }
          ]
        }

    Returns
    -------
    (ensemble_members, lead_days, cube)
    """
    logger.info("Loading precomputed impacts from %s", impacts_path)
    raw = json.loads(impacts_path.read_text(encoding="utf-8"))
    members = [int(m) for m in raw.get("ensemble_members", [])]
    leads = [int(ld) for ld in raw.get("lead_days", [])]

    cube: ImpactCube = {}
    n_records = 0
    for rec in raw.get("records", []):
        unit = str(rec.get("unit_id", ""))
        if not unit:
            continue
        lead = int(rec.get("lead_day", 0))
        member = int(rec.get("member_id", 0))
        if lead <= 0 or member < 0:
            continue

        rp_dict: Dict[int, float] = {}
        for rp_raw, value in (rec.get("rp_affected_people") or {}).items():
            try:
                rp_dict[int(float(rp_raw))] = float(value)
            except (TypeError, ValueError):
                continue

        cube.setdefault(unit, {}).setdefault(lead, {})[member] = rp_dict
        n_records += 1

    logger.info(
        "Loaded impact cube: %d members, %d lead days, %d records across %d units",
        len(members),
        len(leads),
        n_records,
        len(cube),
    )
    return members, leads, cube

# ---------------------------------------------------------------------------
# EVT1 GPD helpers
# ---------------------------------------------------------------------------

def _gpd_exceedance_rate(
    q, u, sigma, xi, lam,
):
    """POT-Poisson-GPD exceedance rate (events/year) for discharge q."""
    import numpy as np
    q, u, sigma, xi, lam = np.broadcast_arrays(
        np.asarray(q, float), np.asarray(u, float),
        np.asarray(sigma, float), np.asarray(xi, float), np.asarray(lam, float),
    )
    y = np.maximum((q - u) / sigma, 0.0)
    near0 = np.isclose(xi, 0.0)
    surv = np.empty_like(y)
    surv[near0] = np.exp(-y[near0])
    non0 = ~near0
    xi_non0 = xi[non0]
    y_non0 = y[non0]
    base = 1.0 + xi_non0 * y_non0

    # For xi < 0, values with 1 + xi*y <= 0 are outside GPD support and have
    # zero survival probability; avoid invalid power on non-positive bases.
    surv_non0 = np.zeros_like(base)
    valid = base > 0.0
    surv_non0[valid] = np.power(base[valid], -1.0 / xi_non0[valid])
    surv[non0] = surv_non0
    return np.clip(lam * surv, 1e-12, None)


def discharge_to_return_period(q, u, sigma, xi, lam, rp_cap: float = _RP_CAP):
    """Convert discharge to return period (years) via EVT1 GPD."""
    import numpy as np
    rate = _gpd_exceedance_rate(q=q, u=u, sigma=sigma, xi=xi, lam=lam)
    return np.clip(1.0 / rate, 1.0, rp_cap)


# ---------------------------------------------------------------------------
# Spatial helpers
# ---------------------------------------------------------------------------

def _parse_cell_coords(cell_id: str) -> Tuple[float, float]:
    """Extract (lat, lon) from cell_id = 'CELL__lat_XX.XXXX__lon_YY.YYYY'."""
    m = re.search(r"lat_([-\d.]+)__lon_([-\d.]+)", str(cell_id))
    if not m:
        raise ValueError(f"Cannot parse lat/lon from cell_id={cell_id!r}")
    return float(m.group(1).rstrip(".")), float(m.group(2).rstrip("."))


def _earth_cell_area_km2(lat_deg, dlat_deg: float, dlon_deg: float):
    """Approximate area of a lat-lon grid cell at each latitude (km²)."""
    import numpy as np
    R = 6371.0
    lat_rad = np.deg2rad(lat_deg)
    return R * R * np.deg2rad(dlat_deg) * np.deg2rad(dlon_deg) * np.cos(lat_rad)


def _build_support_grid(evt_params) -> Tuple:
    """Build the 2-D support grid for connected-component detection."""
    import numpy as np
    ep = evt_params.set_index("cell_id")
    lat_vals = np.sort(ep["lat"].unique())
    lon_vals = np.sort(ep["lon"].unique())
    dlat = float(np.median(np.abs(np.diff(lat_vals)))) if len(lat_vals) > 1 else 0.1
    dlon = float(np.median(np.abs(np.diff(lon_vals)))) if len(lon_vals) > 1 else 0.1
    lat_to_i = {float(v): i for i, v in enumerate(lat_vals)}
    lon_to_j = {float(v): j for j, v in enumerate(lon_vals)}
    nlat, nlon = len(lat_vals), len(lon_vals)
    grid_index = -np.ones((nlat, nlon), dtype=int)
    cell_ids = ep.index.tolist()
    for k, cid in enumerate(cell_ids):
        la = float(ep.loc[cid, "lat"])
        lo = float(ep.loc[cid, "lon"])
        grid_index[lat_to_i[la], lon_to_j[lo]] = k
    area_lat = _earth_cell_area_km2(lat_vals, dlat, dlon)
    area_grid = area_lat[:, None] * np.ones((1, nlon))
    return lat_vals, lon_vals, grid_index, area_grid, lat_vals.copy(), lon_vals.copy()


# ---------------------------------------------------------------------------
# Forecast reading utilities
# ---------------------------------------------------------------------------

def _nearest_index_1d(arr_1d, targets):
    import numpy as np
    arr = np.asarray(arr_1d, float)
    t = np.asarray(targets, float)
    dif = np.diff(arr)
    is_asc = bool(np.all(dif >= 0))
    is_desc = bool(np.all(dif <= 0))
    if not (is_asc or is_desc):
        return np.array([int(np.argmin(np.abs(arr - x))) for x in t])
    rev = False
    arr_use = arr
    if is_desc:
        arr_use = arr[::-1]
        rev = True
    idx = np.searchsorted(arr_use, t, side="left")
    idx = np.clip(idx, 1, len(arr_use) - 1)
    left = arr_use[idx - 1]
    right = arr_use[idx]
    idx_final = np.where(np.abs(right - t) < np.abs(t - left), idx, idx - 1).astype(int)
    if rev:
        idx_final = (len(arr) - 1) - idx_final
    return idx_final


def _dt_from_data_fields(data_date: int, data_time: int):
    import pandas as pd
    return pd.to_datetime(f"{int(data_date):08d}{int(data_time):04d}", format="%Y%m%d%H%M")


def _build_netcdf_index(
    file_paths: Sequence[Path],
    var_name: str,
    time_dim: str,
    forecast_filename_example: Optional[str] = None,
) -> dict:
    """Index NetCDF files without loading all data into memory.

    Returns mapping of (valid_time, member) → (file_path, time_idx).
    """
    import pandas as pd
    import xarray as xr

    msg_index: dict = {}
    vt_lookup: dict = {}
    steps: set = set()
    members: set = set()

    def _member_token(ensemble: int) -> str:
        if forecast_filename_example:
            name = forecast_filename_example.lower()
            replaced = re.sub(
                r"([_-])00(?=[_-])",
                lambda m: f"{m.group(1)}{ensemble:02d}",
                name,
                count=1,
            )
            if replaced != name:
                return replaced
        return f"dis_{ensemble:02d}_"

    # Extract-style convention: one NetCDF file corresponds to one ensemble member
    # and is named with a dis_XX_ token. Members without matching files are skipped.
    member_file_pairs: List[Tuple[int, Path]] = []
    sorted_paths = sorted(file_paths)
    for ensemble in range(0, TOTAL_ENSEMBLE_MEMBERS):
        token = _member_token(ensemble)
        matched_path = next(
            (p for p in sorted_paths if token in p.name.lower()),
            None,
        )
        if matched_path is None:
            continue
        member_file_pairs.append((ensemble, matched_path))

    for member, file_path in member_file_pairs:
        with xr.open_dataset(file_path) as ds:
            times = pd.to_datetime(ds[time_dim].values)

            if len(times) == 0:
                continue

            t0 = pd.Timestamp(times[0])
            for t_idx, vt in enumerate(times):
                vt_ts = pd.Timestamp(vt)
                step_h = int((vt_ts - t0).total_seconds() // 3600)
                steps.add(step_h)

                members.add(member)
                key = (vt_ts, member)
                # Store: file path and time index for on-demand reads.
                msg_index[key] = (str(file_path), t_idx)
                vt_lookup[(vt_ts, member)] = (key, None)

    return {
        "msg_index": msg_index,
        "inits": [],
        "steps": sorted(steps),
        "members": sorted(members),
        "tmpl": None,
        "vt_lookup": vt_lookup,
    }


def _open_forecast_source(
    forecast_paths: Sequence[Path],
    forecast_filename_example: Optional[str] = None,
):
    """
    Index multiple NetCDF forecast files without loading into memory.
    
    Returns source metadata and index for on-demand file access.
    """
    import numpy as np
    import xarray as xr

    if not forecast_paths:
        raise RuntimeError("No forecast file paths were provided")

    # Open first file to discover structure
    with xr.open_dataset(forecast_paths[0]) as ds:
        var_name = "dis"
        if var_name not in ds.data_vars:
            raise RuntimeError(
                f"Expected NetCDF variable '{var_name}' not found in: {forecast_paths[0]}"
            )

        da = ds[var_name]
        lat_dim = "lat"
        lon_dim = "lon"
        time_dim = next((d for d in ("valid_time", "time") if d in da.dims), None)

        if lat_dim not in da.dims or lon_dim not in da.dims or time_dim is None:
            raise RuntimeError(
                f"NetCDF variable '{var_name}' must include lat/lon/time dimensions"
            )

        lat_vals = np.asarray(ds[lat_dim].values)
        lon_vals = np.asarray(ds[lon_dim].values)
        if lat_vals.ndim == 2:
            lat_vals = lat_vals[:, 0]
        if lon_vals.ndim == 2:
            lon_vals = lon_vals[0, :]

    # Index all files for (time, member) → (file, indices) mapping
    nc_index = _build_netcdf_index(
        forecast_paths,
        var_name,
        time_dim,
        forecast_filename_example=forecast_filename_example,
    )
    
    source = {
        "file_paths": [str(p) for p in forecast_paths],  # Keep as strings for serialization
        "var_name": var_name,
        "time_dim": time_dim,
    }
    return source, nc_index, lat_vals.astype(float), lon_vals.astype(float)


def _read_forecast_snapshot(
    forecast_source,
    msg_index: dict,
    vt_lookup: dict,
    valid_date: pd.Timestamp,
    member: int,
    init_dt: Optional[pd.Timestamp],
    cell_lat_idx: np.ndarray,
    cell_lon_idx: np.ndarray,
) -> Optional[np.ndarray]:
    """Read discharge values for *member* at *valid_date*, returns array per cell."""
    import xarray as xr
    
    key_data = vt_lookup.get((valid_date, member, init_dt)) if init_dt else None
    if key_data is None:
        hit = vt_lookup.get((valid_date, member))
        if hit is None:
            return None
        key_data = hit[0]
    try:
        msg_num = msg_index.get(key_data)
        if msg_num is None:
            return None
        
        # On-demand file loading for memory efficiency
        file_path, t_idx = msg_num
        with xr.open_dataset(file_path) as ds:
            da = ds[forecast_source["var_name"]]
            data = da.isel({forecast_source["time_dim"]: t_idx}).values
        
        return data[cell_lat_idx, cell_lon_idx].astype(float)
    except Exception as exc:
        logger.debug("Forecast read error for key %s: %s", key_data, exc)
        return None


# ---------------------------------------------------------------------------
# Connected-component flood detection
# ---------------------------------------------------------------------------

def _detect_flood_patches_for_lead(
    forecast_source,
    forecast_index: dict,
    lead_window,
    basin_cells: List[str],
    evt_params,
    grid_index,
    area_grid_km2,
    cell_lat_idx,
    cell_lon_idx,
    t0_years: float,
    a_min_km2: float,
    connectivity: int,
    init_dt,
    support_lat_vals,
    support_lon_vals,
) -> Tuple[List[int], Dict[int, List[Dict[str, Any]]]]:
    """Phase 1: detect flood members and record qualifying event patches.

    Returns
    -------
    (flood_members, patches_by_member)
        patches_by_member[member] = list of patch dicts with valid_date and bbox.
    """
    import numpy as np
    from scipy.ndimage import label, generate_binary_structure

    ep = evt_params.set_index("cell_id")
    cells = [c for c in basin_cells if c in ep.index]
    if not cells:
        return []

    u = ep.loc[cells, "u"].values[None, :]
    sigma = ep.loc[cells, "sigma"].values[None, :]
    xi = ep.loc[cells, "xi"].values[None, :]
    lam = ep.loc[cells, "lam"].values[None, :]

    struct = generate_binary_structure(2, connectivity)
    flood_members: List[int] = []
    patches_by_member: Dict[int, List[Dict[str, Any]]] = {}

    for member in forecast_index["members"]:
        member_patches: List[Dict[str, Any]] = []
        for vdate in lead_window:
            q_vals = _read_forecast_snapshot(
                forecast_source,
                forecast_index["msg_index"],
                forecast_index["vt_lookup"],
                vdate, member, init_dt, cell_lat_idx, cell_lon_idx,
            )
            if q_vals is None:
                continue
            rp = discharge_to_return_period(
                q_vals[None, :], u, sigma, xi, lam
            ).ravel()
            active_cells = rp >= t0_years
            if not active_cells.any():
                continue
            g = np.zeros(grid_index.shape, dtype=bool)
            for k, cid in enumerate(cells):
                if active_cells[k]:
                    positions = np.argwhere(grid_index == k)
                    for pos in positions:
                        g[pos[0], pos[1]] = True
            if not g.any():
                continue
            labs, nlab = label(g, structure=struct)
            if nlab == 0:
                continue
            areas = np.bincount(labs.ravel(), weights=area_grid_km2.ravel())
            if len(areas) <= 1:
                continue
            for lab_id in range(1, len(areas)):
                if areas[lab_id] < a_min_km2:
                    continue
                patch_mask = labs == lab_id
                pos = np.argwhere(patch_mask)
                if pos.size == 0:
                    continue

                lat_idx = pos[:, 0]
                lon_idx = pos[:, 1]
                lat_min = float(np.min(support_lat_vals[lat_idx]))
                lat_max = float(np.max(support_lat_vals[lat_idx]))
                lon_min = float(np.min(support_lon_vals[lon_idx]))
                lon_max = float(np.max(support_lon_vals[lon_idx]))

                member_patches.append(
                    {
                        "valid_date": vdate,
                        "area_km2": float(areas[lab_id]),
                        "bbox": (lon_min, lat_min, lon_max, lat_max),
                    }
                )

        if member_patches:
            flood_members.append(member)
            patches_by_member[member] = member_patches
    return flood_members, patches_by_member


def _render_depth_raster_for_patch(
    discharge_per_cell,
    evt_params,
    spatial: dict,
    bbox: Tuple[float, float, float, float],
    patch_bbox: Tuple[float, float, float, float],
    out_tif: Path,
) -> bool:
    """Build and write an event depth raster clipped to one detected patch bbox."""
    import numpy as np
    import pandas as pd
    import rasterio
    import xarray as xr
    from shapely.geometry import box as sbox

    ep = evt_params.set_index("cell_id")
    cells = [c for c in discharge_per_cell.index if c in ep.index]
    if not cells:
        return False

    q = discharge_per_cell[cells].fillna(0.0).values[None, :]
    rp_vals = discharge_to_return_period(
        q,
        ep.loc[cells, "u"].values[None, :],
        ep.loc[cells, "sigma"].values[None, :],
        ep.loc[cells, "xi"].values[None, :],
        ep.loc[cells, "lam"].values[None, :],
    ).ravel()
    if np.nanmax(rp_vals) < 2.0:
        return False

    rp_df = pd.DataFrame(
        {
            "lat": ep.loc[cells, "lat"].values,
            "lon": ep.loc[cells, "lon"].values,
            "rp": rp_vals,
        }
    )
    lat_vals = np.sort(rp_df["lat"].unique())
    lon_vals = np.sort(rp_df["lon"].unique())
    grid = np.full((len(lat_vals), len(lon_vals)), np.nan, dtype="float32")
    lat_idx = {v: i for i, v in enumerate(lat_vals)}
    lon_idx = {v: j for j, v in enumerate(lon_vals)}
    for _, row in rp_df.iterrows():
        grid[lat_idx[row["lat"]], lon_idx[row["lon"]]] = row["rp"]
    _ = xr.DataArray(
        grid,
        coords={"latitude": lat_vals, "longitude": lon_vals},
        dims=("latitude", "longitude"),
        name="rp",
    )

    rp_to_files = spatial["rp_to_files"]
    minx, miny, maxx, maxy = bbox
    bbox_geom = sbox(minx, miny, maxx, maxy)
    available_rps = sorted(rp for rp in rp_to_files if rp_to_files[rp])
    if not available_rps:
        return False

    from rasterio.merge import merge

    depth_arrays: Dict[int, np.ndarray] = {}
    ref_transform = None
    ref_crs = None
    ref_shape: Optional[Tuple[int, int]] = None
    for rp in available_rps:
        tif_paths = [p for p in rp_to_files[rp] if sbox(*rasterio.open(p).bounds).intersects(bbox_geom)]
        if not tif_paths:
            tif_paths = rp_to_files[rp]
        try:
            with ExitStack() as stack:
                srcs = [stack.enter_context(rasterio.open(p)) for p in tif_paths]
                mosaic, trans = merge(srcs, bounds=(minx, miny, maxx, maxy))
            arr = mosaic[0].astype(np.float32)
            nodata = rasterio.open(tif_paths[0]).nodata
            if nodata is not None:
                arr[arr == nodata] = np.nan
            if ref_transform is None:
                ref_transform = trans
                ref_shape = arr.shape
                with rasterio.open(tif_paths[0]) as src_ref:
                    ref_crs = src_ref.crs
            depth_arrays[rp] = arr
        except Exception as exc:
            logger.debug("JRC load error RP%d: %s", rp, exc)

    if not depth_arrays or ref_shape is None or ref_transform is None:
        return False

    rps_sorted = sorted(depth_arrays)
    event_rp = float(np.nanmax(rp_vals))
    if event_rp <= rps_sorted[0]:
        depth_np = depth_arrays[rps_sorted[0]].copy()
    elif event_rp >= rps_sorted[-1]:
        depth_np = depth_arrays[rps_sorted[-1]].copy()
    else:
        lo = max(r for r in rps_sorted if r <= event_rp)
        hi = min(r for r in rps_sorted if r >= event_rp)
        if lo == hi:
            depth_np = depth_arrays[lo].copy()
        else:
            t = (np.log(event_rp) - np.log(lo)) / (np.log(hi) - np.log(lo))
            d_lo = np.nan_to_num(depth_arrays[lo], nan=0.0)
            d_hi = np.nan_to_num(depth_arrays[hi], nan=0.0)
            depth_np = ((1 - t) * d_lo + t * d_hi).astype(np.float32)

    h, w = ref_shape
    px_x = float(ref_transform.a)
    px_y = float(ref_transform.e)
    x0 = float(ref_transform.c)
    y0 = float(ref_transform.f)
    lon_centers = x0 + (np.arange(w) + 0.5) * px_x
    lat_centers = y0 + (np.arange(h) + 0.5) * px_y

    p_minx, p_miny, p_maxx, p_maxy = patch_bbox
    keep_x = (lon_centers >= p_minx) & (lon_centers <= p_maxx)
    keep_y = (lat_centers >= p_miny) & (lat_centers <= p_maxy)
    keep_mask = np.outer(keep_y, keep_x)
    depth_np = np.where(keep_mask, depth_np, np.nan)

    if not np.isfinite(depth_np).any():
        return False

    out_tif.parent.mkdir(parents=True, exist_ok=True)
    nodata_val = np.float32(-9999.0)
    write_arr = np.where(np.isfinite(depth_np), depth_np, nodata_val).astype(np.float32)
    with rasterio.open(
        out_tif,
        "w",
        driver="GTiff",
        height=h,
        width=w,
        count=1,
        dtype="float32",
        crs=ref_crs,
        transform=ref_transform,
        nodata=nodata_val,
    ) as dst:
        dst.write(write_arr, 1)
    return True


# ---------------------------------------------------------------------------
# Spatial resources loader
# ---------------------------------------------------------------------------

def _load_spatial_resources(settings: "DetectionSettings") -> dict:
    """Load JRC flood maps, WorldPop raster, and admin rasters.

    Returns a dict with keys: jrc_maps, pop_grid, admin_id_raster,
    admin_ids, admin_id_to_name, rp_to_files.
    Raises FileNotFoundError when required paths are missing.
    """
    import rasterio
    import xarray as xr
    import numpy as np
    from pathlib import Path

    jrc_root = Path(settings.jrc_root)
    worldpop_tif = Path(settings.worldpop_tif)
    adm3_geojson = Path(settings.adm3_geojson)

    if not jrc_root.exists():
        raise FileNotFoundError(f"JRC root not found: {jrc_root}")
    if not worldpop_tif.exists():
        raise FileNotFoundError(f"WorldPop TIF not found: {worldpop_tif}")
    if not adm3_geojson.exists():
        raise FileNotFoundError(f"ADM3 GeoJSON not found: {adm3_geojson}")

    # Scan JRC directory
    jrc_tifs = sorted(jrc_root.rglob("*.tif")) + sorted(jrc_root.rglob("*.tiff"))
    rp_to_files: Dict[int, List[Path]] = {}
    _rp_re = re.compile(r"(?:rp|RP|return_period|RP_)(\d+)", re.IGNORECASE)
    for tif in jrc_tifs:
        m = _rp_re.search(tif.stem)
        if m:
            rp = int(m.group(1))
            rp_to_files.setdefault(rp, []).append(tif)

    # Load WorldPop
    with rasterio.open(worldpop_tif) as src:
        pop_grid = src.read(1).astype(np.float32)
        pop_transform = src.transform
        pop_nodata = src.nodata
    if pop_nodata is not None:
        pop_grid[pop_grid == pop_nodata] = np.nan

    # Load admin rasters from GeoJSON (rasterise to WorldPop grid)
    import geopandas as gpd
    from rasterio.features import rasterize
    from rasterio.transform import from_bounds

    adm3_gdf = gpd.read_file(adm3_geojson).to_crs("EPSG:4326")
    
    admin_id_to_name: Dict[int, str] = {}
    expected_name_col = str(settings.adm3_unit_column)
    name_col = None
    for col in adm3_gdf.columns:
        if expected_name_col.lower() in col.lower():
            name_col = col
            break
    if name_col is None:
        raise ValueError(f"Cannot detect ADM3 PCODE column in ADM3 GeoJSON (expected column like {expected_name_col})")

    shapes = []
    for idx_row, row in enumerate(adm3_gdf.itertuples(), start=1):
        admin_id_to_name[idx_row] = str(getattr(row, name_col))
        shapes.append((row.geometry, idx_row))

    with rasterio.open(worldpop_tif) as src:
        admin_id_raster = rasterize(
            shapes=shapes,
            out_shape=src.shape,
            transform=src.transform,
            fill=0,
            dtype="int32",
        )
    admin_ids = np.array(sorted(admin_id_to_name.keys()), dtype="int32")

    return {
        "rp_to_files": rp_to_files,
        "pop_grid": pop_grid,
        "admin_id_raster": admin_id_raster,
        "admin_ids": admin_ids,
        "admin_id_to_name": admin_id_to_name,
        "adm3_gdf": adm3_gdf,
        "name_col": name_col,
    }

# ---------------------------------------------------------------------------
# Main public function
# ---------------------------------------------------------------------------

def detect_flood_events(
    forecast_paths: Union[str, Sequence[str]],
    forecast_filename_example: Optional[str],
    evt_params_path: Path,
    oep_json_path: Path,
    issue_date: date,
    basin_name: str,
    settings: "DetectionSettings",
    lead_days_list: Optional[List[int]],
) -> Tuple[ImpactCube, List[int], List[int]]:
    """Run the spatial flood-event detection algorithm.

    Parameters
    ----------
    forecast_paths:
        Path or list of paths to GloFAS NetCDF forecast file(s) for *issue_date*.
        When multiple files are provided (for example one per ensemble member),
        they are concatenated along a synthetic member dimension.
    evt_params_path:
        Path to ``evt_pot_calibration.parquet``.
    oep_json_path:
        Path to ``oep_curves_all_units.json`` — used to
        read unit names for the impact cube.
    issue_date:
        Forecast initialisation date.
    basin_name:
        Basin name (for logging).
    settings:
        Detection hyper-parameters (t0_years, a_min_km2, etc.).
    lead_days_list:
        Lead times to evaluate (days)

    Returns
    -------
    (impact_cube, ensemble_members, lead_days)
        *impact_cube* is a nested dict compatible with
        ``step3_impact.ImpactCube``.
    """

    logger.info(
        "Detect phase (detect_flood_events) — basin='%s', issue_date=%s, "
        "leads=%s, t0_years=%.1f, a_min_km2=%.0f",
        basin_name, issue_date, lead_days_list,
        settings.t0_years, settings.a_min_km2,
    )

    # --- Load EVT1 parameters ------------------------------------------------
    evt_params_path = Path(evt_params_path)
    if not evt_params_path.exists():
        raise FileNotFoundError(f"EVT1 params not found: {evt_params_path}")

    import numpy as np
    import pandas as pd

    evt_raw = pd.read_parquet(evt_params_path)
    _col_map = {
        "virtual_gauge_id": "cell_id",
        "threshold_m3s": "u",
        "lambda_events_per_year": "lam",
        "gpd_xi": "xi",
        "gpd_sigma": "sigma",
    }
    evt_params = evt_raw.rename(
        columns={k: v for k, v in _col_map.items() if k in evt_raw.columns}
    ).copy()
    required_evt_cols = {"cell_id", "u", "lam", "xi", "sigma"}
    missing_evt_cols = sorted(required_evt_cols.difference(set(evt_params.columns)))
    if missing_evt_cols:
        raise ValueError(
            "EVT parameters file is missing required columns after normalization: "
            f"{missing_evt_cols}. Available columns: {list(evt_raw.columns)}"
        )
    if "lat" not in evt_params.columns or "lon" not in evt_params.columns:
        ll = evt_params["cell_id"].astype(str).apply(_parse_cell_coords)
        evt_params["lat"] = [v[0] for v in ll]
        evt_params["lon"] = [v[1] for v in ll]

    basin_cells = evt_params["cell_id"].astype(str).tolist()
    logger.info("EVT1 params loaded: %d cells for basin '%s'", len(basin_cells), basin_name)

    # --- Build support grid --------------------------------------------------
    lat_vals, lon_vals, grid_index, area_grid, _, _ = _build_support_grid(evt_params)

    # --- Load spatial resources ----------------------------------------------
    spatial = _load_spatial_resources(settings)
    minx = float(lon_vals.min()) - 0.5
    maxx = float(lon_vals.max()) + 0.5
    miny = float(lat_vals.min()) - 0.5
    maxy = float(lat_vals.max()) + 0.5
    bbox = (minx, miny, maxx, maxy)
    spatial["worldpop_tif"] = settings.worldpop_tif

    # --- Read unit names from OEP JSON (to initialise ImpactCube) -----------
    import json as _json
    oep_raw = _json.loads(Path(oep_json_path).read_text(encoding="utf-8"))
    unit_names: List[str] = []
    for rec in oep_raw.get("units", []):
        pcode = rec.get("pcode")
        if pcode:
            unit_names.append(_unit_key(str(pcode)))

    # --- Open forecast source (.nc/.nc4) -----------------------------------
    if isinstance(forecast_paths, str):
        forecast_paths = [Path(forecast_paths)]
    else:
        forecast_paths = [Path(p) for p in forecast_paths]

    if not forecast_paths:
        raise FileNotFoundError("No forecast file paths provided")

    missing = [p for p in forecast_paths if not p.exists()]
    if missing:
        raise FileNotFoundError(f"Forecast file(s) not found: {missing}")

    try:
        forecast_src, forecast_index, forecast_lat1d, forecast_lon1d = _open_forecast_source(
            forecast_paths,
            # settings.forecast_var_name,
            forecast_filename_example=forecast_filename_example,
        )
    except RuntimeError as exc:
        raise RuntimeError(
            f"Failed to index forecast data {forecast_paths}: {exc}"
        ) from exc

    cell_lats = np.array([_parse_cell_coords(c)[0] for c in basin_cells])
    cell_lons = np.array([_parse_cell_coords(c)[1] for c in basin_cells])
    cell_lat_idx = _nearest_index_1d(forecast_lat1d, cell_lats)
    cell_lon_idx = _nearest_index_1d(forecast_lon1d, cell_lons)

    init_dt = pd.Timestamp(issue_date)
    avail_inits = sorted(
        {_dt_from_data_fields(dd, tt) for dd, tt in forecast_index["inits"]}
    )
    if avail_inits and init_dt not in avail_inits:
        nearest_init = min(avail_inits, key=lambda d: abs(d - init_dt))
        logger.warning(
            "init_dt %s not in forecast index; using nearest %s",
            init_dt.date(),
            nearest_init.date(),
        )
        init_dt = nearest_init

    all_members: List[int] = forecast_index["members"]
    impact_cube: ImpactCube = {}

    # --- Main loop: per lead time -------------------------------------------
    with tempfile.TemporaryDirectory(prefix=f"step3_patch_{basin_name}_") as patch_dir:
        patch_dir_path = Path(patch_dir)
        for lead_days in lead_days_list:
            lead_window = pd.date_range(
                init_dt + pd.Timedelta(days=1),
                init_dt + pd.Timedelta(days=lead_days),
                freq="D",
            )

            # Phase 1: detect flood members and event patches via connected components
            flood_members, patches_by_member = _detect_flood_patches_for_lead(
                forecast_source=forecast_src,
                forecast_index=forecast_index,
                lead_window=lead_window,
                basin_cells=basin_cells,
                evt_params=evt_params,
                grid_index=grid_index,
                area_grid_km2=area_grid,
                cell_lat_idx=cell_lat_idx,
                cell_lon_idx=cell_lon_idx,
                t0_years=settings.t0_years,
                a_min_km2=settings.a_min_km2,
                connectivity=settings.cc_connectivity,
                init_dt=init_dt,
                support_lat_vals=lat_vals,
                support_lon_vals=lon_vals,
            )

            n_patches = sum(len(v) for v in patches_by_member.values())
            logger.info(
                f"Lead {lead_days:2d}d: "
                f"{len(flood_members)}/{len(all_members)} flood members, "
                f"{n_patches} qualifying patches"
            )

            lead_patch_inputs: List[EventPatchImpactInput] = []
            for member in flood_members:
                for patch_idx, patch_meta in enumerate(patches_by_member.get(member, []), start=1):
                    vdate = patch_meta["valid_date"]
                    q_vals = _read_forecast_snapshot(
                        forecast_src,
                        forecast_index["msg_index"],
                        forecast_index["vt_lookup"],
                        vdate,
                        member,
                        init_dt,
                        cell_lat_idx,
                        cell_lon_idx,
                    )
                    if q_vals is None:
                        continue

                    q_snapshot = pd.Series(q_vals, index=basin_cells)
                    depth_raster = patch_dir_path / (
                        f"{basin_name}_lead{lead_days:02d}_m{member:03d}_patch{patch_idx:03d}.tif"
                    )
                    ok = _render_depth_raster_for_patch(
                        discharge_per_cell=q_snapshot,
                        evt_params=evt_params,
                        spatial=spatial,
                        bbox=bbox,
                        patch_bbox=patch_meta["bbox"],
                        out_tif=depth_raster,
                    )
                    if not ok:
                        continue

                    patch_event_id = (
                        f"{basin_name}_lead{lead_days:02d}_m{member:03d}_patch{patch_idx:03d}"
                    )
                    for rp in settings.flood_detect_rps:
                        lead_patch_inputs.append(
                            EventPatchImpactInput(
                                lead_day=lead_days,
                                member_id=member,
                                rp=int(rp),
                                depth_raster=depth_raster,
                                event_id=f"{patch_event_id}_rp{int(rp)}",
                                extra={
                                    "admin_id_raster": spatial["admin_id_raster"],
                                    "id_to_name": spatial["admin_id_to_name"],
                                },
                            )
                        )

            if lead_patch_inputs:
                _, _, lead_cube = _compute_impacts_from_event_patches(
                    patches=lead_patch_inputs,
                    worldpop_tif=Path(settings.worldpop_tif),
                    depth_threshold_m=settings.depth_threshold_m,
                )

                for unit, by_lead in lead_cube.items():
                    for lead, by_member in by_lead.items():
                        for member, by_rp in by_member.items():
                            dst_rp = (
                                impact_cube
                                .setdefault(unit, {})
                                .setdefault(lead, {})
                                .setdefault(member, {})
                            )
                            for rp, val in by_rp.items():
                                dst_rp[int(rp)] = dst_rp.get(int(rp), 0.0) + float(val)

            # Ensure full cube coverage (all unit/member/rp combinations) for this lead.
            for member in all_members:
                for unit in unit_names:
                    dst_rp = impact_cube.setdefault(unit, {}).setdefault(lead_days, {}).setdefault(member, {})
                    for rp in settings.flood_detect_rps:
                        dst_rp.setdefault(int(rp), 0.0)

    logger.info(
        "Detect phase complete — %d units, %d lead days, %d members",
        len(impact_cube), len(lead_days_list), len(all_members),
    )
    return impact_cube, all_members, lead_days_list


def _compute_prob_exceed(
    cube: Dict[str, Dict[int, Dict[int, Dict[int, float]]]],
    thresholds: Dict[str, Dict[int, float]],
    members: Iterable[int],
) -> Dict[str, Dict[int, Dict[int, float]]]:
    """Compute the fraction of ensemble members exceeding OEP thresholds.

    Returns
    -------
    dict
        ``{unit_id: {lead_day: {rp: probability}}}``
    """
    member_set = sorted(set(int(m) for m in members))
    n_members = len(member_set)
    if n_members == 0:
        logger.warning("No ensemble members — returning empty exceedance dict")
        return {}

    logger.info(
        "Computing exceedance probabilities: %d cube units, %d ensemble members",
        len(cube),
        n_members,
    )
    out: Dict[str, Dict[int, Dict[int, float]]] = {}
    for unit, per_lead in cube.items():
        unit_thresholds = thresholds.get(unit, {})
        if not unit_thresholds:
            continue
        for lead, per_member in per_lead.items():
            for rp, thr in unit_thresholds.items():
                exceed = sum(
                    1
                    for member in member_set
                    if per_member.get(member, {}).get(rp, 0.0) >= thr
                )
                out.setdefault(unit, {}).setdefault(lead, {})[rp] = exceed / n_members

    logger.info("Exceedance probabilities computed for %d qualifying units", len(out))
    return out
