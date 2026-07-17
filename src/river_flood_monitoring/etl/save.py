"""Save stage for daily flood monitoring ETL.

This module serializes basin decisions and optional map products for
downstream systems and reporting workflows.
"""

from __future__ import annotations

from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, List
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt

from river_flood_monitoring.logging import get_logger
from .run_spec import PipelineRunSpec
from .utils import BasinRunOutput, expand_template

logger = get_logger(__name__)


def save(
    run_spec: PipelineRunSpec,
    issue_date: date,
    basin_forecasts: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Format basin outputs and persist final trigger files."""

    basin_results = _prepare_trigger_decision_metadata(
        run_spec=run_spec,
        issue_date=issue_date,
        basin_forecasts=basin_forecasts,
    )
    output_ctx = _create_save_output_context(run_spec=run_spec, issue_date=issue_date)

    logger.info(
        "Writing output: %d basins, format=csv, dir=%s",
        len(basin_results),
        output_ctx["output_dir"],
    )

    # Prepare trigger decisions split into activation (ADM2) and operational_information (ADM3)
    trigger_dfs = _prepare_trigger_decision_records(
        basin_results=basin_results,
        timestamp=output_ctx["timestamp"],
    )
    activation_df = trigger_dfs["activation"]
    operational_information_df = trigger_dfs["operational_information"]

    # Save activation (ADM2) file
    activation_file = None
    if not activation_df.empty:
        activation_file = _save_activation_decisions(
            activation_df=activation_df,
            output_dir=output_ctx["output_dir"],
            issue_date=issue_date,
            timestamp=output_ctx["timestamp"],
        )

    # Save operational_information (ADM3) file
    operational_information_file = None
    if not operational_information_df.empty:
        operational_information_file = _save_operational_information(
            operational_information_df=operational_information_df,
            output_dir=output_ctx["output_dir"],
            issue_date=issue_date,
            timestamp=output_ctx["timestamp"],
        )

    decision_summary_payload = _prepare_decision_summary(basin_results)
    decision_summary_file = None
    if decision_summary_payload is not None:
        decision_summary_file = _save_decision_summary_file(
            output_ctx=output_ctx,
            decision_summary_payload=decision_summary_payload,
        )
    else:
        logger.info("No activated tiers found; skipping decision summary file")

    # Use operational_information (ADM3 only) for plotting
    fired_df = _prepare_trigger_decisions_for_plotting(trigger_df=operational_information_df)

    map_plots = _plot_population_exposed(
        run_spec=run_spec,
        trigger_df=operational_information_df,
        fired_df=fired_df,
    )
    map_files = _save_maps(
        map_plots=map_plots,
        output_dir=output_ctx["output_dir"],
    )

    return {
        "basin_results": basin_results,
        "activation_file": activation_file,
        "operational_information_file": operational_information_file,
        "decision_summary_file": decision_summary_file,
        "map_files": map_files,
    }


def _prepare_trigger_decision_metadata(
    run_spec: PipelineRunSpec,
    issue_date: date,
    basin_forecasts: List[Dict[str, Any]],
) -> List[BasinRunOutput]:
    """Create standardized basin decision records for all output writers."""
    basin_results: List[BasinRunOutput] = []
    for basin in basin_forecasts:
        metadata = {
            "rule_tiers": [
                {
                    "name": r.name,
                    "rp": r.rp,
                    "p_thr": r.p_thr,
                    "n_req": r.n_req,
                    "label": r.label,
                }
                for r in run_spec.decision.rules
            ],
            "persist_days": run_spec.decision.persist_days,
            "min_lead": run_spec.decision.min_lead,
            "oep_min": run_spec.decision.oep_min,
            "oep_source": str(basin["oep_path"]),
            "impacts_source": basin["impacts_source"],
        }

        basin_results.append(
            BasinRunOutput(
                basin_name=str(basin["basin_name"]),
                issue_date=issue_date.isoformat(),
                forecast_paths=basin["forecast_paths"],
                units=basin["units"],
                metadata=metadata,
            )
        )
    return basin_results


def _create_save_output_context(run_spec: PipelineRunSpec, issue_date: date) -> Dict[str, Any]:
    """Create common output context shared by all save steps."""
    if run_spec.output is None:
        raise ValueError("Run spec must define output.output_dir_template")

    output_dir = Path(expand_template(run_spec.output.output_dir_template, issue_date))
    output_dir.mkdir(parents=True, exist_ok=True)

    return {
        "output_dir": output_dir,
        "timestamp": datetime.utcnow().strftime("%Y%m%dT%H%M%SZ"),
    }

def _prepare_trigger_decision_records(
    basin_results: List[BasinRunOutput],
    timestamp: str,
) -> Dict[str, pd.DataFrame]:
    """Create and split trigger decisions into activation (ADM2) and operational_information (ADM3).

    Returns dict with 'activation' and 'operational_information' DataFrames.
    Both include issue_time column and severity_rp (renamed from rp), without tier column.

    Args:
        basin_results: List of basin outputs with trigger decisions
        timestamp: UTC timestamp for issue_time column
    """
    # Build rows without tier column, with issue_time and severity_rp
    rows = []
    for basin in basin_results:
        for unit in basin.units:
            for tier in unit.tiers:
                rows.append(
                    {
                        "issue_date": basin.issue_date,
                        "issue_time": timestamp,
                        "basin_name": basin.basin_name,
                        "level": unit.level,
                        "name": unit.name,
                        "pcode": unit.pcode,
                        "severity_rp": tier.rp,
                        "p_threshold": tier.p_threshold,
                        "fired": tier.fired,
                        "fire_lead": tier.fire_lead,
                        "probability_at_fire": tier.probability_at_fire,
                        "impact_population_threshold": tier.impact_population_threshold,
                        "impact_population_at_fire": tier.impact_population_at_fire,
                    }
                )
    
    if not rows:
        return {
            "activation": pd.DataFrame(),
            "operational_information": pd.DataFrame(),
        }
    
    filtered_df = pd.DataFrame(rows)

    # Split: ADM2 (activation) vs ADM3 (operational_information)
    activation_df = filtered_df[
        filtered_df["level"].astype(str).str.upper() == "ADM2"
    ].copy()
    
    operational_information_df = filtered_df[
        filtered_df["level"].astype(str).str.upper() == "ADM3"
    ].copy()
    
    return {
        "activation": activation_df,
        "operational_information": operational_information_df,
    }


def _save_activation_decisions(
    activation_df: pd.DataFrame,
    output_dir: Path,
    issue_date: date,
    timestamp: str,
) -> Path:
    """Persist activation (ADM2) DataFrame as CSV."""
    out_file = output_dir / f"activation_{issue_date.isoformat()}_{timestamp}.csv"
    activation_df.to_csv(out_file, index=False)
    logger.info("Activation file written: %s", out_file)
    return out_file


def _save_operational_information(
    operational_information_df: pd.DataFrame,
    output_dir: Path,
    issue_date: date,
    timestamp: str,
) -> Path:
    """Persist operational_information (ADM3) DataFrame as CSV."""
    out_file = output_dir / f"operational_information_{issue_date.isoformat()}_{timestamp}.csv"
    operational_information_df.to_cvs(out_file, index=False)
    logger.info("Operational information file written: %s", out_file)
    return out_file


def _prepare_decision_summary(basin_results: List[BasinRunOutput]) -> Dict[str, Any] | None:
    """Create decision summary content for downstream automation."""
    total_fired = sum(
        1
        for basin in basin_results
        for unit in basin.units
        for tier in unit.tiers
        if tier.fired
    )
    if total_fired <= 0:
        return None

    return {
        "file_name": "decision.txt",
        "text": "triggered=True\n",
        "triggered": True,
    }


def _save_decision_summary_file(
    output_ctx: Dict[str, Any],
    decision_summary_payload: Dict[str, Any],
) -> Path:
    """Persist the summary decision file (`decision.txt`)."""
    decision_file = output_ctx["output_dir"] / str(decision_summary_payload["file_name"])
    decision_file.write_text(str(decision_summary_payload["text"]), encoding="utf-8")
    logger.info(
        "Decision output written: %s (triggered=%s)",
        decision_file,
        decision_summary_payload["triggered"],
    )
    return decision_file


def _prepare_trigger_decisions_for_plotting(trigger_df: pd.DataFrame) -> pd.DataFrame:
    """Prepare activated ADM3 trigger rows with one highest severity_rp record per basin/lead/unit."""
    if trigger_df.empty:
        logger.info("Skipping map plotting: CSV has no rows")
        return pd.DataFrame()

    # Keep only activated rows that have a valid lead day.
    fired_df = trigger_df.loc[trigger_df["fired"].astype(str).str.lower() == "true"].copy()
    fired_df = fired_df.loc[fired_df["fire_lead"].notna()].copy()
    if fired_df.empty:
        logger.info("Skipping map plotting: no activated rows in CSV")
        return pd.DataFrame()

    fired_df["fire_lead"] = fired_df["fire_lead"].astype(int)
    fired_df = fired_df.loc[fired_df["level"].astype(str).str.upper() == "ADM3"].copy()
    if fired_df.empty:
        logger.info("Skipping map plotting: no ADM3 activated rows in CSV")
        return pd.DataFrame()

    fired_df["pcode"] = fired_df["pcode"].astype(str).str.strip()

    # Map severity_rp values to rank (2→1, 5→2, 10→3 for T1, T2, T3)
    severity_rank = {2: 1, 5: 2, 10: 3}
    fired_df["severity_rank"] = fired_df["severity_rp"].map(severity_rank).fillna(0).astype(int)

    # Select one severity per unit (highest) for each basin and lead.
    fired_df = (
        fired_df.sort_values("severity_rank")
        .groupby(["basin_name", "fire_lead", "pcode"], as_index=False)
        .last()
    )

    return fired_df


def _plot_activated_areas(
    run_spec: PipelineRunSpec,
    trigger_df: pd.DataFrame,
    fired_df: pd.DataFrame,
) -> List[Any]:
    """Create plot objects for activated admin areas. Returns list of (figure, filename) tuples."""
    adm3_geojson_path = Path(run_spec.detection.adm3_geojson)
    if not adm3_geojson_path.exists():
        logger.warning("Skipping map plotting: ADM3 GeoJSON not found: %s", adm3_geojson_path)
        return []

    if fired_df.empty:
        return []

    # Map severity_rp values to colors (2→T1, 5→T2, 10→T3)
    severity_colors = {2: "#FFD54F", 5: "#FB8C00", 10: "#D32F2F"}

    adm3_gdf = gpd.read_file(adm3_geojson_path)
    expected_unit_col = "adm3_pcode"
    pcode_col = None
    for col in adm3_gdf.columns:
        if expected_unit_col.lower() in col.lower():
            pcode_col = col
            break
    if pcode_col is None:
        raise ValueError(
            "Cannot detect admin unit column in ADM3 GeoJSON "
            f"(expected column like {expected_unit_col})"
        )

    admin_gdf = adm3_gdf[[pcode_col, "geometry"]].copy()
    admin_gdf = admin_gdf.rename(columns={pcode_col: "adm3_pcode"})
    admin_gdf["adm3_pcode"] = admin_gdf["adm3_pcode"].astype(str).str.strip()

    plot_specs: List[tuple[Any, str]] = []

    # Generate plot specs for each basin and activated lead day.
    for (basin_name, fire_lead), lead_rows in fired_df.groupby(["basin_name", "fire_lead"]):
        basin_rows = trigger_df.loc[
            (trigger_df["basin_name"] == basin_name)
            & (trigger_df["level"].astype(str).str.upper() == "ADM3")
        ].copy()
        basin_rows["pcode"] = basin_rows["pcode"].astype(str).str.strip()

        basin_units = basin_rows["pcode"].dropna().unique().tolist()
        basin_gdf = admin_gdf.loc[admin_gdf["adm3_pcode"].isin(basin_units)].copy()
        if basin_gdf.empty:
            logger.warning(
                "Skipping map for basin=%s lead=%s: no admin geometry matched",
                basin_name,
                fire_lead,
            )
            continue

        merged = basin_gdf.merge(
            lead_rows[["pcode", "severity_rp"]],
            left_on="adm3_pcode",
            right_on="pcode",
            how="left",
        )
        merged["color"] = merged["severity_rp"].map(severity_colors)

        fig, ax = plt.subplots(figsize=(9, 9))

        # Non-activated units are boundaries only.
        merged.boundary.plot(ax=ax, color="#4D4D4D", linewidth=0.5)

        for rp, color in severity_colors.items():
            severity_gdf = merged.loc[merged["severity_rp"] == rp]
            if not severity_gdf.empty:
                severity_gdf.plot(ax=ax, color=color, edgecolor="#333333", linewidth=0.5)

        ax.set_axis_off()
        ax.set_title(f"{basin_name} | activated alerts at lead day {fire_lead}")

        severity_labels = {2: "T1 (RP2)", 5: "T2 (RP5)", 10: "T3 (RP10)"}
        legend_handles = [
            plt.Line2D(
                [0],
                [0],
                marker="s",
                color="w",
                markerfacecolor=severity_colors[2],
                markersize=10,
                label=severity_labels[2],
            ),
            plt.Line2D(
                [0],
                [0],
                marker="s",
                color="w",
                markerfacecolor=severity_colors[5],
                markersize=10,
                label=severity_labels[5],
            ),
            plt.Line2D(
                [0],
                [0],
                marker="s",
                color="w",
                markerfacecolor=severity_colors[10],
                markersize=10,
                label=severity_labels[10],
            ),
            plt.Line2D(
                [0],
                [0],
                color="#4D4D4D",
                linewidth=1,
                label="Not activated",
            ),
        ]
        ax.legend(handles=legend_handles, loc="lower left", frameon=True)
        fig.tight_layout()

        map_filename = f"{basin_name}_lead{fire_lead}_activated_map.png"
        plot_specs.append((fig, map_filename))

    logger.info("Activated area plotting complete: %d plot(s) created", len(plot_specs))
    return plot_specs


def _plot_population_exposed(
    run_spec: PipelineRunSpec,
    trigger_df: pd.DataFrame,
    fired_df: pd.DataFrame,
) -> List[Any]:
    """Create plot objects for population exposed in activated admin areas."""
    adm3_geojson_path = Path(run_spec.detection.adm3_geojson)
    if not adm3_geojson_path.exists():
        logger.warning("Skipping population map plotting: ADM3 GeoJSON not found: %s", adm3_geojson_path)
        return []

    if fired_df.empty:
        return []

    pop_col = "impact_population_at_fire"
    if pop_col not in fired_df.columns:
        logger.warning("Skipping population map plotting: missing column '%s' in trigger data", pop_col)
        return []

    adm3_gdf = gpd.read_file(adm3_geojson_path)
    expected_unit_col = "adm3_pcode"
    pcode_col = None
    for col in adm3_gdf.columns:
        if expected_unit_col.lower() in col.lower():
            pcode_col = col
            break
    if pcode_col is None:
        raise ValueError(
            "Cannot detect admin unit column in ADM3 GeoJSON "
            f"(expected column like {expected_unit_col})"
        )

    admin_gdf = adm3_gdf[[pcode_col, "geometry"]].copy()
    admin_gdf = admin_gdf.rename(columns={pcode_col: "adm3_pcode"})
    admin_gdf["adm3_pcode"] = admin_gdf["adm3_pcode"].astype(str).str.strip()

    plot_specs: List[tuple[Any, str]] = []

    # Generate population-exposed map specs for each basin and activated lead day.
    for (basin_name, fire_lead), lead_rows in fired_df.groupby(["basin_name", "fire_lead"]):
        basin_rows = trigger_df.loc[
            (trigger_df["basin_name"] == basin_name)
            & (trigger_df["level"].astype(str).str.upper() == "ADM3")
        ].copy()
        basin_rows["pcode"] = basin_rows["pcode"].astype(str).str.strip()

        basin_units = basin_rows["pcode"].dropna().unique().tolist()
        basin_gdf = admin_gdf.loc[admin_gdf["adm3_pcode"].isin(basin_units)].copy()
        if basin_gdf.empty:
            logger.warning(
                "Skipping population map for basin=%s lead=%s: no admin geometry matched",
                basin_name,
                fire_lead,
            )
            continue

        lead_population = lead_rows[["pcode", pop_col]].copy()
        lead_population[pop_col] = pd.to_numeric(lead_population[pop_col], errors="coerce")

        merged = basin_gdf.merge(
            lead_population,
            left_on="adm3_pcode",
            right_on="pcode",
            how="left",
        )

        fig, ax = plt.subplots(figsize=(9, 9))

        # Show all basin units as boundaries; fill activated units by population exposed.
        merged.boundary.plot(ax=ax, color="#4D4D4D", linewidth=0.5)

        activated = merged.loc[merged[pop_col].notna()].copy()
        if not activated.empty:
            activated.plot(
                ax=ax,
                column=pop_col,
                cmap="YlOrRd",
                edgecolor="#333333",
                linewidth=0.5,
                legend=True,
                legend_kwds={"label": "Population exposed"},
            )

        ax.set_axis_off()
        ax.set_title(f"{basin_name} | population exposed at lead day {fire_lead}")
        fig.tight_layout()

        map_filename = f"{basin_name}_lead{fire_lead}_population_exposed_map.png"
        plot_specs.append((fig, map_filename))

    logger.info("Population exposed plotting complete: %d plot(s) created", len(plot_specs))
    return plot_specs


def _save_maps(
    map_plots: List[Any],
    output_dir: Path,
) -> List[Path]:
    """Save plot objects to disk."""
    output_dir = Path(output_dir)
    map_dir = output_dir / "maps"
    map_dir.mkdir(parents=True, exist_ok=True)
    map_paths: List[Path] = []

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        pass

    for fig, filename in map_plots:
        map_file = map_dir / filename
        
        fig.savefig(map_file, dpi=150, bbox_inches="tight")
        plt.close(fig)
        map_paths.append(map_file)

    logger.info("Map files saved: %d map(s) written to %s", len(map_paths), map_dir)
    return map_paths
