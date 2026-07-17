# River flood monitor workflow

A Python tool for monitoring riverine forecast developed by NLRC to support Start Network Philippines.

Code development assisted by: [Copilot]

## Setup

### 1. Install `uv`

`uv` is a Python tool for managing virtual environments and dependencies. It will create a `.venv` with Python 3.13 and all required packages for this project.

**Windows (PowerShell):**
```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

**macOS / Linux:**
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Or via pip: `pip install uv`

### 2. Install dependencies

```bash
uv sync
```

This creates a `.venv` with all required runtime dependencies (including
scientific packages such as numpy, pandas, xarray, scipy, geopandas, and rasterio).

For cross-platform bootstrap helpers that also install Python and `cfgrib`:

```powershell
.\uv-sync.ps1
```

```bash
./uv-sync.sh
```

### 3. Run the workflow

The workflow is running as the following steps in order:
- Extract
- Forecast
- Save

#### Run the CLI tool

For full workflow run:
```bash
uv run flood-monitoring `
    --run-spec config/run_specs/daily_monitoring.yaml `
    --basins cagayan
```

The workflow also supports modular execution. If only specific steps (extract, forecast, or save) need to be executed instead of rerunning the full pipeline each time, intermediate artifacts are cached by run name and issue date. That lets downstream steps resume from prior successful outputs for faster debugging and iteration.

This can be done by adding flags `--<step>` to the cli.
```bash
uv run flood-monitoring `
    --run-spec config/run_specs/daily_monitoring.yaml `
    --basins cagayan `
    --extract
```

You can also run multiple basins at once:
```bash
uv run flood-monitoring `
    --run-spec config/run_specs/daily_monitoring.yaml `
    --basins cagayan abra ilocos
```

### Repo structure:

```
river-flood-workflow/
├── pyproject.toml                 project dependencies
├── uv.lock                        locked dependency versions
├── README.md
├── uv-sync.ps1                    Windows bootstrap
├── uv-sync.sh                     Linux/macOS bootstrap
│
├── config/
│   ├── risk_profiles/
│   │   ├── bicol_oep_curves_all_units.json
│   │   └── cagayan_oep_curves_all_units.json
│   └── run_specs/
│       ├── daily_monitoring.yaml
│       └── daily_monitoring.template.yaml
│
├── data/
│   ├── admin-areas
│   ├── bronze/
│   │   ├── glofas/forecast/
│   │   ├── jrc_flood_maps/
│   │   └── population/
│   ├── silver/
│   ├── etl_step_cache/
│   └── gold/
├── data_update/
│   ├── download_jrc_flood_tiles.py
│   └── README.md
│ 
├── logs/
│   └── YYYY/MM/                   run logs by year/month
│
└──  src/river_flood_monitoring/
    ├──  __init__.py
    ├── cli.py                     console-script: flood-monitoring
    ├── config.py                  config loading helpers
    ├── logging.py                 logging helpers
    └── etl/
        ├── __init__.py
        ├── extract-example.py
        ├── extract.py
        ├── forecast.py
        ├── pipeline.py
        ├── prepare.py
        ├── run_spec.py
        ├── save.py
        └── utils.py
```