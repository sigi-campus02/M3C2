# M3C2 Point Cloud Processing Pipeline

This project provides a Python workflow for comparing 3D point cloud pairs using the Multiscale Model to Model Cloud Comparison (M3C2) algorithm. The pipeline automates parameter estimation, distance computation, statistical analysis, and visualization.

## Features
- Load point clouds from multiple formats (.xyz, .las/.laz, .ply, .obj, .gpc)
- Estimate optimal normal and search radii
- Execute the M3C2 algorithm via [py4dgeo](https://github.com/py4dgeo/py4dgeo)
- Detect and exclude outliers with configurable strategies
- Compute rich statistical metrics and export results to Excel or JSON
- Configure processing and reporting through `config.json` validated by `config.schema.json`
- Generate comparison reports via the lightweight `report_pipeline` CLI
- Orchestrate batch processing across many point cloud pairs

## Installation
Install Python dependencies:

```bash
pip install -r requirements.txt
```

Tkinter must be available as a system package. Install it using your platform's
package manager:

- **Debian/Ubuntu:** `sudo apt install python3-tk`
- **Fedora/RHEL:** `sudo dnf install python3-tkinter`
- **Arch Linux:** `sudo pacman -S tk`
- **macOS (Homebrew):** `brew install python-tk`
- **Windows:** included with the standard Python installer

## Usage
Configuration is supplied via `config.json` (validated by `config.schema.json`).
Run the pipeline with the configuration file:

```bash
python -m main
```

Alternatively, pass parameters directly on the command line:

```bash
python -m main --data_dir ./data --folders 0342-0349 --filename_reference reference.ply --filename_comparison comparison.ply
```

Outputs, statistics, and logs are written to dataset folders and the `outputs/` directory.

## CLI Examples

### Single Cloud Statistics

```bash
python -m main --data_dir ./data --folders scan1 --filename_singlecloud surface.las --stats_singleordistance single
python -m main --data_dir ./data --folders scanA scanB --filename_singlecloud points.ply --stats_singleordistance single --output_format json
```

### Distance Metrics

```bash
python -m main --data_dir ./data --folders pair_01 --filename_reference ref.ply --filename_comparison cmp.ply --stats_singleordistance distance
python -m main --data_dir ./data --folders pair_A pair_B --filename_reference ref.las --filename_comparison cmp.las --stats_singleordistance distance --use_subsampled_corepoints 5 --outlier_detection_method nmad
```

### Plot Generation

The reporting workflow is exposed through the `report_pipeline` commands. Invoke
it either with `python -m report_pipeline` or, when installed as a package, via
the `m3c2-report` console script. Some typical commands are:

```bash
python -m report_pipeline folder --folder results/case_07 --pattern "*_distances.txt" --out case_07.pdf
python -m report_pipeline multifolder --folders results/c1 results/c2 --pattern "*_dist.txt" --paired --out all_cases.pdf
python -m report_pipeline files --files a1.txt b1.txt a2.txt --out ai_overlay.pdf --legend
```

## Logging

Control verbosity with the `--log_level` option or by setting the `LOG_LEVEL`
environment variable, which takes precedence over the value in `config.json`.
For example:

```bash
export LOG_LEVEL=DEBUG
python -m main
```

If neither is provided, the log level defaults to the `logging.level` entry in
`config.json`.

## Repository Structure
- `config/` – dataclasses and plotting configuration
- `datasource/` – unified loading of point cloud data
- `generation/` – utilities for producing derived clouds
- `orchestration/` – batch runners and M3C2 execution logic
- `report_pipeline/` – lightweight CLI for generating comparison reports
- `tests/` – unit tests covering key functionality

## Configuration

Global settings for the pipeline and report generation live in `config.json` and are validated against `config.schema.json`. The `arguments` section controls the main pipeline, while sections prefixed with `arguments_plot_` configure the `report_pipeline` CLI.

### Logging

| Parameter            | Type    | Default / Example                                   | Description |
|----------------------|---------|----------------------------------------------------|-------------|
| `logging.level`      | string  | `"INFO"`                                           | Logging level (`DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL`). |
| `logging.file`       | string  | `"logs/orchestration.log"`                         | File path for log output. |
| `logging.format`     | string  | `"%(asctime)s [%(levelname)-8s] %(name)s - %(message)s"` | Format string for log messages. |
| `logging.date_format`| string  | `"%Y-%m-%d %H:%M:%S"`                              | Date format for log messages. |

---

### Arguments

| Parameter                 | Type           | Default / Example | Description |
|---------------------------|----------------|------------------|-------------|
| `data_dir`                | string         | `"data"` | Input directory containing point cloud folders. |
| `folders`                 | list\[string]  | `["pointclouds"]`      | List of folders containing point clouds. |
| `filename_reference`            | string         | `""`             | Reference point cloud file for distance calculation (not used for single cloud statistics). |
| `filename_comparison`            | string         | `""`             | Comparison point cloud file for distance calculation (not used for single cloud statistics). |
| `filename_singlecloud`    | string         | `""` | Single point cloud filename used for single-cloud statistics. |
| `use_subsampled_corepoints` | int          | `1`              | Use subsampled core points for distance computation. Eg. 3: Every 3rd point is used for the subsample. |
| `sample_size`             | int            | `10000`          | Number of core points used for parameter estimation (normal & projection scale). Not used to subsample distances. |
| `scale_strategy`          | string         | `"radius"`       | Strategy used for parameter estimation (currently only 'radius' is supported). |
| `only_stats`              | boolean        | `true`           | If true, compute only statistics (no parameter estimation or M3C2 distance calculation). |
| `stats_singleordistance`  | string         | `"single"`       | "Statistics type: 'single' for single-cloud statistics or 'distance' for pairwise distance analysis." |
| `output_format`           | string         | `"excel"`        | Output format for results. |
| `project`                 | string         | `"PROJECT"`     | Project name used as prefix for folders and output files. |
| `normal_override`         | float / null   | `null`           | Manual override for normal scale instead of automatic estimation. |
| `proj_override`           | float / null   | `null`           | Manual override for projection scale instead of automatic estimation. |
| `use_existing_params`     | boolean        | `false`          | If true, use existing parameter file instead of recalculating. |
| `outlier_detection_method`| string         | `"rmse"`         | Outlier detection method (`rmse`, `mad`, `nmad`, `std`). |
| `outlier_multiplicator`   | float          | `3.0`            | Multiplicator applied by the chosen outlier method. |
| `log_level`               | string         | `"INFO"`         | Logging level for output (overrides `logging.level`). |
