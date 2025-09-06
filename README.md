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

### Example Point Clouds

The repository ships sample LAZ files in `data/examplepointclouds` with one
reference cloud (`reference.laz`) and four comparison clouds
(`comparison1.laz` – `comparison4.laz`). The following commands demonstrate how
to invoke the main pipeline with these files.

#### Single-cloud statistics

The examples below use `reference.laz`; replace it with `comparison1.laz`
through `comparison4.laz` to process the other sample clouds.

```bash
# Recalculate scales for this run
python -m main \
  --stats_singleordistance single \
  --data_dir ./data \
  --folders examplepointclouds \
  --filename_singlecloud reference.laz \
  --use_subsampled_corepoints 1 \
  --sample_size 10000 \
  --scale_strategy radius \
  --only_stats \
  --output_format excel \
  --project EXAMPLE \
  --use_existing_params false

# Override normal and projection scales

python -m main \
  --stats_singleordistance single \
  --data_dir ./data \
  --folders examplepointclouds \
  --filename_singlecloud reference.laz \
  --use_subsampled_corepoints 1 \
  --sample_size 10000 \
  --scale_strategy radius \

  --only_stats \
  --output_format excel \
  --project EXAMPLE \
  --normal_override 0.1 \
  --proj_override 0.2 \
  --use_existing_params false

# Reuse previously saved parameters
python -m main \
  --stats_singleordistance single \
  --data_dir ./data \
  --folders examplepointclouds \
  --filename_singlecloud reference.laz \
  --use_subsampled_corepoints 1 \
  --sample_size 10000 \
  --scale_strategy radius \
  --only_stats \
  --output_format excel \
  --project EXAMPLE \
  --use_existing_params true
```

#### Distance comparisons


Comparison commands use `comparison1.laz` as the example target; swap in
`comparison2.laz`–`comparison4.laz` for the other datasets.

```bash
# Recalculate scales before measuring distances
python -m main \
  --stats_singleordistance distance \
  --data_dir ./data \
  --folders examplepointclouds \
  --filename_reference reference.laz \
  --filename_comparison comparison1.laz \
  --use_subsampled_corepoints 5 \
  --sample_size 10000 \
  --scale_strategy radius \
  --no-only_stats \
  --output_format json \
  --project EXAMPLE \
  --use_existing_params false \
  --outlier_detection_method nmad \
  --outlier_multiplicator 3.0

# Override normal and projection scales
python -m main \
  --stats_singleordistance distance \
  --data_dir ./data \
  --folders examplepointclouds \
  --filename_reference reference.laz \
  --filename_comparison comparison1.laz \
  --use_subsampled_corepoints 5 \
  --sample_size 10000 \
  --scale_strategy radius \
  --no-only_stats \
  --output_format json \
  --project EXAMPLE \
  --normal_override 0.1 \
  --proj_override 0.2 \
  --use_existing_params false \
  --outlier_detection_method nmad \
  --outlier_multiplicator 3.0

# Reuse parameters from a previous run
python -m main \
  --stats_singleordistance distance \
  --data_dir ./data \
  --folders examplepointclouds \
  --filename_reference reference.laz \
  --filename_comparison comparison1.laz \
  --use_subsampled_corepoints 5 \
  --sample_size 10000 \
  --scale_strategy radius \
  --no-only_stats \
  --output_format json \
  --project EXAMPLE \
  --use_existing_params true \
  --outlier_detection_method nmad \
  --outlier_multiplicator 3.0
```

#### Plot commands

After computing distances, the following commands visualize the results.

```bash
# Plot with the main CLI
python -m main \
  --stats_singleordistance plot \
  --data_dir ./data \
  --folders examplepointclouds \
  --filename_reference reference.laz \
  --filename_comparison comparison1.laz \
  --project EXAMPLE

# Create a PDF report
python -m report_pipeline folder \
  --folder outputs/EXAMPLE_output \
  --pattern '*reference_comparison1_distances.txt' \
  --out EXAMPLE_reference_comparison1.pdf
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
- `m3c2/`
  - `cli/` – command-line interface and GUI helpers
  - `config/` – dataclasses and plotting configuration
  - `importer/` – unified loading of point cloud data
  - `exporter/` – utilities for writing PLY and statistics outputs
  - `m3c2_core/` – core algorithm execution and parameter estimation
  - `pipeline/` – orchestrators for batch and single-cloud processing
  - `statistics/` – computation and aggregation of metrics
- `report_pipeline/` – lightweight CLI for generating comparison reports
- `docu/` – project documentation and UML diagrams
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

| Parameter                 | Type           | Default / Example | Options                                         | Description |
|---------------------------|----------------|------------------|-------------------------------------------------|-------------|
| `data_dir`                | string         | `"data"`         | —                                               | Input directory containing point cloud folders. |
| `folders`                 | list\[string]  | `["pointclouds"]`| —                                               | List of folders containing point clouds. |
| `filename_reference`      | string         | `""`             | —                                               | Reference point cloud file for distance calculation (not used for single cloud statistics). |
| `filename_comparison`     | string         | `""`             | —                                               | Comparison point cloud file for distance calculation (not used for single cloud statistics). |
| `filename_singlecloud`    | string         | `""`             | —                                               | Single point cloud filename used for single-cloud statistics. |
| `use_subsampled_corepoints` | int          | `1`              | ≥1                                              | Use subsampled core points for distance computation. Eg. 3: Every 3rd point is used for the subsample. |
| `sample_size`             | int            | `10000`          | ≥1                                              | Number of core points used for parameter estimation (normal & projection scale). Not used to subsample distances. |
| `scale_strategy`          | string         | `"radius"`       | `radius`                                        | Parameter estimation strategy. `radius`: derive scales from neighborhood radius. |
| `only_stats`              | boolean        | `true`           | `true`, `false`                                 | Whether to skip M3C2 distances. `true`: compute only statistics; `false`: run full distance analysis. |
| `stats_singleordistance`  | string         | `"single"`       | `single`, `distance`, `plot`                    | Statistics mode. `single`: metrics for a single cloud; `distance`: pairwise distance analysis; `plot`: create plots from existing distance files. |
| `output_format`           | string         | `"excel"`        | `excel`, `json`                                 | Result export format. `excel`: write `.xlsx`; `json`: write JSON file. |
| `project`                 | string         | `"PROJECT"`      | —                                               | Project name used as prefix for folders and output files. |
| `normal_override`         | float / null   | `null`           | —                                               | Manual override for normal scale instead of automatic estimation. |
| `proj_override`           | float / null   | `null`           | —                                               | Manual override for projection scale instead of automatic estimation. |
| `use_existing_params`     | boolean        | `false`          | `true`, `false`                                 | Whether to reuse parameter file. `true`: load existing parameters; `false`: re-estimate parameters. |
| `outlier_detection_method`| string         | `"rmse"`         | `rmse`, `mad`, `nmad`, `std`                    | Outlier removal strategy. `rmse`: threshold by root mean square error; `mad`: median absolute deviation; `nmad`: normalized MAD; `std`: standard deviation. |
| `outlier_multiplicator`   | float          | `3.0`            | —                                               | Multiplicator applied by the chosen outlier method. |
| `log_level`               | string         | `"INFO"`         | `DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL` | Overrides logging verbosity. `DEBUG`: detailed diagnostics; `INFO`: general messages; `WARNING`: potential issues; `ERROR`: errors; `CRITICAL`: fatal errors. |
