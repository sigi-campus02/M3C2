# M3C2 Point Cloud Processing Pipeline

This project provides a Python workflow for comparing 3D point cloud pairs using the Multiscale Model to Model Cloud Comparison (M3C2) algorithm. The pipeline automates parameter estimation, distance computation, statistical analysis, and visualization.

## Features
- Load point clouds from multiple formats (.xyz, .las/.laz, .ply, .obj, .gpc)
- Estimate optimal normal and search radii
- Execute the M3C2 algorithm via [py4dgeo](https://github.com/py4dgeo/py4dgeo)
- Detect and exclude outliers with configurable strategies
- Compute rich statistical metrics and export results to Excel or JSON
- Generate plots and visualizations for distance distributions
- Orchestrate batch processing across many point cloud pairs

## Installation
```bash
pip install -r requirements.txt
```

## Usage
Adjust the settings in `main.py` and run:

```bash
python main.py
```

Outputs, statistics, and logs are written to dataset folders and the `outputs/` directory.

## Logging

Control verbosity with the `--log_level` option or by setting the `LOG_LEVEL` environment variable. For example:

```bash
export LOG_LEVEL=DEBUG
python main.py
```

If neither is provided, the log level defaults to `INFO`.

## Repository Structure
- `config/` – dataclasses and plotting configuration
- `datasource/` – unified loading of point cloud data
- `generation/` – utilities for producing derived clouds
- `orchestration/` – batch runners and M3C2 execution logic
- `services/` – parameter estimation, statistics, outlier handling, visualization
- `tests/` – unit tests covering key functionality

For detailed descriptions of the calculated statistics, see `docu/README.md`.
