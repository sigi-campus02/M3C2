from batch_orchestrator import BatchOrchestrator
from pipeline_config import PipelineConfig
import os
from logging_utils import setup_logging

# "points_40", "points_100", "points_80", "points_overlap2", "points_zshift"

# Arguments for the pipeline
folder = os.path.join("data", "rocks") # Folder containing the point clouds
filename_mov = "points_100" # Moving point cloud filename
filename_ref = "points_zshift" # Reference point cloud filename / Name for single cloud statistics
mov_as_corepoints = True # Use the moving point cloud as corepoints
use_subsampled_corepoints = 1 # Use subsampled corepoints (1 for no subsampling, 1000 for 1000 points subsampling)
strategy = "radius" # Strategy for the M3C2 algorithm, can be "radius" or "voxel"
sample_size = 10000 # Sample size for parameter estimation, NOT USED FOR ALGORITHM
process_python_CC = "python" # Use "python" implementation of M3C2, "CC" to analyse CloudCompare output
only_stats = True # Only compute statistics on existing files, no visualization or running the algorithm
stats_singleordistance = "single" # Compute statistics for "single" point cloud or for "distance" output of M3C2


def main() -> None:
    cfgs = [
        PipelineConfig(
            folder,
            filename_mov,
            filename_ref,
            mov_as_corepoints,
            use_subsampled_corepoints,
            process_python_CC,
            only_stats,
            stats_singleordistance
        ),
    ]
    orchestrator = BatchOrchestrator(cfgs, strategy, sample_size)
    orchestrator.run_all()

if __name__ == "__main__":
    setup_logging()
    main()
