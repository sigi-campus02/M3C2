from m3c2.pipeline.batch_orchestrator import BatchOrchestrator
from m3c2.config.pipeline_config import PipelineConfig
import os
from m3c2.io.logging_utils import setup_logging

# folder_ids = ["0342-0349", "0817-0821", "0910-0913", "1130-1133", "1203-1206", "1306-1311"]

# folders in folder "data" to be iterated
folder_ids = ["0342-0349"]

# names of reference cloud files to be compared
ref_variants = ["ref", "ref_ai"]

# name of moving point cloud file
filename_mov = "mov"

# TRUE: use mov point cloud as corepoints
# FALSE: use ref point cloud as corepoints
mov_as_corepoints = True

# run M3C2 distance algorithm on subsampled corepoints
# 1 = no subsampling; corepoints = complete mov
# e.g. 5 = every 5. point
use_subsampled_corepoints = 1

# sample size used for parameter estimation (normal & projection scale)
sample_size = 10000

# TRUE: only statistics are computed based on distance file in folder (no processing of M3C2)
# FALSE: Runs M3C2 pipeline
only_stats = False

# "single": Only single-cloud statistics
# "distance": Distance-based statistics on M3C2 output
stats_singleordistance = "distance"

# "excel": appends data to excel file
# "json": appends data to json file
output_format = "excel"

# name of project used for file names & folder names
project = "MARS"

# specify overrides for M3C2 parameters
normal_override = None               # Normal Scale Override
proj_override = None                 # Projection Scale Override

# TRUE: use existing parameters (in folder) if available
# FALSE: compute parameters with param_estimator
use_existing_params = False

# specify outlier removal parameter
# default = 3 (3 * RMSE = Outlier Threshold)
outlier_rmse_multiplicator = 3


def main() -> None:
    cfgs = []
    for fid in folder_ids:
        folder = os.path.join("data", fid)
        for filename_ref in ref_variants:
            cfgs.append(
                PipelineConfig(
                    folder,
                    filename_mov,
                    filename_ref,
                    mov_as_corepoints,
                    use_subsampled_corepoints,
                    only_stats,
                    stats_singleordistance,
                    project,
                    normal_override,
                    proj_override,
                    use_existing_params,
                    outlier_rmse_multiplicator
                )
            )

    orchestrator = BatchOrchestrator(cfgs, sample_size, output_format)
    orchestrator.run_all()


if __name__ == "__main__":
    setup_logging()
    main()
