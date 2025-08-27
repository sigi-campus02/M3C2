from orchestration.batch_orchestrator import BatchOrchestrator
from config.pipeline_config import PipelineConfig
import os
from log_utils.logging_utils import setup_logging

# folder_ids = ["0342-0349", "0817-0821", "0910-0913", "1130-1133", "1203-1206", "1306-1311"]

# folders in folder "data" to be iterated
folder_ids = ["1-1", "1-2", "1-3", "1-4", "1-5", "1-6", "1-7"]

# names of reference cloud files to be compared
ref_variants = ["Job_0378_8400-110-rad-1-1-AI_cloud"]

# name of moving point cloud file
filename_mov = "Job_0378_8400-110-rad-1-1_cloud"

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
project = "MARS_Multi_Illumination"

# specify overrides for M3C2 parameters
normal_override = 0.002               # Normal Scale Override
proj_override = 0.004                 # Projection Scale Override

# TRUE: use existing parameters (in folder) if available
# FALSE: compute parameters with param_estimator
use_existing_params = False

#-------------------------------------------
# specify outlier removal parameter 

# rmse = np.sqrt(np.mean(distances_valid ** 2)) 
        # outlier_mask = np.abs(distances_valid) > (outlier_multiplicator * rmse)
# iqr = q3 - q1; 
        # lower_bound = q1 - 1.5 * iqr; 
        # upper_bound = q3 + 1.5 * iqr
        # outlier_mask = (distances_valid < lower_bound) | (distances_valid > upper_bound)
# std = np.std(distances_valid)
        # mu = np.mean(distances_valid)
        # std = np.std(distances_valid)
        # outlier_mask = np.abs(distances_valid - mu) > (outlier_multiplicator * std)
# nmad = 1.4826 * np.median(np.abs(distances_valid - med))
        # med  = np.median(distances_valid)
        # outlier_mask = np.abs(distances_valid - med) > (outlier_multiplicator * nmad)

# Default = RMSE
outlier_detection_method = "std"  # Options: "rmse", "iqr", "std", "nmad"

# Multiplikator used for methods rmse, std, nmad
# default = 3 (e.g. 3 * RMSE = Outlier Threshold)
outlier_multiplicator = 3

#-------------------------------------------

def main() -> None:
    cfgs = []
    for fid in folder_ids:
        folder = os.path.join("data", "Multi-illumination", "Job_0378_8400-110", fid) # angepasst wenn es Unterordner in Data gibt ansonsten "data"
        for filename_ref in ref_variants:
            filename_ref = f"Job_0378_8400-110-rad-{fid}-AI_cloud"
            filename_mov = f"Job_0378_8400-110-rad-{fid}_cloud"
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
                    outlier_multiplicator,
                    outlier_detection_method
                )
            )

    orchestrator = BatchOrchestrator(cfgs, sample_size, output_format)
    orchestrator.run_all()


if __name__ == "__main__":
    setup_logging()
    main()
