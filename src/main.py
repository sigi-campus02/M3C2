import numpy as np

from batch_orchestrator import BatchOrchestrator
from pipeline_config import PipelineConfig
from pathlib import Path
import py4dgeo
from datasource import DataSource
import os
import matplotlib.pyplot as plt

# class PipelineConfig:
#     folder_id: str
#     filename_mov: str
#     filename_ref: str
#     normal_override: Optional[float] = None
#     proj_override: Optional[float] = None

# "points_40", "points_100", "points_80", "points_overlap2", "points_zshift"

folder = os.path.join("data", "rocks")
filename_mov = "points_100"
filename_ref = "points_zshift"
mov_as_corepoints = True
use_subsampled_corepoints = 1 # 1 for no subsampling, 1000 for 1000 points subsampling
strategy = "radius"
sample_size = 10000 # for parameter estimation, not used in radius strategy
process_python_CC = "python" # alternative CC for CloudCompare Distance Files

cfgs = [
    PipelineConfig(folder, filename_mov, filename_ref, mov_as_corepoints, use_subsampled_corepoints, process_python_CC),
]

orchestrator = BatchOrchestrator(cfgs, strategy, sample_size)
orchestrator.run_all()
