from batch_orchestrator import BatchOrchestrator
from pipeline_config import PipelineConfig
import os
import argparse
from logging_utils import setup_logging

# Variationen
folder_ids = ["0342-0349", "0817-0821", "0910-0913", "1130-1133", "1203-1206", "1306-1311"]
ref_variants = ["ref", "ref_ai"]

# Fix-Parameter
filename_mov = "mov"                 # Moving point cloud
mov_as_corepoints = True
use_subsampled_corepoints = 1        # 1 = kein Subsampling
strategy = "radius"                  # "radius" oder "voxel"
sample_size = 10000                  # nur für Parameterschätzung, nicht für Algorithmus
process_python_CC = "CC"         # "python" oder "CC"
only_stats = False                   # nur Stats berechnen (True) oder Pipeline laufen lassen (False)
stats_singleordistance = "distance"  # "single" oder "distance"

def main(stats_format: str) -> None:
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
                    process_python_CC,
                    only_stats,
                    stats_singleordistance,
                )
            )

    orchestrator = BatchOrchestrator(cfgs, strategy, sample_size, output_format=stats_format)
    orchestrator.run_all()


if __name__ == "__main__":
    setup_logging()
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stats-format",
        choices=["excel", "json"],
        default="excel",
        help="Format für Statistik-Exports",
    )
    args = parser.parse_args()
    main(args.stats_format)
