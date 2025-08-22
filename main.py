from batch_orchestrator import BatchOrchestrator
from pipeline_config import PipelineConfig
import os
from logging_utils import setup_logging

# Variationen

# TUNSPEKT FOLDERS: "TUNSPEKT_Altone(mov)-Faro(ref)", "TUNSPEKT_Handheld(mov)-Faro(ref)", "TUNSPEKT_Mavic(mov)-Faro(ref)"
# MARS FOLDERS:     "0342-0349", "0817-0821", "0910-0913", "1130-1133", "1203-1206", "1306-1311"
# MARS REF VARIANTS: "ref", "ref_ai"


folder_ids = ["0342-0349"]
ref_variants = ["ref"]

# Fix-Parameter
filename_mov = "mov"                 # Moving point cloud
mov_as_corepoints = True
use_subsampled_corepoints = 1        # 1 = kein Subsampling
strategy = "radius"                  # "radius" oder "voxel"
sample_size = 10000                  # nur für Parameterschätzung, nicht für Algorithmus
process_python_CC = "CC"         # "python" oder "CC"
only_stats = True                   # nur Stats berechnen (True) oder Pipeline laufen lassen (False)
stats_singleordistance = "distance"  # "single" oder "distance"
output_format = "excel"              # "excel" oder "json"
project = "MARS"                     # "TUNSPEKT" "MARS"
normal_override = None               # Normal Scale Override
proj_override = None                 # Projection Scale Override
use_existing_params = True           # ob vorhandene Parameter (in Ordner) genutzt werden (True) oder neu berechnet (False)       

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
                    process_python_CC,
                    only_stats,
                    stats_singleordistance,
                    project,
                    normal_override,
                    proj_override,
                    use_existing_params
                )
            )

    orchestrator = BatchOrchestrator(cfgs, strategy, sample_size, output_format)
    orchestrator.run_all()


if __name__ == "__main__":
    setup_logging()
    main()
