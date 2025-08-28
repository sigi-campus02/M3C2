import os, re
from orchestration.batch_orchestrator import BatchOrchestrator
from config.pipeline_config import PipelineConfig
from log_utils.logging_utils import setup_logging

# --- deine bestehenden Settings ---
base_data_dir = "data"
folder_ids = ["Multi-Illumination"]  # enthält die Files 1-1_cloud, 1-1-AI_cloud, ...
mov_as_corepoints = True
use_subsampled_corepoints = 1
sample_size = 10000
only_stats = False
stats_singleordistance = "distance"
output_format = "excel"
project = "MARS_Multi_Illumination"
normal_override = 0.002
proj_override = 0.004
use_existing_params = False
outlier_detection_method = "rmse"
outlier_multiplicator = 3
# ----------------------------------

def parse_files(dirpath: str):
    """Liest Dateinamen und baut Dicts: {idx: name} für 1-* / 2-* jeweils plain & AI."""
    re_pat = re.compile(r'^(?Pgrp>[12])-(?Pidx>\d+)(?Pai>-AI)?_cloud$')
    one_plain, one_ai, two_plain, two_ai = {}, {}, {}, {}
    for fn in os.listdir(dirpath):
        m = re_pat.match(fn)
        if not m: 
            continue
        grp = m.group('grp')
        idx = int(m.group('idx'))
        ai  = bool(m.group('ai'))
        if grp == '1':
            (one_ai if ai else one_plain)[idx] = fn
        else:
            (two_ai if ai else two_plain)[idx] = fn
    return one_plain, one_ai, two_plain, two_ai

def main() -> None:
    cfgs = []

    for fid in folder_ids:
        folder = os.path.join(base_data_dir, fid)
        one_plain, one_ai, two_plain, two_ai = parse_files(folder)

        # 1) 1-i_cloud  vs 2-i_cloud
        for i in sorted(set(one_plain) & set(two_plain)):
            mov_name = one_plain[i]
            ref_name = two_plain[i]
            cfgs.append(PipelineConfig(
                folder, mov_name, ref_name,
                mov_as_corepoints, use_subsampled_corepoints,
                only_stats, stats_singleordistance,
                project, normal_override, proj_override,
                use_existing_params, outlier_multiplicator, outlier_detection_method
            ))

        # 2) 1-i_cloud  vs 2-i-AI_cloud
        for i in sorted(set(one_plain) & set(two_ai)):
            mov_name = one_plain[i]
            ref_name = two_ai[i]
            cfgs.append(PipelineConfig(
                folder, mov_name, ref_name,
                mov_as_corepoints, use_subsampled_corepoints,
                only_stats, stats_singleordistance,
                project, normal_override, proj_override,
                use_existing_params, outlier_multiplicator, outlier_detection_method
            ))

        # 3) 1-i-AI_cloud vs 2-i_cloud
        for i in sorted(set(one_ai) & set(two_plain)):
            mov_name = one_ai[i]
            ref_name = two_plain[i]
            cfgs.append(PipelineConfig(
                folder, mov_name, ref_name,
                mov_as_corepoints, use_subsampled_corepoints,
                only_stats, stats_singleordistance,
                project, normal_override, proj_override,
                use_existing_params, outlier_multiplicator, outlier_detection_method
            ))

    orchestrator = BatchOrchestrator(cfgs, sample_size, output_format)
    orchestrator.run_all()

if __name__ == "__main__":
    setup_logging()
    main()
