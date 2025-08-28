import os, re, logging
from orchestration.batch_orchestrator import BatchOrchestrator
from config.pipeline_config import PipelineConfig
from log_utils.logging_utils import setup_logging

# --- Settings ---
base_data_dir = "data"
folder_ids = ["Multi-Illumination"]
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
# ----------------

log = logging.getLogger(__name__)

def parse_files(dirpath: str):
    """Liest Dateinamen und baut Dicts: {idx: name_stem} für 1-* / 2-* jeweils plain & AI."""
    import os, re

    # erlaubt optional eine Dateiendung und matched auf dem stem
    re_pat = re.compile(r'^(?P<grp>[12])-(?P<idx>\d+)(?P<ai>-AI)?_cloud$')

    one_plain, one_ai, two_plain, two_ai = {}, {}, {}, {}

    with os.scandir(dirpath) as it:
        for entry in it:
            if not entry.is_file():
                continue
            stem, ext = os.path.splitext(entry.name)  # z.B. ("1-1_cloud", ".ply")
            m = re_pat.match(stem)
            if not m:
                continue
            grp = m.group('grp')
            idx = int(m.group('idx'))
            ai  = bool(m.group('ai'))
            # wir speichern den STEM (ohne Extension). Das passt zu deiner Pipeline.
            if grp == '1':
                (one_ai if ai else one_plain)[idx] = stem
            else:
                (two_ai if ai else two_plain)[idx] = stem

    return one_plain, one_ai, two_plain, two_ai


def main() -> None:
    cfgs = []

    for fid in folder_ids:
        folder = os.path.join(base_data_dir, fid)
        one_plain, one_ai, two_plain, two_ai = parse_files(folder)

        # --- Übersicht / Logging ---
        log.info("[Scan] %s", folder)
        log.info("  1-*  plain: %d | AI: %d | indices=%s / %s",
                 len(one_plain), len(one_ai),
                 sorted(one_plain.keys()), sorted(one_ai.keys()))
        log.info("  2-*  plain: %d | AI: %d | indices=%s / %s",
                 len(two_plain), len(two_ai),
                 sorted(two_plain.keys()), sorted(two_ai.keys()))

        # Case 1: 1-i_cloud vs 2-i_cloud
        case1_idx = sorted(set(one_plain) & set(two_plain))
        log.info("  Case1 (1-i vs 2-i): %d Runs, indices=%s", len(case1_idx), case1_idx)
        for i in case1_idx:
            cfgs.append(PipelineConfig(
                folder, one_plain[i], two_plain[i],
                mov_as_corepoints, use_subsampled_corepoints,
                only_stats, stats_singleordistance,
                project, normal_override, proj_override,
                use_existing_params, outlier_multiplicator, outlier_detection_method
            ))

        # Case 2: 1-i_cloud vs 2-i-AI_cloud
        case2_idx = sorted(set(one_plain) & set(two_ai))
        log.info("  Case2 (1-i vs 2-i-AI): %d Runs, indices=%s", len(case2_idx), case2_idx)
        for i in case2_idx:
            cfgs.append(PipelineConfig(
                folder, one_plain[i], two_ai[i],
                mov_as_corepoints, use_subsampled_corepoints,
                only_stats, stats_singleordistance,
                project, normal_override, proj_override,
                use_existing_params, outlier_multiplicator, outlier_detection_method
            ))

        # Case 3: 1-i-AI_cloud vs 2-i_cloud
        case3_idx = sorted(set(one_ai) & set(two_plain))
        log.info("  Case3 (1-i-AI vs 2-i): %d Runs, indices=%s", len(case3_idx), case3_idx)
        for i in case3_idx:
            cfgs.append(PipelineConfig(
                folder, one_ai[i], two_plain[i],
                mov_as_corepoints, use_subsampled_corepoints,
                only_stats, stats_singleordistance,
                project, normal_override, proj_override,
                use_existing_params, outlier_multiplicator, outlier_detection_method
            ))

        log.info("  => Total Runs in %s: %d", fid, len(cfgs))

    orchestrator = BatchOrchestrator(cfgs, sample_size, output_format)
    orchestrator.run_all()

if __name__ == "__main__":
    setup_logging()
    main()
