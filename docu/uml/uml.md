@startuml
' =======================
'   Pipeline
' =======================
package "Pipeline" {
  class PipelineConfig {
    +folder_id: str
    +filename_comparison: str
    +filename_reference: str
    +comparison_as_corepoints: bool
    +use_subsampled_corepoints: int
    +process_python_CC: str
    +only_stats: bool
    +stats_singleordistance: str
    normal_override: float?
    proj_override: float?
  }

  class BatchOrchestrator {
    +BatchOrchestrator()
    +run_all()
    -_resolve_strategy()
    -_run_single()
    -_load_data()
    -_determine_scales()
    -_save_params()
    -_run_m3c2()
    -_compute_statistics()
    -_generate_visuals()
  }
}

' =======================
'   Data & Parameters
' =======================
package "Data" {
  class DataSource {
    +DataSource()
    -_exists()
    -_detect()
    -_read_las_or_laz_to_xyz_array()
    -_ensure_xyz()
    +load_points()
  }

  class ParamEstimator {
    +estimate_min_spacing()
    +scan_scales()
    +select_scales()
  }

  interface ScaleStrategy {
    +scan()
  }

  class RadiusScanStrategy {
    +evaluate_radius_scale()
    +scan()
  }

  class VoxelScanStrategy {
    +evaluate_voxel_scale()
    +scan()
  }

  class ScaleScan {
    +scale: float
    +valid_normals: int
    +mean_population: float
    +roughness: float
    +coverage: float
    +mean_lambda3: float
    total_points: int?
    std_population: float?
    perc97_population: int?
    relative_roughness: float?
    total_voxels: int?
  }
}

' =======================
'   Core Algorithm
' =======================
package "M3C2" {
  class M3C2Runner {
    +run()
  }

  class StatisticsService {
    +calc_stats()
    +compute_m3c2_statistics()
    +calc_single_cloud_stats()
    +write_cloud_stats()
  }
}

' =======================
'   Visualization & Plots
' =======================
package "Visualization" {
  class VisualizationService {
    +histogram()
    +colorize()
    +export_valid()
  }

  class PlotOptions {
    +plot_hist: bool
    +plot_gauss: bool
    +plot_weibull: bool
    +plot_box: bool
    +plot_qq: bool
    +plot_grouped_bar: bool
  }

  class PlotConfig {
    +folder_id: str
    +filenames: List<str>
    +versions: List<str>
    +bins: int
    +colors: Dict<str,str>
    +outdir: str
    +ensure_colors()
  }

  class PlotService {
    +overlay_plots()
    +summary_pdf()
    -_resolve()
    -_load_data()
    -_get_common_range()
    -_plot_overlay_histogram()
    -_plot_overlay_gauss()
    -_plot_overlay_weibull()
    -_plot_overlay_boxplot()
    -_plot_overlay_qq()
    -_plot_grouped_bar_means_stds()
  }
}

' =======================
'   Beziehungen
' =======================
BatchOrchestrator --> PipelineConfig
BatchOrchestrator --> DataSource
BatchOrchestrator --> ParamEstimator
BatchOrchestrator --> M3C2Runner
BatchOrchestrator --> VisualizationService
BatchOrchestrator --> StatisticsService
ParamEstimator --> ScaleStrategy
ScaleStrategy <|.. RadiusScanStrategy
ScaleStrategy <|.. VoxelScanStrategy
@enduml
