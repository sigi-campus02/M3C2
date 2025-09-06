@startuml
' =======================
'   Pipeline
' =======================
package "Pipeline" {
  class PipelineConfig {
    +folder_id: str
    +filename_comparison: str
    +filename_reference: str
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
 '   Reporting CLI
 ' =======================
 package "report_pipeline" {
   class ReportPipelineCLI {
     +folder()
     +multifolder()
     +files()
   }
 }

 ' =======================
 '   Beziehungen
 ' =======================
 BatchOrchestrator --> PipelineConfig
 BatchOrchestrator --> DataSource
 BatchOrchestrator --> ParamEstimator
 BatchOrchestrator --> M3C2Runner
 BatchOrchestrator --> StatisticsService
 ParamEstimator --> ScaleStrategy
 ScaleStrategy <|.. RadiusScanStrategy
ScaleStrategy <|.. VoxelScanStrategy
@enduml
