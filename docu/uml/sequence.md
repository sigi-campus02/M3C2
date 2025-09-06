@startuml
actor User

User -> main : main()
main -> BatchOrchestrator : run_all()

loop für jede PipelineConfig
  BatchOrchestrator -> DataSource : load_points()
  DataSource --> BatchOrchestrator : comparison, reference, corepoints

  alt process_python_CC == "python" && !only_stats
    BatchOrchestrator -> ParamEstimator : estimate_min_spacing()
    BatchOrchestrator -> ParamEstimator : scan_scales()
    BatchOrchestrator -> ParamEstimator : select_scales()
    ParamEstimator --> BatchOrchestrator : normal, projection

    BatchOrchestrator -> M3C2Runner : run(comparison, reference, corepoints, normal, projection)
    M3C2Runner --> BatchOrchestrator : distances, uncertainties

  end

  BatchOrchestrator -> StatisticsService : compute_m3c2_statistics()/calc_single_cloud_stats()
end
User -> report_pipeline : generate_report()
@enduml
