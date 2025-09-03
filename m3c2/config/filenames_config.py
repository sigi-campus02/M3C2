from __future__ import annotations

from dataclasses import dataclass

from m3c2.config.pipeline_config import PipelineConfig


@dataclass(frozen=True)
class FileNameParams:
    """Resolved filename components derived from a :class:`PipelineConfig`."""

    prefix: str
    tag: str
    fid: str
    method: str

    @classmethod
    def from_config(cls, cfg: PipelineConfig) -> "FileNameParams":
        if cfg.stats_singleordistance == "single":
            tag = cfg.filename_singlecloud
        else:
            tag = f"{cfg.filename_mov}-{cfg.filename_ref}"
        return cls(
            prefix=cfg.project,
            tag=tag,
            fid=cfg.folder_id,
            method=cfg.outlier_detection_method,
        )

    def distances(self) -> str:
        return FileNames.distances(self.prefix, self.tag)

    def distances_coordinates(self) -> str:
        return FileNames.distances_coordinates(self.prefix, self.tag)

    def distances_coordinates_inlier(self) -> str:
        return FileNames.distances_coordinates_inlier(
            self.prefix, self.tag, self.method
        )

    def uncertainties(self) -> str:
        return FileNames.uncertainties(self.prefix, self.tag)

    def params(self) -> str:
        return FileNames.params(self.prefix, self.tag)

    def ply(self) -> str:
        return FileNames.ply(self.prefix, self.tag)

    def ply_include_nonvalid(self) -> str:
        return FileNames.ply_include_nonvalid(self.prefix, self.tag)

    def stats_distances(self, ext: str) -> str:
        return FileNames.stats_distances(self.prefix, ext)

    def stats_clouds(self, ext: str) -> str:
        return FileNames.stats_clouds(self.prefix, ext)

    def report_cloud_moved_distances(self) -> str:
        return FileNames.report_cloud_moved_distances(self.prefix, self.fid)

    def report_cloud_moved_distances_inlier(self) -> str:
        return FileNames.report_cloud_moved_distances_inlier(
            self.prefix, self.fid, self.method
        )


class FileNames:
    """Central repository for common filename patterns used in the project."""

    # Generic configuration file
    CONFIG = "config.json"

    # Core M3C2 processing outputs
    DISTANCES = "{prefix}_{tag}_m3c2_distances.txt"
    DISTANCES_COORDS = "{prefix}_{tag}_m3c2_distances_coordinates.txt"
    DISTANCES_COORDS_INLIER = (
        "{prefix}_{tag}_m3c2_distances_coordinates_inlier_{method}.txt"
    )
    DISTANCES_PATTERN = r"_m3c2_distances\.txt"
    DISTANCES_COORDS_INLIER_PATTERN = (
        r"_m3c2_distances_coordinates_inlier_(?P<meth>[a-zA-Z0-9_]+)\.txt"
    )
    UNCERTAINTIES = "{prefix}_{tag}_m3c2_uncertainties.txt"
    PARAMS = "{prefix}_{tag}_m3c2_params.txt"
    PLY = "{prefix}_{tag}.ply"
    PLY_INCLUDE_NONVALID = "{prefix}_{tag}_includenonvalid.ply"

    # Statistics outputs
    STATS_DISTANCES_JSON = (
        "outputs/{project}_output/{project}_m3c2_stats_distances.json"
    )
    STATS_DISTANCES_XLSX = (
        "outputs/{project}_output/{project}_m3c2_stats_distances.xlsx"
    )
    STATS_CLOUDS_JSON = (
        "outputs/{project}_output/{project}_m3c2_stats_clouds.json"
    )
    STATS_CLOUDS_XLSX = (
        "outputs/{project}_output/{project}_m3c2_stats_clouds.xlsx"
    )
    STATS_ALL_XLSX = "m3c2_stats_all.xlsx"
    STATS_CLOUDS_XLSX_SIMPLE = "m3c2_stats_clouds.xlsx"
    CLOUD_STATS_XLSX = "cloud_stats.xlsx"

    # Report service patterns
    REPORT_CLOUD_MOVED_DISTANCES = (
        "{prefix}_Job_0378_8400-110-rad-{fid}_cloud_moved_m3c2_distances.txt"
    )
    REPORT_CLOUD_MOVED_DISTANCES_INLIER = (
        "{prefix}_Job_0378_8400-110-rad-{fid}_cloud_moved_m3c2_distances_coordinates_inlier_{method}.txt"
    )

    @staticmethod
    def from_config(cfg: PipelineConfig) -> FileNameParams:
        return FileNameParams.from_config(cfg)

    @staticmethod
    def config() -> str:
        return FileNames.CONFIG

    @staticmethod
    def distances(prefix: str, tag: str) -> str:
        return FileNames.DISTANCES.format(prefix=prefix, tag=tag)

    @staticmethod
    def distances_coordinates(prefix: str, tag: str) -> str:
        return FileNames.DISTANCES_COORDS.format(prefix=prefix, tag=tag)

    @staticmethod
    def distances_coordinates_inlier(prefix: str, tag: str, method: str) -> str:
        return FileNames.DISTANCES_COORDS_INLIER.format(
            prefix=prefix, tag=tag, method=method
        )

    @staticmethod
    def uncertainties(prefix: str, tag: str) -> str:
        return FileNames.UNCERTAINTIES.format(prefix=prefix, tag=tag)

    @staticmethod
    def params(prefix: str, tag: str) -> str:
        return FileNames.PARAMS.format(prefix=prefix, tag=tag)

    @staticmethod
    def ply(prefix: str, tag: str) -> str:
        return FileNames.PLY.format(prefix=prefix, tag=tag)

    @staticmethod
    def ply_include_nonvalid(prefix: str, tag: str) -> str:
        return FileNames.PLY_INCLUDE_NONVALID.format(prefix=prefix, tag=tag)

    @staticmethod
    def stats_distances(project: str, ext: str) -> str:
        if ext == "json":
            return FileNames.STATS_DISTANCES_JSON.format(project=project)
        if ext == "xlsx":
            return FileNames.STATS_DISTANCES_XLSX.format(project=project)
        raise ValueError("Unsupported extension: {ext}")

    @staticmethod
    def stats_clouds(project: str, ext: str) -> str:
        if ext == "json":
            return FileNames.STATS_CLOUDS_JSON.format(project=project)
        if ext == "xlsx":
            return FileNames.STATS_CLOUDS_XLSX.format(project=project)
        raise ValueError("Unsupported extension: {ext}")

    @staticmethod
    def stats_all_xlsx() -> str:
        return FileNames.STATS_ALL_XLSX

    @staticmethod
    def stats_clouds_xlsx() -> str:
        return FileNames.STATS_CLOUDS_XLSX_SIMPLE

    @staticmethod
    def cloud_stats_xlsx() -> str:
        return FileNames.CLOUD_STATS_XLSX

    @staticmethod
    def report_cloud_moved_distances(prefix: str, fid: str) -> str:
        return FileNames.REPORT_CLOUD_MOVED_DISTANCES.format(
            prefix=prefix, fid=fid
        )

    @staticmethod
    def report_cloud_moved_distances_inlier(
        prefix: str, fid: str, method: str
    ) -> str:
        return FileNames.REPORT_CLOUD_MOVED_DISTANCES_INLIER.format(
            prefix=prefix, fid=fid, method=method
        )