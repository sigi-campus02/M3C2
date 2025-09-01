import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from m3c2.cli import singlecloud_stats


def test_main_writes_output(tmp_path, monkeypatch):
    out_file = tmp_path / "stats.xlsx"

    def fake_calc(cls, **kwargs):
        Path(kwargs["out_path"]).write_text("stats")

    monkeypatch.setattr(
        singlecloud_stats.StatisticsService,
        "calc_single_cloud_stats",
        classmethod(fake_calc),
    )

    singlecloud_stats.main(folder_ids=["fid"], out_path=str(out_file))

    assert out_file.exists()

