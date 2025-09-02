import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from m3c2.cli import outlier_plots


def test_main_creates_plot(tmp_path, monkeypatch):
    base = tmp_path / "0342-0349"
    base.mkdir()

    (base / "python_ref_m3c2_distances.txt").write_text("0.1\n0.2\n")
    (base / "python_ref_m3c2_distances_coordinates_inlier_std.txt").write_text(
        "i v\n0 0.1\n1 0.2\n"
    )

    out_dir = tmp_path / "out"

    def fake_savefig(path):
        print("savefig called with:", path)
        Path(path).write_text("img")

    monkeypatch.setattr(outlier_plots.plt, "savefig", fake_savefig)

    outlier_plots.main(
        base=str(base),
        variants=[("ref", "python_ref_m3c2_distances.txt")],
        inlier_suffixes=[("std", "Inlier _STD")],
        outdir=str(out_dir),
    )

    assert (out_dir / "0342-0349_OutlierComparison_ref.png").exists()

