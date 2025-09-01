import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from m3c2.cli import plot_report


def test_main_builds_pdfs(tmp_path, monkeypatch):
    def fake_build_parts_pdf(outdir, pdf_path=None, include_with=True, include_inlier=True):
        Path(pdf_path).write_text("pdf")
        return pdf_path

    monkeypatch.setattr(
        plot_report.PlotService,
        "build_parts_pdf",
        staticmethod(fake_build_parts_pdf),
    )
    monkeypatch.setattr(plot_report, "setup_logging", lambda **_: None)

    pdf_incl, pdf_excl = plot_report.main(data_dir=str(tmp_path), out_dir=str(tmp_path))

    assert Path(pdf_incl).exists()
    assert Path(pdf_excl).exists()

