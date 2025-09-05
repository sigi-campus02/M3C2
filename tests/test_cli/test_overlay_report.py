from pathlib import Path

from m3c2.cli import overlay_report


def test_main_builds_pdf(tmp_path, monkeypatch):
    a = tmp_path / "a.txt"
    b = tmp_path / "b.txt"
    a.write_text("1\n2\n3\n")
    b.write_text("4\n5\n6\n")

    def fake_generate_overlay_plots(data, outdir):
        names = [
            "OverlayHistogramm",
            "OverlayGaussFits",
            "OverlayWeibullFits",
            "Boxplot",
            "QQPlot",
        ]
        paths = []
        for name in names:
            p = Path(outdir) / f"Part_0_WITH_{name}.png"
            p.write_text("img")
            paths.append(str(p))
        return paths

    def fake_build_parts_pdf(outdir, pdf_path=None, include_with=True, include_inlier=False):
        p = Path(pdf_path)
        p.write_text("pdf")
        return str(p)

    monkeypatch.setattr(overlay_report, "generate_overlay_plots", fake_generate_overlay_plots)
    monkeypatch.setattr(
        overlay_report.PlotService,
        "build_parts_pdf",
        staticmethod(fake_build_parts_pdf),
    )

    pdf = overlay_report.main([str(a), str(b)], outdir=str(tmp_path))
    assert Path(pdf).exists()
