from pathlib import Path
from unittest.mock import MagicMock

from report_pipeline.orchestrator import ReportOrchestrator


class Job:
    def __init__(self, items, title):
        self.items = items
        self.page_title = title


def test_orchestrator_flattens_figures_and_forwards_options():
    plotter = MagicMock()
    fig1 = MagicMock()
    fig2 = MagicMock()
    fig3 = MagicMock()
    plotter.make_overlay.side_effect = [[fig1, fig2], [fig3]]

    pdf_writer = MagicMock()
    pdf_writer.write.return_value = Path("final.pdf")

    jobs = [Job([1], None), Job([2], None)]
    builder = MagicMock()
    builder.build_jobs.return_value = jobs

    orchestrator = ReportOrchestrator(plotter, pdf_writer, builder)
    result = orchestrator.run(
        max_per_page=4,
        color_mapping="by_label",
        legend=True,
        title="doc",
        out=Path("final.pdf"),
    )

    builder.build_jobs.assert_called_once_with()
    assert plotter.make_overlay.call_count == 2
    plotter.make_overlay.assert_any_call(
        jobs[0].items,
        title="doc",
        max_per_page=4,
        color_mapping="by_label",
        legend=True,
    )
    pdf_writer.write.assert_called_once_with([fig1, fig2, fig3], Path("final.pdf"))
    assert result == Path("final.pdf")
