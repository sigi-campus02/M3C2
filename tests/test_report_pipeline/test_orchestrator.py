from pathlib import Path
from unittest.mock import MagicMock

from report_pipeline.orchestrator import ReportOrchestrator


class Job:
    def __init__(self, items, title):
        self.items = items
        self.page_title = title


def test_orchestrator_flattens_figures():
    plotter = MagicMock()
    fig1 = MagicMock()
    fig2 = MagicMock()
    fig3 = MagicMock()
    plotter.make_overlay.side_effect = [[fig1, fig2], [fig3]]

    pdf_writer = MagicMock()
    pdf_writer.write.return_value = Path("out.pdf")

    jobs = [Job([1], "p1"), Job([2], "p2")]
    orchestrator = ReportOrchestrator(plotter, pdf_writer)
    result = orchestrator.run(jobs)

    pdf_writer.write.assert_called_once_with([fig1, fig2, fig3])
    assert result == Path("out.pdf")
