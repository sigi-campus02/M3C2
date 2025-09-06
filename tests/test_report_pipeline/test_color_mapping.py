from pathlib import Path
from unittest.mock import MagicMock

from report_pipeline.domain import DistanceFile
from report_pipeline.plotting import figure_factory


def test_group_color_strategy_assigns_same_color(monkeypatch):
    items = [
        DistanceFile(path=Path("a.txt"), label="a", group="g1"),
        DistanceFile(path=Path("b.txt"), label="b", group="g2"),
        DistanceFile(path=Path("c.txt"), label="c", group="g1"),
    ]
    monkeypatch.setattr(figure_factory, "load_distance_series", lambda p: [1, 2, 3])
    fig = MagicMock()
    ax = MagicMock()
    monkeypatch.setattr(figure_factory.plt, "subplots", lambda: (fig, ax))

    figure_factory.make_overlay(items, color_strategy="group")
    colors = [call.kwargs.get("color") for call in ax.hist.call_args_list]
    assert colors[0] == colors[2]
    assert colors[0] != colors[1]
