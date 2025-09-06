from pathlib import Path
from unittest.mock import MagicMock

from report_pipeline.domain import DistanceFile
from report_pipeline.plotting import figure_factory

def test_make_overlay_splits_by_max_per_page(monkeypatch):
    items = [DistanceFile(path=Path(f"{i}.txt"), label=str(i)) for i in range(7)]
    monkeypatch.setattr(figure_factory, "load_distance_series", lambda p: [1, 2, 3])

    axes: list[MagicMock] = []

    def fake_subplots():
        fig = MagicMock()
        ax = MagicMock()
        axes.append(ax)
        return fig, ax

    monkeypatch.setattr(figure_factory.plt, "subplots", fake_subplots)

    figs = figure_factory.make_overlay(items, max_per_page=4)
    assert len(figs) == 2
    assert len(axes[0].hist.call_args_list) == 4
    assert len(axes[1].hist.call_args_list) == 3


def test_make_overlay_legend_visibility(monkeypatch):
    items = [DistanceFile(path=Path("a.txt"), label="a")]
    monkeypatch.setattr(figure_factory, "load_distance_series", lambda p: [1, 2, 3])

    fig = MagicMock()
    ax = MagicMock()
    monkeypatch.setattr(figure_factory.plt, "subplots", lambda: (fig, ax))

    figure_factory.make_overlay(items, show_legend=False)
    ax.legend.assert_not_called()

    ax.legend.reset_mock()
    figure_factory.make_overlay(items, show_legend=True)
    ax.legend.assert_called_once_with()
