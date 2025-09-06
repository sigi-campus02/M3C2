import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

from report_pipeline.pdf import writer


def test_write_creates_pdf():
    fig, ax = plt.subplots()
    ax.plot([1, 2], [3, 4])
    fig.suptitle('Demo')
    pdf_path = writer.write([fig])
    plt.close(fig)
    assert pdf_path.exists()
    assert pdf_path.suffix == '.pdf'
    assert pdf_path.read_bytes().startswith(b'%PDF')


def test_write_to_explicit_path(tmp_path):
    fig, ax = plt.subplots()
    ax.plot([1, 2], [3, 4])
    target = tmp_path / "report.pdf"
    pdf_path = writer.write([fig], out=target)
    plt.close(fig)
    assert pdf_path == target
    assert pdf_path.exists()
