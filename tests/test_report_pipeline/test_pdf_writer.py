import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pytest
from report_pipeline.pdf import writer


def test_write_creates_pdf(tmp_path):
    fig, ax = plt.subplots()
    ax.plot([1, 2], [3, 4])
    out_path = tmp_path / "out.pdf"
    pdf_path = writer.write([fig], out_path, "Demo")
    plt.close(fig)
    assert pdf_path.exists()
    assert pdf_path.suffix == '.pdf'
    assert pdf_path.read_bytes().startswith(b'%PDF')


def test_write_empty_raises(tmp_path):
    out_path = tmp_path / "out.pdf"
    with pytest.raises(ValueError):
        writer.write([], out_path, "Title")
    assert not out_path.exists()
