from pathlib import Path

from report_pipeline.domain import normalize_group, parse_label_group


def test_normalize_group_trailing_digits():
    assert normalize_group("g12") == "g"


def test_normalize_group_none():
    assert normalize_group(None) is None


def test_parse_label_group_with_group():
    label, group = parse_label_group(Path("sample__g3.txt"))
    assert label == "sample"
    assert group == "g"


def test_parse_label_group_without_group():
    label, group = parse_label_group(Path("sample.txt"))
    assert label == "sample"
    assert group is None
