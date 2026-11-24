import pytest

from helmet_detection.labels import build_remap, convert_label_lines, polygon_to_bbox


def test_build_remap_normalizes_names_and_drops_unknown():
    remap = build_remap(["Helmet", " no-helmet ", "rider", "null"])
    assert remap == {0: 0, 1: 1, 2: None, 3: None}


def test_polygon_to_bbox_is_tight_box():
    # Triangle with x in [0.2, 0.6] and y in [0.1, 0.5]
    line = polygon_to_bbox(["1", "0.2", "0.1", "0.6", "0.3", "0.4", "0.5"])
    cls, cx, cy, w, h = line.split()
    assert cls == "1"
    assert float(cx) == pytest.approx(0.4)
    assert float(cy) == pytest.approx(0.3)
    assert float(w) == pytest.approx(0.4)
    assert float(h) == pytest.approx(0.4)


def test_convert_label_lines_remaps_converts_and_drops():
    remap = {0: 1, 1: 0, 2: None}
    lines = [
        "0 0.5 0.5 0.1 0.1",                 # bbox, remapped 0 -> 1
        "1 0.1 0.1 0.3 0.1 0.3 0.3",         # polygon, remapped 1 -> 0
        "2 0.5 0.5 0.1 0.1",                 # class dropped
        "0 0.5 0.5",                          # malformed
        "",                                   # blank, ignored
    ]
    out, stats = convert_label_lines(lines, remap)
    assert out[0] == "1 0.5 0.5 0.1 0.1"
    assert out[1].startswith("0 0.200000 0.200000 0.200000 0.200000")
    assert len(out) == 2
    assert (stats.remapped, stats.polygons, stats.dropped) == (3, 1, 2)


def test_convert_label_lines_is_idempotent():
    remap = {0: 0, 1: 1}
    lines = ["0 0.5 0.5 0.1 0.1", "1 0.2 0.2 0.1 0.1"]
    out, stats = convert_label_lines(lines, remap)
    assert out == lines
    assert (stats.remapped, stats.polygons, stats.dropped) == (0, 0, 0)
