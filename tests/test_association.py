import pytest

from helmet_detection.association import ExpansionFactors, associate, center_in_box, expand_box

FACTORS = ExpansionFactors(x=1.3, up=2.0, down=0.1)


def test_expand_box_is_asymmetric():
    # 100x50 motorcycle box in a large image
    x1, y1, x2, y2 = expand_box((100, 200, 200, 250), FACTORS, 1000, 1000)
    assert (x1, x2) == pytest.approx((85, 215))       # width 100 * 1.3, centred on 150
    assert y1 == pytest.approx(100)                   # 2.0 * 50 above
    assert y2 == pytest.approx(255)                   # 0.1 * 50 below


def test_expand_box_is_clipped_to_image():
    assert expand_box((0, 10, 100, 60), FACTORS, 110, 62) == pytest.approx((0, 0, 110, 62))


def test_center_in_box_includes_edges():
    box = (0, 0, 10, 10)
    assert center_in_box(box, 10, 0)
    assert not center_in_box(box, 10.01, 5)


def test_associate_splits_violation_from_pedestrian():
    moto = [(100, 200, 200, 250)]                     # zone: x 85..215, y 100..255
    dets = [
        ((140, 110, 160, 130), 0.9, 1),               # head above the bike -> violation
        ((600, 110, 620, 130), 0.8, 1),               # far from any bike -> suppressed
        ((150, 150, 170, 170), 0.7, 0),               # helmet, never filtered
    ]
    r = associate(dets, moto, 1000, 1000, FACTORS)
    assert [c for _, c in r.violations] == [0.9]
    assert [c for _, c in r.suppressed] == [0.8]
    assert [c for _, c in r.helmets] == [0.7]


def test_associate_without_motorcycles_suppresses_all_violations():
    r = associate([((0, 0, 10, 10), 0.9, 1)], [], 100, 100, FACTORS)
    assert r.violations == [] and len(r.suppressed) == 1
