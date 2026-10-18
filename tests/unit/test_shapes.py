"""Compound shapes: the T is bit-identical to Push-T's original geometry, and the other
factories match their analytic area, centroid and inertia."""

import itertools
import math

import pytest

from swarp.scenarios.pusht import PushTScenario
from swarp.scenarios.shapes import SHAPES, bar, compound, ell, plus, tee


def _legacy_tee(bar_len, bar_width, stem_len, stem_width, mass):
    """Push-T's T geometry exactly as ``PushTScenario.__init__`` computed it before
    ``swarp.scenarios.shapes`` existed."""
    a_bar = bar_len * bar_width
    a_stem = stem_width * stem_len
    y_stem = -(bar_width + stem_len) / 2.0
    cy = a_stem * y_stem / (a_bar + a_stem)
    half = ((bar_len / 2.0, bar_width / 2.0), (stem_width / 2.0, stem_len / 2.0))
    off = ((0.0, -cy), (0.0, y_stem - cy))
    inertia = 0.0
    for (hx, hy), (_, oy) in zip(half, off, strict=True):
        m_i = mass * (4.0 * hx * hy) / (a_bar + a_stem)
        inertia += m_i * ((2.0 * hx) ** 2 + (2.0 * hy) ** 2) / 12.0 + m_i * oy * oy
    radius = max(
        math.hypot(hx, oy + sy * hy)
        for (hx, hy), (_, oy) in zip(half, off, strict=True)
        for sy in (-1.0, 1.0)
    )
    return half, off, inertia, radius


def test_default_tee_is_bit_identical_to_the_original_push_t_geometry():
    # Printed by the pre-refactor PushTScenario(); repr round-trips a float exactly.
    s = PushTScenario()
    assert s.box_half == ((0.2, 0.06), (0.06, 0.14))
    assert s.box_off == ((0.0, 0.0823529411764706), (0.0, -0.11764705882352941))
    assert s.tee_inertia == 0.021421914648212227
    assert s.tee_radius == 0.2645411251968496
    assert s.n_boxes == 2 and s.tee_mass == 1.0


@pytest.mark.parametrize(
    "dims",
    [
        (0.40, 0.12, 0.28, 0.12, 1.0),
        (0.5, 0.1, 0.3, 0.08, 2.5),
        (0.33, 0.07, 0.41, 0.13, 0.7),
    ],
)
def test_tee_matches_the_legacy_formula_exactly(dims):
    half, off, inertia, radius = _legacy_tee(*dims)
    t = tee(*dims[:4], mass=dims[4])
    assert t.half == half
    assert t.off == off
    assert t.inertia == inertia
    assert t.radius == radius
    s = PushTScenario(
        bar_len=dims[0], bar_width=dims[1], stem_len=dims[2], stem_width=dims[3], tee_mass=dims[4]
    )
    assert (s.box_half, s.box_off, s.tee_inertia, s.tee_radius) == (half, off, inertia, radius)


def _box_bounds(shape):
    return [
        (ox - hx, ox + hx, oy - hy, oy + hy)
        for (hx, hy), (ox, oy) in zip(shape.half, shape.off, strict=True)
    ]


@pytest.mark.parametrize("name", sorted(SHAPES))
def test_every_shape_is_centred_non_overlapping_and_bounded(name):
    s = SHAPES[name]()
    areas = [4.0 * hx * hy for hx, hy in s.half]
    assert sum(areas) == pytest.approx(s.area, rel=1e-12)
    # Centroid at the body origin.
    for k in (0, 1):
        assert sum(a * o[k] for a, o in zip(areas, s.off, strict=True)) == pytest.approx(
            0.0, abs=1e-12
        )
    # No two boxes share area (touching edges are fine).
    for (l1, r1, b1, t1), (l2, r2, b2, t2) in itertools.combinations(_box_bounds(s), 2):
        assert min(r1, r2) - max(l1, l2) <= 1e-12 or min(t1, t2) - max(b1, b2) <= 1e-12
    # The radius is the farthest corner, and every corner is inside it.
    corners = [
        math.hypot(x, y) for x0, x1, y0, y1 in _box_bounds(s) for x in (x0, x1) for y in (y0, y1)
    ]
    assert max(corners) == pytest.approx(s.radius, rel=1e-12)
    assert s.mass == 1.0 and s.inertia > 0.0


def test_bar_matches_the_rectangle_formula():
    length, width, mass = 0.5, 0.2, 3.0
    s = bar(length, width, mass=mass)
    assert s.area == pytest.approx(length * width)
    assert s.inertia == pytest.approx(mass * (length**2 + width**2) / 12.0, rel=1e-12)
    assert s.radius == pytest.approx(math.hypot(length / 2, width / 2), rel=1e-12)
    assert s.off == ((0.0, 0.0),)


def _rect_moments(w, h, cx, cy):
    """Area, first moments and second polar moment about the origin of a w x h rectangle
    centred at (cx, cy)."""
    a = w * h
    return a, a * cx, a * cy, a * (w * w + h * h) / 12.0 + a * (cx * cx + cy * cy)


def test_plus_matches_two_crossing_bars_minus_the_doubled_centre():
    span, width, mass = 0.5, 0.1, 2.0
    s = plus(span, width, mass=mass)
    # horizontal bar + vertical bar - the centre square they both cover
    terms = [
        _rect_moments(span, width, 0, 0),
        _rect_moments(width, span, 0, 0),
        _rect_moments(width, width, 0, 0),
    ]
    area = terms[0][0] + terms[1][0] - terms[2][0]
    j = terms[0][3] + terms[1][3] - terms[2][3]
    assert s.area == pytest.approx(area, rel=1e-12)
    assert s.inertia == pytest.approx(mass / area * j, rel=1e-12)
    assert s.radius == pytest.approx(math.hypot(span / 2, width / 2), rel=1e-12)


def test_ell_matches_its_bounding_box_minus_the_missing_corner():
    leg, foot, width, mass = 0.45, 0.3, 0.1, 1.5
    s = ell(leg, foot, width, mass=mass)
    # In the input frame the leg spans x in [-w/2, w/2], y in [-leg/2, leg/2], and the
    # foot extends right of it along the bottom. The L is that bounding box minus the
    # empty top-right rectangle.
    x0, x1 = -width / 2, width / 2 + foot
    y0, y1 = -leg / 2, leg / 2
    full = _rect_moments(x1 - x0, y1 - y0, (x0 + x1) / 2, (y0 + y1) / 2)
    hole_x0, hole_y0 = width / 2, y0 + width
    hole = _rect_moments(x1 - hole_x0, y1 - hole_y0, (hole_x0 + x1) / 2, (hole_y0 + y1) / 2)
    area = full[0] - hole[0]
    cx, cy = (full[1] - hole[1]) / area, (full[2] - hole[2]) / area
    j_origin = full[3] - hole[3]
    j_centroid = j_origin - area * (cx * cx + cy * cy)  # parallel axis, backwards
    assert s.area == pytest.approx(area, rel=1e-12)
    assert s.inertia == pytest.approx(mass / area * j_centroid, rel=1e-12)
    # The leg's offset is minus the centroid (it was centred on the input origin).
    assert s.off[0] == pytest.approx((-cx, -cy), rel=1e-12)


def test_default_shapes_are_t_sized():
    t = tee()
    for name in ("ell", "plus"):
        assert SHAPES[name]().area == pytest.approx(t.area, rel=1e-12)


def test_pusht_accepts_a_shape_by_name_or_object():
    s = PushTScenario(shape="plus", tee_mass=2.0)
    assert s.n_boxes == 3 and s.tee_mass == 2.0
    assert s.tee_inertia == plus(mass=2.0).inertia
    obj = ell(mass=0.5)
    s = PushTScenario(shape=obj)
    assert s.shape is obj and s.tee_mass == 0.5 and s.box_off == obj.off
    with pytest.raises(ValueError, match="unknown shape"):
        PushTScenario(shape="hexagon")


def test_compound_rejects_degenerate_input():
    with pytest.raises(ValueError):
        compound([], 1.0)
    with pytest.raises(ValueError):
        compound([(0.0, 0.0, 0.1, 0.0)], 1.0)
