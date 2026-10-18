"""Compound rigid shapes built from oriented boxes, for movable-body scenarios.

A :class:`CompoundShape` is what :class:`~swarp.scenarios.pusht.PushTScenario` hands the
engine as one movable compound obstacle: per-box half-extents and body-frame offsets,
plus the body's mass, rotational inertia and bounding radius. :func:`compound` derives
all of it from a list of boxes placed in any frame:

* the body origin is moved to the **area centroid**, so the rotation origin and the
  inertia origin coincide (a contact off the centroid then produces the torque that
  makes orientation controllable);
* the mass splits between boxes by area, and the inertia about the centroid is each
  box's own ``m (w^2 + h^2) / 12`` plus the parallel-axis term ``m |offset|^2``;
* the radius is the farthest box corner from the centroid, which is what keeps the
  whole body inside the world bounds and what spawn clearance is measured against.

The boxes must not overlap: an overlap would count its area (and mass) twice. The
factories below all satisfy that, and ``tests/unit/test_shapes.py`` checks it.

Every factory defaults to a body the size of Push-T's T (area 0.0816 for ``tee``,
``ell`` and ``plus``), so swapping the shape does not also change how hard it is to
push. :data:`SHAPES` maps a name to its factory.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass

#: One box before centring: ``(cx, cy, hx, hy)``, its centre and half-extents.
Box = tuple[float, float, float, float]


@dataclass(frozen=True)
class CompoundShape:
    """A rigid body made of non-overlapping axis-aligned boxes, in its own frame.

    Attributes:
        half: per-box half-extents ``(hx, hy)``.
        off: per-box centre in the body frame, whose origin is the area centroid.
        mass: total mass, split between the boxes by area.
        inertia: rotational inertia about the centroid.
        radius: distance from the centroid to the farthest box corner.
        area: total area.
    """

    half: tuple[tuple[float, float], ...]
    off: tuple[tuple[float, float], ...]
    mass: float
    inertia: float
    radius: float
    area: float

    @property
    def n_boxes(self) -> int:
        return len(self.half)


def compound(boxes: Sequence[Box], mass: float) -> CompoundShape:
    """Centre ``boxes`` on their area centroid and derive mass, inertia and radius.

    The float operations follow the order Push-T's T was originally computed in, so
    :func:`tee` reproduces its geometry bit for bit (``tests/unit/test_shapes.py`` pins
    the values).
    """
    if not boxes:
        raise ValueError("a compound shape needs at least one box")
    areas = [4.0 * hx * hy for _, _, hx, hy in boxes]
    if min(areas) <= 0.0:
        raise ValueError(f"every box needs positive half-extents, got {list(boxes)}")
    area = sum(areas)
    cx = sum(a * bx for a, (bx, _, _, _) in zip(areas, boxes, strict=True)) / area
    cy = sum(a * by for a, (_, by, _, _) in zip(areas, boxes, strict=True)) / area
    half = tuple((hx, hy) for _, _, hx, hy in boxes)
    off = tuple((bx - cx, by - cy) for bx, by, _, _ in boxes)

    # Mass splits by area; inertia about the centroid by the parallel-axis theorem.
    inertia = 0.0
    for a_i, (hx, hy), (ox, oy) in zip(areas, half, off, strict=True):
        m_i = mass * a_i / area
        inertia += m_i * ((2.0 * hx) ** 2 + (2.0 * hy) ** 2) / 12.0 + m_i * ox * ox + m_i * oy * oy
    radius = max(
        math.hypot(ox + sx * hx, oy + sy * hy)
        for (hx, hy), (ox, oy) in zip(half, off, strict=True)
        for sx in (-1.0, 1.0)
        for sy in (-1.0, 1.0)
    )
    return CompoundShape(half=half, off=off, mass=mass, inertia=inertia, radius=radius, area=area)


def tee(
    bar_len: float = 0.40,
    bar_width: float = 0.12,
    stem_len: float = 0.28,
    stem_width: float = 0.12,
    mass: float = 1.0,
) -> CompoundShape:
    """Push-T's T: a crossbar on top, a stem hanging from its middle.

    Box 0 is the crossbar, box 1 the stem.
    """
    y_stem = -(bar_width + stem_len) / 2.0
    return compound(
        [
            (0.0, 0.0, bar_len / 2.0, bar_width / 2.0),
            (0.0, y_stem, stem_width / 2.0, stem_len / 2.0),
        ],
        mass,
    )


def ell(
    leg_len: float = 0.40,
    foot_len: float = 0.28,
    width: float = 0.12,
    mass: float = 1.0,
) -> CompoundShape:
    """An L: a vertical leg, and a foot running right from its bottom end.

    Box 0 is the full-height leg, box 1 the foot beside it (``foot_len`` excludes the
    leg's own width, so the two boxes only touch).
    """
    return compound(
        [
            (0.0, 0.0, width / 2.0, leg_len / 2.0),
            ((width + foot_len) / 2.0, (width - leg_len) / 2.0, foot_len / 2.0, width / 2.0),
        ],
        mass,
    )


def plus(span: float = 0.40, width: float = 0.12, mass: float = 1.0) -> CompoundShape:
    """A plus sign, ``span`` across in both directions.

    Box 0 is the full horizontal bar; boxes 1 and 2 are the vertical arms above and below
    it, each ``(span - width) / 2`` long, so the centre square is counted once.
    """
    if span <= width:
        raise ValueError(f"plus needs span > width, got span={span}, width={width}")
    arm = (span - width) / 2.0
    y_arm = (width + arm) / 2.0
    return compound(
        [
            (0.0, 0.0, span / 2.0, width / 2.0),
            (0.0, y_arm, width / 2.0, arm / 2.0),
            (0.0, -y_arm, width / 2.0, arm / 2.0),
        ],
        mass,
    )


def bar(length: float = 0.40, width: float = 0.12, mass: float = 1.0) -> CompoundShape:
    """A single box: the T's crossbar on its own."""
    return compound([(0.0, 0.0, length / 2.0, width / 2.0)], mass)


#: Shape name -> factory. Every factory takes keyword arguments with T-sized defaults
#: and a ``mass``.
SHAPES: dict[str, Callable[..., CompoundShape]] = {
    "tee": tee,
    "ell": ell,
    "plus": plus,
    "bar": bar,
}
