"""Placeholder for a future 6-DOF drone dynamics model.

A drone model will plug in exactly like the 2D vehicles:

1. Add ``DynamicsModel.DRONE`` handling to :mod:`wmas.dynamics.kernels`
   (a new branch computing the state derivative from rotor commands).
2. Extend :class:`wmas.core.state.WorldState` with the extra state fields
   (altitude, attitude quaternion / roll-pitch, body rates).
3. Extend the parameter matrix in :mod:`wmas.dynamics.base` with drone
   parameters (thrust limits, inertia, arm length).

The core stepping, autograd bridge, neighbor search, and scenario API are
agnostic to the per-agent model tag, so no other code changes are required.
"""

from __future__ import annotations


class DronePlaceholder:
    """6-DOF drone dynamics are not implemented yet."""

    def __init__(self, *args, **kwargs):
        raise NotImplementedError(
            "6-DOF drone dynamics are not implemented yet. "
            "See wmas/dynamics/drone.py for the intended plug-in point."
        )
