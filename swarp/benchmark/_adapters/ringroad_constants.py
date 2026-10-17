"""Ring-road constants shared by the VMAS and JAX ports (``ringroad_vmas`` / ``ringroad_jax``).

They mirror :class:`swarp.scenarios.ringroad.RingRoadScenario`'s defaults, so the three
implementations the cross-simulator benchmark times run the same task. Kept in a module
with no simulator imports, because each port is imported in its own simulator's venv.
"""

from __future__ import annotations

# RingRoadScenario's defaults.
SPACING = 230.0 / 22.0
CAR_LENGTH = 5.0
LANE_WIDTH = 3.7
V_MAX = 30.0
IDM = dict(v0=30.0, T=1.0, a=1.0, b=1.5, delta=4.0, s0=2.0)
S_EPS, DECEL_FLOOR = 0.1, -9.0
MAX_ACCEL, MAX_STEER = 2.0, 0.6
GAP_AMP, V_AMP = 3.0, 3.0
SPEED_COEF, ACCEL_COEF, LANE_COEF = 0.2, 1.0, 0.1
CRASH_PENALTY, TIME_PENALTY, LANE_CAP = -20.0, -1.5, 3.0
STD_THRESHOLD, SPEED_FRACTION, HOLD_STEPS = 0.3, 0.5, 100
V_SCALE, GAP_SCALE = 15.0, 10.0
WHEELBASE = 0.5 * CAR_LENGTH


def idm_equilibrium_speed(gap: float) -> float:
    """``1 - (v/v0)^delta = ((s0 + v T)/gap)^2`` by bisection (host side, once)."""
    lo, hi = 0.0, IDM["v0"]
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        q = (IDM["s0"] + mid * IDM["T"]) / gap
        lo, hi = (mid, hi) if 1.0 - (mid / IDM["v0"]) ** IDM["delta"] - q * q > 0.0 else (lo, mid)
    return min(0.5 * (lo + hi), V_MAX)
