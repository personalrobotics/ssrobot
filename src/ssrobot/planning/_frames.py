"""Poses as homogeneous transforms, and ssrobot regions as sstsr regions."""

from __future__ import annotations

import numpy as np
import tsr

from ssrobot.conventions import Pose
from ssrobot.planning_contracts import Tsr


def matrix(pose: Pose) -> np.ndarray:
    """The 4x4 homogeneous transform of ``pose``."""
    w, x, y, z = pose.quat_wxyz
    T = np.eye(4)
    T[:3, :3] = (
        (1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)),
        (2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)),
        (2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)),
    )
    T[:3, 3] = pose.position
    return T


def region(region: Tsr) -> tsr.TSR:
    """The sstsr region with the same frames and bounds."""
    bounds = np.array([*region.position_bounds, *region.rotation_bounds], dtype=float)
    return tsr.TSR(T0_w=matrix(region.origin), Tw_e=matrix(region.offset), Bw=bounds)


def box(T0_w: np.ndarray, Tw_e: np.ndarray, position: np.ndarray, rotation: float) -> tsr.TSR:
    """A region about ``T0_w``: per-axis position bounds (3x2) and ±``rotation``."""
    bounds = np.vstack([position, [[-rotation, rotation]] * 3])
    return tsr.TSR(T0_w=T0_w, Tw_e=Tw_e, Bw=bounds)
