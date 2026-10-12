"""Inverse kinematics for one manipulator, from ssik on the package's own model."""

from __future__ import annotations

import numpy as np
import ssik
from sscbirrt.backends.ssik import SSIKSolver

from ssrobot.errors import CapabilityError, ValidationError
from ssrobot.package import ModelFormat, RobotPackage
from ssrobot.planning._frames import matrix
from ssrobot.scene import PlanningScene

AGREEMENT = 1e-6
"""How far, in meters and rotation-matrix entries, ssik's forward kinematics may differ
from the planning scene's before the kinematics are refused."""

AGREEMENT_SAMPLES = 8
"""Seeded configurations, besides the start, at which agreement is checked."""


class SsikKinematics:
    """Analytic IK for one manipulator, built by ssik from the package's canonical MJCF
    between the manipulator's base and tool frames.

    It is bound to a planning scene at plan time: the base frame's world pose is read
    from the scene at the snapshot, so anything below the base that the group does not
    move, such as a lift, is wherever the snapshot put it; and the end effector's offset
    from the tool frame is read from the scene too. Binding checks that the composed
    kinematics agree with the scene's forward kinematics, and refuses them otherwise.
    """

    def __init__(
        self,
        solver: ssik.Manipulator,
        *,
        manipulator: str,
        group: str,
        joints: tuple[str, ...],
        base_frame: str,
        tool_frame: str,
    ) -> None:
        self.solver = solver
        self.manipulator = manipulator
        self.group = group
        self.joints = joints
        self.base_frame = base_frame
        self.tool_frame = tool_frame

    @classmethod
    def from_package(cls, package: RobotPackage, manipulator: str) -> SsikKinematics:
        """IK for ``manipulator`` from ``package``'s canonical model, which must be MJCF.

        Fails with ``kinematics_unavailable`` when ssik cannot solve the chain, such as
        one that is not 6 or 7 revolute joints."""
        description = package.description
        arm = description.manipulator(manipulator)
        entry = next(
            m for m in package.manifest.models if m.name == package.manifest.canonical_model
        )
        if entry.format is not ModelFormat.MJCF:
            raise CapabilityError(
                "kinematics_unavailable",
                f"ssik kinematics need an MJCF model; the canonical model is {entry.format}",
            )
        try:
            solver = ssik.Manipulator.from_mjcf(
                package.root / entry.path, base=arm.base_frame, ee=arm.tool_frame
            )
        except (ValueError, NotImplementedError) as e:
            raise CapabilityError(
                "kinematics_unavailable", f"ssik cannot solve {manipulator!r}: {e}"
            ) from None
        joints = description.group(arm.group).joints
        if solver.dof != len(joints):
            raise CapabilityError(
                "kinematics_unavailable",
                f"ssik's chain has {solver.dof} joints; {arm.group!r} has {len(joints)}",
            )
        return cls(
            solver,
            manipulator=manipulator,
            group=arm.group,
            joints=joints,
            base_frame=arm.base_frame,
            tool_frame=arm.tool_frame,
        )

    @property
    def name(self) -> str:
        return f"ssik:{self.solver.solver_name}"

    def bind(self, scene: PlanningScene, frame: str) -> SSIKSolver:
        """sscbirrt's IK for ``frame`` in ``scene``'s world, checked against the scene."""
        if scene.group != self.group:
            raise ValidationError(
                "kinematics_mismatch", f"the kinematics are for {self.group!r}, not {scene.group!r}"
            )
        start = tuple(scene.snapshot.position(j) for j in self.joints)
        T_base = matrix(scene.forward_kinematics(start, self.base_frame))
        T_tool = matrix(scene.forward_kinematics(start, self.tool_frame))
        T_frame = matrix(scene.forward_kinematics(start, frame))
        T_ee = np.linalg.solve(T_tool, T_frame)
        bound = SSIKSolver(self.solver, T_base=T_base, T_ee=T_ee)
        rng = np.random.default_rng(0)
        lower = np.array([-np.pi if lim is None else lim[0] for lim in self.solver.joint_limits])
        upper = np.array([np.pi if lim is None else lim[1] for lim in self.solver.joint_limits])
        samples = [np.array(start)] + [rng.uniform(lower, upper) for _ in range(AGREEMENT_SAMPLES)]
        worst = max(
            float(np.abs(bound.fk(q) - matrix(scene.forward_kinematics(tuple(q), frame))).max())
            for q in samples
        )
        if worst > AGREEMENT:
            raise ValidationError(
                "kinematics_mismatch",
                f"ssik's forward kinematics of {frame!r} differ from the planning scene's by "
                f"{worst:.3g}, above {AGREEMENT}",
            )
        return bound
