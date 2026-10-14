"""sscbirrt's public validator behind ssrobot's MotionValidator contract."""

from __future__ import annotations

import math
from collections.abc import Sequence
from importlib.metadata import version
from typing import Any

import numpy as np
from sscbirrt.trajectory import Piece, Ramp, Trajectory

from ssrobot._wire import fingerprint
from ssrobot.errors import ValidationError
from ssrobot.planning_contracts import (
    GeometricPath,
    InvalidWitness,
    MotionSubject,
    MotionValidation,
    PathInterpolation,
    PlanningRequest,
    TimedTrajectory,
    TrajectorySubject,
    ValidationFailure,
    ValidationGuarantee,
    ValidationPolicy,
    ValidationSubject,
    ValidatorIdentity,
    check_validation,
    constraints_identity,
    motion_subject,
    path_subject,
    state_subject,
    trajectory_subject,
)
from ssrobot.scene import Contact, PlanningScene

_FAILURES = {
    "position_limits": ValidationFailure.POSITION_LIMITS,
    "collision": ValidationFailure.COLLISION,
    "path_constraint": ValidationFailure.PATH_CONSTRAINT,
    "unsupported": ValidationFailure.UNSUPPORTED_MOTION,
}

DURATION_AGREEMENT = 1e-9
"""How far, in seconds, sscbirrt's trajectory duration may differ from ssrobot's own sum
of the same pieces before the report is refused."""


def policy(source: Any) -> ValidationPolicy:
    """ssrobot's record of an sscbirrt ``ValidationPolicy``, every setting included."""
    return ValidationPolicy(
        name=source.name,
        mechanism=source.mechanism,
        guarantee=ValidationGuarantee(source.guarantee),
        resolution=float(source.resolution),
        ambiguity_band_m=float(source.ambiguity_band_m),
        parameters=tuple(sorted((str(k), str(v)) for k, v in source.parameters)),
        summary=source.summary,
    )


class SscbirrtValidator:
    """The ``MotionValidator`` for one request and planning scene: sscbirrt's validator,
    the one ``CBiRRT.plan`` applies to the same request (``CBiRRT.validator``), with its
    answers recorded as ``MotionValidation``.

    It runs no sampling, interpolation, or constraint check of its own. Subjects and
    identities are ssrobot's, so reports bind to the exact motion asked about; contacts
    in a witness come from the planning scene, so they carry ssrobot's names.
    """

    def __init__(self, validator: Any, *, request: PlanningRequest, scene: PlanningScene) -> None:
        self._validator = validator
        self._scene = scene
        self._group = request.group
        self._joints = request.joints
        self._implementation = f"sscbirrt {version('sscbirrt')} {validator.backend}"
        self._identity = ValidatorIdentity(
            implementation=self._implementation,
            policy=policy(validator.policy),
            snapshot=request.snapshot,
            group=request.group,
            joints=request.joints,
            constraints=constraints_identity(request),
        )

    @property
    def identity(self) -> ValidatorIdentity:
        return self._identity

    @property
    def native(self) -> Any:
        """The wrapped ``sscbirrt.Validator``."""
        return self._validator

    def validate_state(self, q: Sequence[float]) -> MotionValidation:
        subject = state_subject(self._group, self._joints, q)
        report = self._validator.validate_state(np.array(subject.waypoints[0]))
        return self.record(report, subject)

    def validate_motion(self, q0: Sequence[float], q1: Sequence[float]) -> MotionValidation:
        subject = motion_subject(self._group, self._joints, q0, q1)
        a, b = (np.array(q) for q in subject.waypoints)
        return self.record(self._validator.validate_motion(a, b), subject)

    def validate_path(self, path: GeometricPath) -> MotionValidation:
        subject = path_subject(path)
        if path.interpolation is not PathInterpolation.LINEAR:
            raise ValidationError(
                "unsupported_motion", "sscbirrt validates linear paths", path="interpolation"
            )
        report = self._validator.validate_path([np.array(q) for q in subject.waypoints])
        return self.record(report, subject)

    def validate_trajectory(
        self, trajectory: TimedTrajectory, path: GeometricPath | None = None
    ) -> MotionValidation:
        subject = trajectory_subject(trajectory, path)
        native = Trajectory(
            pieces=[
                Piece(
                    ramps=tuple(
                        Ramp(
                            ramp.position,
                            ramp.velocity,
                            tuple((i.duration, i.acceleration) for i in ramp.intervals),
                        )
                        for ramp in piece.ramps
                    ),
                    validated=piece.validation.value,
                )
                for piece in trajectory.pieces
            ],
            velocity_limits=np.array(trajectory.limits.effective_velocity),
            acceleration_limits=np.array(trajectory.limits.effective_acceleration),
            joint_names=trajectory.joints,
            path=None if path is None else [np.array(q) for q in path.waypoints],
        )
        return self.record(self._validator.validate_trajectory(native), subject)

    def record(self, report: Any, subject: MotionSubject | TrajectorySubject) -> MotionValidation:
        """ssrobot's record of an sscbirrt report about ``subject``, checked against it:
        for the planner's own reports, ``PlanResult.validation`` and
        ``Trajectory.validation``, as well as this validator's answers."""
        if f"{report.implementation} {report.backend}" != self._implementation:
            raise ValidationError(
                "planner_contract",
                f"{report.implementation} on {report.backend} answered for {self._implementation}",
            )
        timed = isinstance(subject, TrajectorySubject)
        if timed:
            assert isinstance(subject, TrajectorySubject)
            duration: float | None = subject.duration
            if report.duration_s is None or abs(report.duration_s - subject.duration) > (
                DURATION_AGREEMENT
            ):
                raise ValidationError(
                    "planner_contract",
                    f"sscbirrt timed the trajectory at {report.duration_s} s, "
                    f"not {subject.duration} s",
                )
            kind = ValidationSubject.TRAJECTORY
        else:
            assert isinstance(subject, MotionSubject)
            duration, kind = None, subject.kind
        record = MotionValidation(
            subject=kind,
            motion=fingerprint(subject),
            interpolation=None if timed else getattr(subject, "interpolation", None),
            law=subject.law if isinstance(subject, TrajectorySubject) else None,
            intervals=int(report.intervals),
            duration_s=duration,
            validator=self._identity,
            valid=bool(report.valid),
            witness=None if report.witness is None else self._witness(report.witness, kind),
            checks=int(report.checks),
        )
        check_validation(record, validator=self._identity, subject=subject)
        return record

    def _witness(self, witness: Any, kind: ValidationSubject) -> InvalidWitness:
        configuration = tuple(float(v) for v in witness.configuration)
        failure = _FAILURES[witness.failure]
        linear = kind in (ValidationSubject.MOTION, ValidationSubject.PATH)
        detail = witness.detail
        if kind is ValidationSubject.TRAJECTORY and witness.segment is not None:
            detail = f"piece {witness.segment}: {detail}"
        contacts: tuple[Contact, ...] = ()
        if failure is ValidationFailure.COLLISION and all(math.isfinite(v) for v in configuration):
            contacts = self._scene.contacts(configuration)
        return InvalidWitness(
            failure=failure,
            segment=int(witness.segment) if linear else None,
            fraction=float(witness.fraction) if linear else None,
            time_s=float(witness.time_s) if kind is ValidationSubject.TRAJECTORY else None,
            configuration=configuration,
            contacts=contacts,
            constraint=int(witness.constraint)
            if failure is ValidationFailure.PATH_CONSTRAINT
            else None,
            detail=detail,
        )
