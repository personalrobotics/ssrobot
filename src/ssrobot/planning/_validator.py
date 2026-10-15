"""sscbirrt's public validator behind ssrobot's MotionValidator contract."""

from __future__ import annotations

import math
from collections.abc import Sequence
from importlib.metadata import version
from typing import Any

import numpy as np
from sscbirrt import ValidationReport
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
    path_identity,
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

_LAW = "piecewise_constant_acceleration"
"""The law sscbirrt reports for a timed trajectory: ssrobot's
``TimingLaw.PIECEWISE_CONSTANT_ACCELERATION``."""

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
        subject = trajectory_subject(trajectory)
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
        return self.record(self._validator.validate_trajectory(native), subject, evidence=path)

    def record(
        self,
        report: Any,
        subject: MotionSubject | TrajectorySubject,
        evidence: GeometricPath | None = None,
    ) -> MotionValidation:
        """ssrobot's record of an sscbirrt report about ``subject``, validated field by
        field before it is translated: for the planner's own reports
        (``PlanResult.validation``, ``Trajectory.validation``) as well as this
        validator's answers. A report that disagrees with the subject, the policy, or its
        own types is refused (``planner_contract``), never relabelled."""

        def refuse(message: str) -> ValidationError:
            return ValidationError("planner_contract", f"sscbirrt's report {message}")

        if not isinstance(report, ValidationReport):
            raise refuse(f"is a {type(report).__name__}, not a ValidationReport")
        for name, kind_of in (("valid", bool), ("intervals", int), ("checks", int)):
            value = getattr(report, name)
            if not isinstance(value, kind_of) or (kind_of is int and isinstance(value, bool)):
                raise refuse(f"has a malformed {name}: {value!r}")
        if f"{report.implementation} {report.backend}" != self._implementation:
            raise refuse(f"is from {report.implementation} on {report.backend}")
        if policy(report.policy) != self._identity.policy:
            raise refuse("applies another policy than this validator's")
        timed = isinstance(subject, TrajectorySubject)
        kind = ValidationSubject.TRAJECTORY if timed else subject.kind  # type: ignore[union-attr]
        if report.subject != kind.value:
            raise refuse(f"is about a {report.subject}, not a {kind.value}")
        linear = kind in (ValidationSubject.MOTION, ValidationSubject.PATH)
        if report.interpolation != ("raw_linear" if linear else None):
            raise refuse(f"interpolates {report.interpolation!r}")
        if report.law != (_LAW if timed else None):
            raise refuse(f"has the law {report.law!r}")
        if timed:
            assert isinstance(subject, TrajectorySubject)
            expected_intervals = len(subject.pieces)
            duration: float | None = subject.duration
            if (
                not isinstance(report.duration_s, float)
                or abs(report.duration_s - subject.duration) > DURATION_AGREEMENT
            ):
                raise refuse(f"times it at {report.duration_s} s, not {subject.duration} s")
        else:
            assert isinstance(subject, MotionSubject)
            expected_intervals = len(subject.waypoints) - 1
            duration = None
            if report.duration_s is not None:
                raise refuse("has a duration for an untimed subject")
        if report.intervals != expected_intervals:
            raise refuse(f"counts {report.intervals} intervals, not {expected_intervals}")
        if report.valid != (report.witness is None):
            raise refuse("has a witness exactly when it is valid")
        if evidence is not None and not timed:
            raise refuse("cannot cite a path as evidence for an untimed subject")
        record = MotionValidation(
            subject=kind,
            motion=fingerprint(subject),
            interpolation=PathInterpolation.LINEAR if linear else None,
            law=subject.law if isinstance(subject, TrajectorySubject) else None,
            intervals=report.intervals,
            duration_s=duration,
            validator=self._identity,
            valid=report.valid,
            witness=None if report.witness is None else self._witness(report.witness, kind),
            checks=report.checks,
            evidence=None if evidence is None else path_identity(evidence),
        )
        check_validation(record, validator=self._identity, subject=subject, evidence=evidence)
        return record

    def _witness(self, witness: Any, kind: ValidationSubject) -> InvalidWitness:
        if witness.failure not in _FAILURES or len(witness.configuration) != len(self._joints):
            raise ValidationError(
                "planner_contract",
                f"sscbirrt's witness is malformed: {witness.failure!r}, "
                f"{len(witness.configuration)} values for {len(self._joints)} joints",
            )
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
