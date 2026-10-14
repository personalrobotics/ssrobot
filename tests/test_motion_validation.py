"""The neutral MotionValidator contract (#132, step 1).

A test-only validator for a toy world, with no planner, simulator, or runtime, shows that
the contract can be implemented independently of sscbirrt. The world has two joints:
``turn`` is continuous, ``lift`` is bounded to [-2, 2], and a box in joint space is
forbidden. A straight segment or a piecewise-quadratic trajectory can be checked against
a box exactly, so this validator honestly reports a continuous guarantee. The reports
are written to $SSROBOT_ARTIFACTS/<test>/reports.json; a second run must be
byte-identical. Reproduce with ``uv run pytest tests/test_motion_validation.py``.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import math
from collections.abc import Callable, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

from ssrobot import (
    EffectiveLimits,
    GeometricPath,
    MotionValidation,
    SsrobotError,
    TimedTrajectory,
    dumps,
    loads,
)
from ssrobot.planning_contracts import (
    InvalidWitness,
    PathInterpolation,
    PieceValidation,
    Ramp,
    RampInterval,
    TimingLaw,
    TrajectoryPiece,
    ValidationFailure,
    ValidationGuarantee,
    ValidationPolicy,
    ValidationSubject,
    ValidatorIdentity,
    check_validation,
    motion_identity,
    path_identity,
    state_identity,
    trajectory_identity,
)
from tests.test_planning_contracts import _triangles

GROUP = "toy"
JOINTS = ("turn", "lift")
CONTINUOUS = (True, False)
LIFT_LIMITS = (-2.0, 2.0)
BOX = ((0.4, 0.6), (0.4, 0.6))  # forbidden: turn and lift both in [0.4, 0.6]


def _digest(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def _inside_times(
    start: float, velocity: float, acceleration: float, duration: float, low: float, high: float
) -> list[tuple[float, float]]:
    """The times in [0, duration] at which start + v t + a t^2 / 2 lies in [low, high]."""
    cuts = {0.0, duration}
    for bound in (low, high):
        a, b, c = acceleration / 2, velocity, start - bound
        if a == 0.0:
            if b != 0.0:
                cuts.add(-c / b)
        else:
            disc = b * b - 4 * a * c
            if disc >= 0.0:
                root = math.sqrt(disc)
                cuts.update({(-b - root) / (2 * a), (-b + root) / (2 * a)})
    times = sorted(t for t in cuts if 0.0 <= t <= duration)
    spans: list[tuple[float, float]] = []
    for t0, t1 in itertools.pairwise(times):
        mid = (t0 + t1) / 2
        x = start + velocity * mid + acceleration * mid * mid / 2
        if low <= x <= high:
            spans.append((t0, t1))
    for t in times:  # a touch at a single instant counts too
        x = start + velocity * t + acceleration * t * t / 2
        if low <= x <= high and not any(a <= t <= b for a, b in spans):
            spans.append((t, t))
    return sorted(spans)


def _first_inside(
    axes: Sequence[tuple[float, float, float]], duration: float, box: Sequence[tuple[float, float]]
) -> float | None:
    """The first time in [0, duration] at which every axis (start, velocity,
    acceleration) is inside its box bounds, or None."""
    candidates: list[tuple[float, float]] = [(0.0, duration)]
    for (x0, v, a), (low, high) in zip(axes, box, strict=True):
        spans = _inside_times(x0, v, a, duration, low, high)
        candidates = [
            (max(c0, s0), min(c1, s1))
            for c0, c1 in candidates
            for s0, s1 in spans
            if max(c0, s0) <= min(c1, s1)
        ]
    return min((c0 for c0, _ in candidates), default=None)


def _first_outside(
    axis: tuple[float, float, float], duration: float, low: float, high: float
) -> float | None:
    """The first time in [0, duration] at which the axis is strictly outside [low, high];
    reaching a bound exactly is within it."""
    x0, v, a = axis
    if not low <= x0 <= high:
        return 0.0
    reach = 0.0
    for t0, t1 in _inside_times(x0, v, a, duration, low, high):
        if t0 > reach:
            return reach
        reach = max(reach, t1)
    return None if reach >= duration else reach


class IntervalWorld:
    """An exact validator for the toy world, bound to one policy. ``band`` inflates the
    box: a motion entering the band but never the box is refused as ambiguous."""

    def __init__(self, band: float = 0.0) -> None:
        self._band = band
        policy = ValidationPolicy(
            name="exact_box",
            mechanism="exact",
            guarantee=ValidationGuarantee.CONTINUOUS,
            ambiguity_band_m=band,
            summary="Every state along the motion is checked against the box exactly.",
        )
        self._identity = ValidatorIdentity(
            implementation="interval-world-1",
            policy=policy,
            snapshot=_digest("interval-world snapshot"),
            group=GROUP,
            joints=JOINTS,
            constraints=_digest("interval-world constraints"),
        )

    @property
    def identity(self) -> ValidatorIdentity:
        return self._identity

    def _boxes(self) -> tuple[Sequence[tuple[float, float]], Sequence[tuple[float, float]]]:
        grown = tuple((low - self._band, high + self._band) for low, high in BOX)
        return BOX, grown

    def _report(self, subject: ValidationSubject, **fields: Any) -> MotionValidation:
        witness = fields.pop("witness", None)
        return MotionValidation(
            subject=subject,
            validator=self._identity,
            valid=witness is None,
            witness=witness,
            **fields,
        )

    def _segment(
        self, q0: Sequence[float], q1: Sequence[float]
    ) -> tuple[float, ValidationFailure] | None:
        """The first fraction at which the straight segment becomes invalid, and why."""
        if any(c and abs(b - a) >= math.pi for c, a, b in zip(CONTINUOUS, q0, q1, strict=True)):
            return 0.0, ValidationFailure.UNSUPPORTED_MOTION
        axes = [(a, b - a, 0.0) for a, b in zip(q0, q1, strict=True)]
        return self._first(axes, 1.0)

    def _first(
        self, axes: Sequence[tuple[float, float, float]], duration: float
    ) -> tuple[float, ValidationFailure] | None:
        box, grown = self._boxes()
        found: list[tuple[float, ValidationFailure]] = []
        hit = _first_inside(axes, duration, grown)
        if hit is not None:
            collides = _first_inside(axes, duration, box) is not None
            found.append(
                (hit, ValidationFailure.COLLISION if collides else ValidationFailure.AMBIGUOUS)
            )
        exit_time = _first_outside(axes[1], duration, *LIFT_LIMITS)
        if exit_time is not None:
            found.append((exit_time, ValidationFailure.POSITION_LIMITS))
        return min(found, default=None)

    def validate_state(self, q: Sequence[float]) -> MotionValidation:
        q = tuple(float(v) for v in q)
        problem = self._first([(x, 0.0, 0.0) for x in q], 0.0)
        witness = None
        if problem is not None:
            witness = InvalidWitness(failure=problem[1], configuration=q)
        return self._report(
            ValidationSubject.STATE,
            motion=state_identity(GROUP, JOINTS, q),
            intervals=0,
            checks=1,
            witness=witness,
        )

    def validate_motion(self, q0: Sequence[float], q1: Sequence[float]) -> MotionValidation:
        return self._path((tuple(map(float, q0)), tuple(map(float, q1))), single=True)

    def validate_path(self, path: GeometricPath) -> MotionValidation:
        return self._path(path.waypoints, single=False)

    def _path(self, waypoints: Sequence[tuple[float, ...]], single: bool) -> MotionValidation:
        witness = None
        checks = 0
        for i, (q0, q1) in enumerate(itertools.pairwise(waypoints)):
            checks += 1
            problem = self._segment(q0, q1)
            if problem is not None:
                s, failure = problem
                witness = InvalidWitness(
                    failure=failure,
                    segment=i,
                    fraction=s,
                    configuration=tuple(a + s * (b - a) for a, b in zip(q0, q1, strict=True)),
                )
                break
        if single:
            identity = motion_identity(GROUP, JOINTS, waypoints[0], waypoints[1])
            subject = ValidationSubject.MOTION
        else:
            identity = path_identity(
                GeometricPath(
                    group=GROUP,
                    joints=JOINTS,
                    waypoints=tuple(waypoints),
                    validation_resolution=1.0,
                )
            )
            subject = ValidationSubject.PATH
        return self._report(
            subject,
            motion=identity,
            interpolation=PathInterpolation.LINEAR,
            intervals=len(waypoints) - 1,
            checks=checks,
            witness=witness,
        )

    def validate_trajectory(self, trajectory: TimedTrajectory) -> MotionValidation:
        """Every constant-acceleration span of every piece, exactly."""
        witness = None
        checks = 0
        elapsed = 0.0
        for piece in trajectory.pieces:
            cuts = sorted(
                {0.0}
                | {
                    sum(i.duration for i in r.intervals[: k + 1])
                    for r in piece.ramps
                    for k in range(len(r.intervals))
                }
            )
            for t0, t1 in itertools.pairwise(cuts):
                if t1 <= t0:
                    continue
                checks += 1
                # Each joint's state at t0, and its acceleration on (t0, t1).
                axes = [(*ramp.state(t0)[:2], ramp.state((t0 + t1) / 2)[2]) for ramp in piece.ramps]
                problem = self._first(axes, t1 - t0)
                if problem is not None:
                    s, failure = problem
                    time = elapsed + t0 + s
                    witness = InvalidWitness(
                        failure=failure,
                        time_s=time,
                        configuration=trajectory.evaluate(time)[0],
                    )
                    break
            if witness is not None:
                break
            elapsed += piece.duration
        return self._report(
            ValidationSubject.TRAJECTORY,
            motion=trajectory_identity(trajectory),
            law=trajectory.law,
            intervals=len(trajectory.pieces),
            duration_s=trajectory.duration,
            checks=checks,
            witness=witness,
        )


def _limits() -> EffectiveLimits:
    return EffectiveLimits(
        joints=JOINTS,
        velocity=(1.0, 1.0),
        acceleration=(2.0, 2.0),
        velocity_scaling=1.0,
        acceleration_scaling=1.0,
        effective_velocity=(1.0, 1.0),
        effective_acceleration=(2.0, 2.0),
    )


def _l_shaped(q0: tuple[float, float], q1: tuple[float, float]) -> TimedTrajectory:
    """From q0 to q1 with ``turn`` moving first and ``lift`` second: the same endpoints
    as the straight line, but an L-shaped trace."""
    d0, d1 = q1[0] - q0[0], q1[1] - q0[1]
    quarter = 0.5
    a0, a1 = d0 / quarter**2, d1 / quarter**2  # each joint: up then down over half the piece
    rest = RampInterval(duration=2 * quarter, acceleration=0.0)
    piece = TrajectoryPiece(
        validation=PieceValidation.PLANNED,
        ramps=(
            Ramp(
                position=q0[0],
                velocity=0.0,
                intervals=(
                    RampInterval(duration=quarter, acceleration=a0),
                    RampInterval(duration=quarter, acceleration=-a0),
                    rest,
                ),
            ),
            Ramp(
                position=q0[1],
                velocity=0.0,
                intervals=(
                    rest,
                    RampInterval(duration=quarter, acceleration=a1),
                    RampInterval(duration=quarter, acceleration=-a1),
                ),
            ),
        ),
    )
    peak_v = max(abs(a0), abs(a1)) * quarter
    peak_a = max(abs(a0), abs(a1))
    limits = EffectiveLimits(
        joints=JOINTS,
        velocity=(peak_v, peak_v),
        acceleration=(peak_a, peak_a),
        velocity_scaling=1.0,
        acceleration_scaling=1.0,
        effective_velocity=(peak_v, peak_v),
        effective_acceleration=(peak_a, peak_a),
    )
    return TimedTrajectory(
        group=GROUP,
        joints=JOINTS,
        law=TimingLaw.PIECEWISE_CONSTANT_ACCELERATION,
        pieces=(piece,),
        limits=limits,
        validated=(),
        validation_resolution=1.0,
        producer="l-shaped",
    )


def _path(*waypoints: tuple[float, float]) -> GeometricPath:
    return GeometricPath(group=GROUP, joints=JOINTS, waypoints=waypoints, validation_resolution=1.0)


def _scenario(out: Path) -> dict[str, Any]:
    world = IntervalWorld()
    banded = IntervalWorld(band=0.05)
    clear = _path((0.0, 0.0), (1.0, 0.0), (1.0, 1.0))  # around the box
    clipping = _path((0.0, 0.0), (1.0, 1.0))  # both knots clear, the line crosses the box
    timed = _triangles(GROUP, clear.waypoints, _limits())  # the same path, timed exactly
    # From (0, 0) to (1, 1): the straight line crosses the box, while the L-shaped timing
    # sweeps turn at lift 0, then lift at turn 1, and never enters it.
    l_shaped = _l_shaped((0.0, 0.0), (1.0, 1.0))
    reports = {
        "state_clear": world.validate_state((0.0, 0.0)),
        "state_in_box": world.validate_state((0.5, 0.5)),
        "state_beyond_limit": world.validate_state((0.0, 2.5)),
        "motion_clear": world.validate_motion((0.0, 0.0), (1.0, 0.0)),
        "motion_half_turn": world.validate_motion((0.0, 0.0), (math.pi, 0.0)),
        "motion_grazing_band": banded.validate_motion((0.0, 0.63), (1.0, 0.63)),
        "path_clear": world.validate_path(clear),
        "path_clipping_between_knots": world.validate_path(clipping),
        "trajectory_exact_reparameterization": world.validate_trajectory(timed),
        "trajectory_same_endpoints_l_shaped": world.validate_trajectory(l_shaped),
        "path_same_endpoints_straight": world.validate_path(_path((0.0, 0.0), (1.0, 1.0))),
    }
    encoded = {name: json.loads(dumps(report)) for name, report in reports.items()}
    (out / "reports.json").write_text(json.dumps(encoded, indent=2, sort_keys=True) + "\n")
    return {"reports": reports, "encoded": encoded}


def test_motion_validator_contract_is_planner_neutral(artifacts: Path, tmp_path: Path) -> None:
    """A validator with no planner behind it satisfies the contract: deterministic,
    schema-valid reports bound to the exact subject, with a witness at the first invalid
    state, and the same answer however and however often it is asked."""
    run = _scenario(artifacts)
    _scenario(tmp_path)
    assert (artifacts / "reports.json").read_bytes() == (tmp_path / "reports.json").read_bytes()
    reports: dict[str, MotionValidation] = run["reports"]

    for name, report in reports.items():
        assert loads(dumps(report), MotionValidation) == report, name

    verdicts = {name: r.valid for name, r in reports.items()}
    assert verdicts == {
        "state_clear": True,
        "state_in_box": False,
        "state_beyond_limit": False,
        "motion_clear": True,
        "motion_half_turn": False,
        "motion_grazing_band": False,
        "path_clear": True,
        "path_clipping_between_knots": False,
        "trajectory_exact_reparameterization": True,
        "trajectory_same_endpoints_l_shaped": True,
        "path_same_endpoints_straight": False,
    }
    failures = {name: r.witness.failure for name, r in reports.items() if r.witness is not None}
    assert failures == {
        "state_in_box": ValidationFailure.COLLISION,
        "state_beyond_limit": ValidationFailure.POSITION_LIMITS,
        "motion_half_turn": ValidationFailure.UNSUPPORTED_MOTION,
        "motion_grazing_band": ValidationFailure.AMBIGUOUS,
        "path_clipping_between_knots": ValidationFailure.COLLISION,
        "path_same_endpoints_straight": ValidationFailure.COLLISION,
    }
    clip = reports["path_clipping_between_knots"].witness
    assert clip is not None and clip.segment == 0 and clip.fraction == 0.4  # enters at (0.4, 0.4)

    # An exact reparameterization keeps the path's geometric verdict; geometry-changing
    # timing is its own subject, with its own identity and verdict.
    assert reports["trajectory_exact_reparameterization"].valid == reports["path_clear"].valid
    line = reports["path_same_endpoints_straight"]
    curve = reports["trajectory_same_endpoints_l_shaped"]
    assert line.motion != curve.motion and line.valid != curve.valid

    # Order and repetition change nothing.
    world = IntervalWorld()
    clipping = _path((0.0, 0.0), (1.0, 1.0))
    first = world.validate_path(clipping)
    world.validate_state((0.5, 0.5))
    world.validate_trajectory(_l_shaped((0.0, 0.0), (1.0, 1.0)))
    assert world.validate_path(clipping) == first == reports["path_clipping_between_knots"]


def _refusal(action: Callable[[], object]) -> str:
    try:
        action()
    except SsrobotError as e:
        return e.code
    return "accepted"


def test_malformed_validation_reports_are_refused(artifacts: Path) -> None:
    """A report that claims more than it shows, or contradicts itself, does not decode."""
    good = json.loads(dumps(IntervalWorld().validate_path(_path((0.0, 0.0), (1.0, 1.0)))))

    def edited(edit: Callable[[dict[str, Any]], None]) -> Callable[[], object]:
        data = json.loads(json.dumps(good))
        edit(data)
        return lambda: loads(json.dumps(data), MotionValidation)

    def claim_valid(d: dict[str, Any]) -> None:
        d["valid"] = True

    def drop_witness(d: dict[str, Any]) -> None:
        d["witness"] = None

    def witness_outside(d: dict[str, Any]) -> None:
        d["witness"]["segment"] = 3

    def discrete_without_spacing(d: dict[str, Any]) -> None:
        d["validator"]["policy"]["guarantee"] = "discrete"

    def path_without_intervals(d: dict[str, Any]) -> None:
        d["intervals"] = 0
        d["witness"]["segment"] = None

    def no_checks(d: dict[str, Any]) -> None:
        d["checks"] = 0

    def trajectory_without_law(d: dict[str, Any]) -> None:
        d["subject"] = "trajectory"

    codes = {
        "valid_with_witness": _refusal(edited(claim_valid)),
        "invalid_without_witness": _refusal(edited(drop_witness)),
        "witness_outside_subject": _refusal(edited(witness_outside)),
        "discrete_without_spacing": _refusal(edited(discrete_without_spacing)),
        "path_without_intervals": _refusal(edited(path_without_intervals)),
        "no_checks": _refusal(edited(no_checks)),
        "trajectory_without_law": _refusal(edited(trajectory_without_law)),
    }
    (artifacts / "refusals.json").write_text(json.dumps(codes, indent=2) + "\n")
    assert codes == {
        "valid_with_witness": "inconsistent_validation",
        "invalid_without_witness": "inconsistent_validation",
        "witness_outside_subject": "inconsistent_witness",
        "discrete_without_spacing": "inconsistent_policy",
        "path_without_intervals": "inconsistent_subject",
        "no_checks": "out_of_limits",
        "trajectory_without_law": "inconsistent_subject",
    }


def test_reports_are_bound_to_their_exact_subject_and_validator() -> None:
    """The exact trajectory identity covers q(t) and the limits, not validation
    provenance; and a well-formed report about any other motion, or from any other
    validator, is refused by the one core binding check."""
    timed = _triangles(GROUP, ((0.0, 0.0), (1.0, 0.0), (1.0, 1.0)), _limits())
    retagged = replace(
        timed,
        pieces=tuple(replace(p, validation=PieceValidation.SHORTCUT) for p in timed.pieces),
        validated=(),
        validation_resolution=0.25,
        producer="elsewhere",
        provenance=(("note", "same motion"),),
    )
    assert trajectory_identity(retagged) == trajectory_identity(timed)  # provenance only
    looser = EffectiveLimits(
        joints=JOINTS,
        velocity=(2.0, 2.0),
        acceleration=(4.0, 4.0),
        velocity_scaling=1.0,
        acceleration_scaling=1.0,
        effective_velocity=(2.0, 2.0),
        effective_acceleration=(4.0, 4.0),
    )
    # The same q(t), timed under other limits, is another trajectory.
    assert trajectory_identity(replace(timed, limits=looser)) != trajectory_identity(timed)

    world, banded = IntervalWorld(), IntervalWorld(band=0.05)
    path = _path((0.0, 0.0), (1.0, 1.0))
    report = world.validate_path(path)
    check_validation(
        report, validator=world.identity, subject=ValidationSubject.PATH, motion=path_identity(path)
    )
    other = _path((0.0, 0.0), (1.0, 0.0))
    codes = {
        "other_motion": _refusal(
            lambda: check_validation(
                report,
                validator=world.identity,
                subject=ValidationSubject.PATH,
                motion=path_identity(other),
            )
        ),
        "other_validator": _refusal(
            lambda: check_validation(
                report,
                validator=banded.identity,
                subject=ValidationSubject.PATH,
                motion=path_identity(path),
            )
        ),
        "other_subject_kind": _refusal(
            lambda: check_validation(
                report,
                validator=world.identity,
                subject=ValidationSubject.TRAJECTORY,
                motion=path_identity(path),
            )
        ),
    }
    assert codes == {
        "other_motion": "validation_mismatch",
        "other_validator": "validation_mismatch",
        "other_subject_kind": "validation_mismatch",
    }
    # A continuous guarantee may itself subdivide under a covering bound.
    ValidationPolicy(
        name="adaptive",
        mechanism="adaptive_bounded",
        guarantee=ValidationGuarantee.CONTINUOUS,
        resolution=0.05,
        clearance_m=0.002,
        ambiguity_band_m=0.0005,
        summary="Subdivides until the workspace motion between states is within the clearance.",
    )
