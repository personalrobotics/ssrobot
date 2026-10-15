"""The path-preserving trajectory processor."""

from __future__ import annotations

import math

from ssrobot.errors import ValidationError
from ssrobot.planning_contracts import (
    ConstraintClass,
    EffectiveLimits,
    GeometricPath,
    PieceValidation,
    Ramp,
    RampInterval,
    TimedTrajectory,
    TimingLaw,
    TrajectoryPiece,
)


class RestToRestProcessor:
    """Times a geometric path along its own lines, coming to rest at every waypoint.

    Each segment follows one synchronized trapezoid (or triangle) in its path
    parameter, as fast as the most limiting joint allows. The motion never leaves the
    path, so its collision and path-constraint validity are the path's; it validates
    only the velocity and acceleration limits. Zero-length segments are skipped.
    """

    name = "rest_to_rest"
    constraints = (ConstraintClass.ACCELERATION_LIMITS, ConstraintClass.VELOCITY_LIMITS)

    def process(self, path: GeometricPath, limits: EffectiveLimits) -> TimedTrajectory:
        if limits.joints != path.joints:
            raise ValidationError("joint_mismatch", "limits must cover the path's joints")
        vmax, amax = limits.effective_velocity, limits.effective_acceleration
        pieces = []
        for q0, q1 in zip(path.waypoints, path.waypoints[1:], strict=False):
            delta = [b - a for a, b in zip(q0, q1, strict=True)]
            moving = [k for k, d in enumerate(delta) if d != 0.0]
            if not moving:
                continue
            # The path parameter s runs 0 to 1; joint k moves delta[k] * s.
            speed = min(vmax[k] / abs(delta[k]) for k in moving)
            accel = min(amax[k] / abs(delta[k]) for k in moving)
            if speed * speed / accel >= 1.0:  # never reaches top speed: a triangle
                t_ramp, t_cruise = math.sqrt(1.0 / accel), 0.0
            else:
                t_ramp, t_cruise = speed / accel, (1.0 - speed * speed / accel) / speed
            pieces.append(
                TrajectoryPiece(
                    validation=PieceValidation.PLANNED,
                    ramps=tuple(
                        Ramp(
                            position=a,
                            velocity=0.0,
                            intervals=(
                                RampInterval(duration=t_ramp, acceleration=d * accel),
                                RampInterval(duration=t_cruise, acceleration=0.0),
                                RampInterval(duration=t_ramp, acceleration=-d * accel),
                            ),
                        )
                        for a, d in zip(q0, delta, strict=True)
                    ),
                )
            )
        if not pieces:
            raise ValidationError("empty_path", "the path does not move")
        return TimedTrajectory(
            group=path.group,
            joints=path.joints,
            law=TimingLaw.PIECEWISE_CONSTANT_ACCELERATION,
            pieces=tuple(pieces),
            limits=limits,
            validated=tuple(sorted(self.constraints)),
            producer=self.name,
        )
