"""The simulated world: which package model and profile a MuJoCo world is built from,
and how that model is composed with a scene and compiled. ``MujocoRuntime``, its
planning scenes, and ``materialize_world`` all build a world here, so they build the
same one."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import mujoco

from ssrobot.errors import CapabilityError, ValidationError
from ssrobot.mujoco._model import (
    MUJOCO_VERSION,
    MujocoProfile,
    load_profile,
    select_model,
    select_profile,
    versions,
)
from ssrobot.mujoco._scene import SceneSource
from ssrobot.package import ModelEntry, ProfileEntry, RobotPackage

_FREE = mujoco.mjtJoint.mjJNT_FREE


def require_version() -> None:
    """Fail unless the installed, loaded, and compiled MuJoCo are the supported version."""
    found = versions()
    if set(found) != {MUJOCO_VERSION}:
        installed, loaded, compiled = found
        raise CapabilityError(
            "unsupported_backend_version",
            f"MuJoCo {MUJOCO_VERSION} is required; installed {installed}, "
            f"loaded {loaded}, compiled {compiled}",
        )


def select(
    package: RobotPackage, profile: str | None
) -> tuple[ProfileEntry | None, MujocoProfile, ModelEntry]:
    """The MuJoCo profile entry (``profile``, or the package's only one), the profile it
    loads, and the model it compiles."""
    entry = select_profile(package, profile)
    loaded = MujocoProfile() if entry is None else load_profile(package, entry)
    return entry, loaded, select_model(package, loaded, entry)


def compile_world(
    model_path: Path,
    scene: SceneSource | None,
    *,
    excludes: Sequence[tuple[str, str]] = (),
    path: str = "model",
) -> tuple[Any, dict[str, str]]:
    """Compile the robot model at ``model_path`` with ``scene``'s bodies, from the
    scene's bytes read once, and with a contact exclude for each body pair in
    ``excludes``. Return the model and each top-level scene body's kind (``object``, one
    free joint; ``fixture``, no joints). ``path`` names the model in errors."""
    try:
        spec = mujoco.MjSpec.from_file(str(model_path))
    except ValueError as e:
        raise ValidationError("model_compile_failed", str(e), path=path) from None
    kinds: dict[str, str] = {}
    if scene is not None:
        try:
            child = scene.spec()
        except (ValueError, UnicodeDecodeError) as e:
            raise ValidationError("invalid_scene", str(e), path="scene") from None
        for body in child.worldbody.bodies:
            if not body.name:
                raise ValidationError(
                    "invalid_scene", "every top-level scene body needs a name", path="scene"
                )
            joints = [j.type for j in body.joints]
            if joints == [_FREE]:
                kinds[body.name] = "object"
            elif not joints:
                kinds[body.name] = "fixture"
            else:
                raise ValidationError(
                    "invalid_scene",
                    f"{body.name!r} must have one free joint (an object) or none (a fixture)",
                    path=f"scene: {body.name}",
                )
        spec.attach(child, frame=spec.worldbody.add_frame(), prefix="")
    for first, second in excludes:
        exclude = spec.add_exclude()
        exclude.bodyname1, exclude.bodyname2 = first, second
    try:
        model = spec.compile()
    except ValueError as e:
        message = str(e).strip()
        if scene is None:
            raise ValidationError("model_compile_failed", message, path=path) from None
        code = "scene_conflict" if "repeated name" in message else "invalid_scene"
        raise ValidationError(code, message, path="scene") from None
    return model, kinds
