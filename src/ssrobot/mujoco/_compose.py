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
    collidable_only: bool = False,
) -> tuple[Any, dict[str, str]]:
    """Compile the robot model at ``model_path`` with ``scene``'s bodies, from the
    scene's bytes read once, and with a contact exclude for each body pair in
    ``excludes``. Return the model and each top-level scene body's kind (``object``, one
    free joint; ``fixture``, no joints). ``path`` names the model in errors.
    ``collidable_only`` leaves out the geometry no contact can involve
    (``_drop_uncollidable``): a model for kinematics and contacts, not for dynamics or
    rendering."""
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
    if collidable_only:
        _drop_uncollidable(spec)
    try:
        model = spec.compile()
    except ValueError as e:
        message = str(e).strip()
        if scene is None:
            raise ValidationError("model_compile_failed", message, path=path) from None
        code = "scene_conflict" if "repeated name" in message else "invalid_scene"
        raise ValidationError(code, message, path="scene") from None
    return model, kinds


def _drop_uncollidable(spec: Any) -> None:
    """Delete from ``spec`` what no contact can involve: every geom whose ``contype`` and
    ``conaffinity`` are both 0 and that no explicit ``<pair>`` names, then every mesh,
    material and texture nothing left uses. Bodies, joints, sites and cameras stay, so
    every frame and every contact is the same; a body's mass may change.

    MuJoCo refuses a moving body whose weld group (it and its jointless descendants)
    has no mass, so a group without explicit inertia and without a collidable geom
    keeps its geoms."""
    paired = {name for p in spec.pairs for name in (p.geomname1, p.geomname2) if name}
    explicit = spec.compiler.inertiafromgeom != mujoco.mjtInertiaFromGeom.mjINERTIAFROMGEOM_TRUE

    def uncollidable(geom: Any) -> bool:
        return geom.contype == 0 and geom.conaffinity == 0 and geom.name not in paired

    def welded(body: Any) -> list[Any]:
        return [body, *(b for c in body.bodies if not c.joints for b in welded(c))]

    def massive(group: list[Any]) -> bool:
        return any(
            (explicit and b.explicitinertial) or not all(uncollidable(g) for g in b.geoms)
            for b in group
        )

    dropped: list[Any] = []

    def walk(body: Any, keep: bool) -> None:
        if not keep:
            dropped.extend(g for g in body.geoms if uncollidable(g))
        for child in body.bodies:
            walk(child, not massive(welded(child)) if child.joints else keep)

    walk(spec.worldbody, False)
    for geom in dropped:
        spec.delete(geom)
    meshes = {g.meshname for g in spec.geoms if g.meshname}
    for mesh in [m for m in spec.meshes if m.name not in meshes]:
        spec.delete(mesh)
    owners = [*spec.geoms, *spec.sites, *spec.tendons, *spec.skins, *spec.flexes]
    materials = {o.material for o in owners if o.material}
    for material in [m for m in spec.materials if m.name not in materials]:
        spec.delete(material)
    textures = {t for m in spec.materials for t in m.textures if t}
    for texture in [t for t in spec.textures if t.name not in textures]:
        spec.delete(texture)
