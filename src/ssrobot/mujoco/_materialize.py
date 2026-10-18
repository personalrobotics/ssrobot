"""A recorded MuJoCo world, built and checked without a runtime.

``materialize_world`` compiles the world a run bundle recorded, and
``materialize_world_from`` the world of a package and scene, through the same
composition ``MujocoRuntime`` uses. Each recomputes the world identity from the files it
compiled and refuses (``world_mismatch``) a world other than the one expected before
returning anything. What it returns is the caller's own: a new ``MjModel``, and a
mapping that puts a ``SceneSnapshot`` into an ``MjData``. Nothing else is created: no
context, clock, controller, actuator binding, or renderer, and physics never steps.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mujoco

from ssrobot._world import model_signature, world_identity
from ssrobot.bundle import Bundle
from ssrobot.errors import CapabilityError, ValidationError
from ssrobot.mujoco._compose import compile_world, require_version, select
from ssrobot.mujoco._model import unchanged
from ssrobot.mujoco._scene import load_scene
from ssrobot.package import ModelEntry, ModelFormat, RobotPackage
from ssrobot.scene import SceneSnapshot

_BODY = mujoco.mjtObj.mjOBJ_BODY
_JOINT = mujoco.mjtObj.mjOBJ_JOINT


@dataclass(frozen=True)
class SnapshotMapping:
    """Where a world's snapshots go in its ``qpos``: each description joint, in order,
    and each scene object's free joint."""

    description: str
    world: str
    joints: tuple[tuple[str, int], ...]
    """Each description joint and its qpos address."""
    objects: tuple[tuple[str, int], ...]
    """Each scene object, sorted, and its free joint's qpos address."""

    def apply(self, snapshot: SceneSnapshot, data: Any) -> None:
        """Write ``snapshot``'s joint positions and object poses into ``data.qpos``.
        Nothing else changes: run ``mujoco.mj_kinematics`` (or ``mj_forward``) for the
        body poses. Fails with ``incompatible_snapshot`` for a snapshot of another
        description or world, or of other joints or objects."""
        if snapshot.description != self.description or snapshot.model != self.world:
            raise ValidationError(
                "incompatible_snapshot",
                "the snapshot is of another description or world",
                path="snapshot.model",
            )
        if snapshot.joints != tuple(name for name, _ in self.joints):
            raise ValidationError(
                "incompatible_snapshot", "the snapshot has other joints", path="snapshot.joints"
            )
        objects = {o.name: o.pose for o in snapshot.objects}
        if sorted(objects) != [name for name, _ in self.objects]:
            raise ValidationError(
                "incompatible_snapshot", "the snapshot has other objects", path="snapshot.objects"
            )
        for (_, at), position in zip(self.joints, snapshot.positions, strict=True):
            data.qpos[at] = position
        for name, at in self.objects:
            pose = objects[name]
            data.qpos[at : at + 3] = pose.position
            data.qpos[at + 3 : at + 7] = pose.quat_wxyz


@dataclass(frozen=True)
class MaterializedWorld:
    """A world's compiled model, owned by the caller, its identity, and its snapshot
    mapping."""

    model: Any
    world: str
    description: str
    mapping: SnapshotMapping

    def data(self) -> Any:
        """A new ``MjData`` for the model."""
        return mujoco.MjData(self.model)


def materialize_world(bundle: Bundle) -> MaterializedWorld:
    """The world ``bundle`` recorded, compiled from its bundled files, which must still
    be the world its manifest names."""
    entry = _model(bundle.package, bundle.model)
    return _build(bundle.package, entry, bundle.scene, bundle.scene_root, bundle.manifest.world)


def materialize_world_from(
    package: RobotPackage,
    *,
    world: str,
    scene: str | os.PathLike[str] | None = None,
    scene_root: str | os.PathLike[str] | None = None,
    profile: str | None = None,
) -> MaterializedWorld:
    """The world of ``package`` (its model chosen by ``profile`` as ``MujocoRuntime``
    chooses it) and ``scene``, which must be ``world``: the identity a runtime's
    ``world()`` and its snapshots name."""
    if scene_root is not None and scene is None:
        raise ValidationError("invalid_argument", "scene_root needs a scene", path="scene_root")
    _, _, entry = select(package, profile)
    return _build(
        package,
        entry,
        None if scene is None else Path(scene),
        None if scene_root is None else Path(scene_root),
        world,
    )


def _model(package: RobotPackage, name: str) -> ModelEntry:
    entry = next((m for m in package.manifest.models if m.name == name), None)
    if entry is None:
        raise ValidationError("unknown_reference", f"the package has no model {name!r}")
    if entry.format is not ModelFormat.MJCF:
        raise CapabilityError(
            "unsupported_model_format", f"MuJoCo compiles MJCF models, not {entry.format.value}"
        )
    return entry


def _build(
    package: RobotPackage,
    entry: ModelEntry,
    scene: Path | None,
    scene_root: Path | None,
    expected: str,
) -> MaterializedWorld:
    require_version()
    # As the runtime does: compile only what was loaded and hashed, checked before and
    # after; the scene is compiled from the bytes its identity is computed over.
    unchanged(package)
    source = None if scene is None else load_scene(scene, scene_root)
    model, kinds = compile_world(package.root / entry.path, source, path=entry.path)
    unchanged(package)
    world = world_identity(
        model_signature(package, entry.name), None if source is None else source.identity
    )
    if world != expected:
        raise ValidationError(
            "world_mismatch",
            f"the files compiled are world {world!r}, not {expected!r}",
            path="world",
        )

    def address(kind: Any, name: str, path: str) -> int:
        i = int(mujoco.mj_name2id(model, kind, name))
        if i < 0:
            raise ValidationError("model_mismatch", f"the model has no {name!r}", path=path)
        return i

    joints = tuple(
        (j.name, int(model.jnt_qposadr[address(_JOINT, j.name, "joints")]))
        for j in package.description.joints
    )
    objects = tuple(
        (name, int(model.jnt_qposadr[int(model.body_jntadr[address(_BODY, name, "objects")])]))
        for name, kind in sorted(kinds.items())
        if kind == "object"
    )
    description = package.description.fingerprint()
    return MaterializedWorld(
        model=model,
        world=world,
        description=description,
        mapping=SnapshotMapping(
            description=description, world=world, joints=joints, objects=objects
        ),
    )
