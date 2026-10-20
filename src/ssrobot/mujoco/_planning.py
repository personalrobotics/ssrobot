"""Planning scenes over MuJoCo snapshots, built on sscbirrt's native MuJoCo scene.

Imported only when a planning scene is materialized, so opening a runtime never loads
sscbirrt.
"""

from __future__ import annotations

import threading
from collections import OrderedDict
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import mujoco
import numpy as np
from sscbirrt.backends import native_mujoco

from ssrobot.conventions import Pose
from ssrobot.description import RobotDescription
from ssrobot.errors import CapabilityError, ValidationError
from ssrobot.mujoco._compose import compile_world
from ssrobot.mujoco._model import FrameBinding, MujocoObject
from ssrobot.mujoco._runtime import relative, world_pose
from ssrobot.mujoco._scene import SceneSource
from ssrobot.scene import Contact, ContactKind, SceneSnapshot

_BODY = mujoco.mjtObj.mjOBJ_BODY
_JOINT = mujoco.mjtObj.mjOBJ_JOINT
_OBJECTS = {
    MujocoObject.BODY: mujoco.mjtObj.mjOBJ_BODY,
    MujocoObject.SITE: mujoco.mjtObj.mjOBJ_SITE,
    MujocoObject.CAMERA: mujoco.mjtObj.mjOBJ_CAMERA,
}


def _matrix(position: Any, quat: Any) -> Any:
    """A homogeneous transform from a position and a wxyz quaternion."""
    rotation = np.empty(9)
    mujoco.mju_quat2Mat(rotation, np.asarray(quat, dtype=float))
    transform = np.eye(4)
    transform[:3, :3] = rotation.reshape(3, 3)
    transform[:3, 3] = position
    return transform


def _pose(position: Any, quat: Any) -> Pose:
    x, y, z = (float(v) for v in position)
    w, i, j, k = (float(v) for v in quat)
    return Pose(position=(x, y, z), quat_wxyz=(w, i, j, k))


def planning_model(
    model_path: Path, scene: SceneSource | None, excludes: Sequence[tuple[str, str]]
) -> Any:
    """The runtime's model, rebuilt from the same sources by the same composition, with
    ``excludes`` added: the collision allowances that sscbirrt's native policy would not
    otherwise honour."""
    return compile_world(model_path, scene, excludes=excludes)[0]


PLANNING_MODELS = 16
"""How many compiled planning models a runtime keeps, least recently used dropped first:
one per set of excludes and group, and a run sees few (one per distinct set of held
objects' allowances)."""


class PlanningModels:
    """A runtime's compiled planning models and their native scenes, by excludes and group.

    A planning model depends only on the runtime's sources and its excludes, so a model
    compiled once serves every later snapshot with the same allowances. Each is compiled
    from the runtime's sources with ``check`` (the package is unchanged) before and after,
    as every compile is; a model taken from here is one such compile, read only, shared
    like sscbirrt's native scene. A file changed after it was compiled is noticed only at
    the next compile."""

    def __init__(self, sources: tuple[Path, SceneSource | None], check: Callable[[], None]) -> None:
        self._sources = sources
        self._check = check
        self._models: OrderedDict[tuple[Any, ...], tuple[Any, Any]] = OrderedDict()
        self._lock = threading.Lock()

    def get(
        self, excludes: tuple[tuple[str, str], ...], joints: tuple[str, ...]
    ) -> tuple[Any, Any]:
        """The planning model with ``excludes`` and its native scene over ``joints``."""
        key = (excludes, joints)
        with self._lock:
            found = self._models.get(key)
            if found is not None:
                self._models.move_to_end(key)
                return found
            self._check()
            model = planning_model(self._sources[0], self._sources[1], excludes)
            self._check()
            found = (model, native_mujoco.NativeScene.from_model(model, joints))
            self._models[key] = found
            if len(self._models) > PLANNING_MODELS:
                self._models.popitem(last=False)
            return found


class MujocoPlanningScene:
    """A ``PlanningScene`` over one joint group of a MuJoCo snapshot.

    Validity and contacts come from sscbirrt's native validator, which holds the
    snapshot and moves each held object with its end effector. Forward kinematics uses a
    private ``MjData``. Nothing is shared with the live runtime or with another scene
    except the planning model and sscbirrt's native scene, which nothing writes.
    """

    def __init__(
        self,
        *,
        snapshot: SceneSnapshot,
        group: str,
        joints: tuple[str, ...],
        model: Any,
        data: Any,
        frames: dict[str, FrameBinding],
        addresses: tuple[int, ...],
        names: dict[int, str],
        checker: Any,
    ) -> None:
        self._snapshot = snapshot
        self._group = group
        self._joints = joints
        self._model = model
        self._data = data
        self._frames = frames
        self._addresses = addresses
        self._names = names
        self._checker = checker

    @property
    def group(self) -> str:
        return self._group

    @property
    def joints(self) -> tuple[str, ...]:
        return self._joints

    @property
    def snapshot(self) -> SceneSnapshot:
        return self._snapshot

    def forward_kinematics(self, q: Sequence[float], frame: str) -> Pose:
        for address, value in zip(self._addresses, q, strict=True):
            self._data.qpos[address] = value
        mujoco.mj_kinematics(self._model, self._data)
        position, quat = world_pose(self._data, self._frames[frame])
        quat = np.array(quat, dtype=float)
        mujoco.mju_normalize4(quat)
        return _pose(position, quat)

    def is_valid(self, q: Sequence[float]) -> bool:
        return bool(self._checker.is_valid(list(q)))

    def contacts(self, q: Sequence[float]) -> tuple[Contact, ...]:
        return tuple(
            Contact(
                kind=ContactKind(c.kind),
                first=self._names[c.body1],
                second=self._names[c.body2],
                distance=float(c.dist),
            )
            for c in self._checker.native.invalid_contacts(list(q))
        )

    def native(self) -> object | None:
        """sscbirrt's ``NativeCollisionChecker`` for this scene."""
        checker: object = self._checker
        return checker


def materialize(
    *,
    description: RobotDescription,
    frames: dict[str, FrameBinding],
    runtime_model: Any,
    models: PlanningModels,
    objects: dict[str, int],
    fixtures: dict[str, int],
    snapshot: SceneSnapshot,
    group: str,
) -> MujocoPlanningScene:
    """Build a planning scene for ``group`` from a snapshot the runtime has checked."""
    reason = native_mujoco.unavailable_reason()
    if reason is not None:
        raise CapabilityError("planning_unavailable", f"sscbirrt's native MuJoCo scene: {reason}")

    def body_name(frame: str) -> str:
        name = mujoco.mj_id2name(runtime_model, _BODY, frames[frame].body)
        if name is None:
            raise ValidationError(
                "unsupported_allowance",
                f"frame {frame!r} is carried by an unnamed MuJoCo body",
                path=f"frames[{frame}]",
            )
        return str(name)

    excludes = [
        (body_name(c.frame_a), body_name(c.frame_b))
        for c in description.collision_allowances
        if frames[c.frame_a].body != frames[c.frame_b].body
    ] + [
        # Every allowed name, a scene fixture or a robot frame, by its MuJoCo body: sscbirrt
        # honours a held object's allowance only for bodies the arm moves, so a static
        # robot body such as a work surface needs the exclude too.
        (a.object, name if name in fixtures else body_name(name))
        for a in snapshot.attachments
        for name in a.allow
    ]
    joints = description.group(group).joints
    # one exclude per body pair
    model, scene = models.get(tuple(dict.fromkeys(excludes)), joints)

    def resolve(kind: Any, name: str, path: str) -> int:
        """A name's id in the planning model; a snapshot naming what is not there would
        otherwise index another entity."""
        i = int(mujoco.mj_name2id(model, kind, name))
        if i < 0:
            raise ValidationError(
                "invalid_snapshot", f"the model has nothing named {name!r}", path=path
            )
        return i

    data = mujoco.MjData(model)
    for joint, position in zip(snapshot.joints, snapshot.positions, strict=True):
        data.qpos[model.jnt_qposadr[resolve(_JOINT, joint, "snapshot.joints")]] = position
    for state in snapshot.objects:
        free = int(model.body_jntadr[resolve(_BODY, state.name, "snapshot.objects")])
        at = int(model.jnt_qposadr[free])
        data.qpos[at : at + 3] = state.pose.position
        data.qpos[at + 3 : at + 7] = state.pose.quat_wxyz
    mujoco.mj_kinematics(model, data)

    bound = {}
    for name, binding in frames.items():
        i = mujoco.mj_name2id(model, _OBJECTS[binding.object], name)
        if binding.object is MujocoObject.BODY:
            body = i
        elif binding.object is MujocoObject.SITE:
            body = int(model.site_bodyid[i])
        else:
            body = int(model.cam_bodyid[i])
        bound[name] = FrameBinding(
            name=name,
            object=binding.object,
            id=i,
            body=body,
            collision_geoms=binding.collision_geoms,
        )

    module = native_mujoco._load()
    lowered = []
    for attachment in snapshot.attachments:
        effector = bound[description.end_effector(attachment.end_effector).frame]
        if not scene.native.is_arm_body(effector.body):
            continue  # the group does not move it: the object stays where it is
        p_body, q_body = data.xpos[effector.body], data.xquat[effector.body]
        p_ee, q_ee = world_pose(data, effector)
        body_ee = _matrix(*relative(p_ee, q_ee, p_body, q_body))
        ee_object = _matrix(attachment.transform.position, attachment.transform.quat_wxyz)
        allowed = sorted({bound[f].body for f in attachment.allow if f in bound})
        lowered.append(
            module.Attachment(
                mujoco.mj_name2id(model, _BODY, attachment.object),
                effector.body,
                (body_ee @ ee_object).tolist(),
                allowed,
            )
        )
    try:
        native_snapshot = module.Snapshot(
            scene.native,
            [float(v) for v in data.qpos],
            [float(v) for v in data.mocap_pos.reshape(-1)],
            [float(v) for v in data.mocap_quat.reshape(-1)],
            lowered,
        )
    except ValueError as e:
        raise ValidationError("invalid_snapshot", str(e), path="snapshot") from None
    checker = native_mujoco.NativeCollisionChecker(
        scene, native_mujoco.Snapshot(native_snapshot, scene)
    )

    names = {0: "world"}
    for name, binding in bound.items():
        if binding.object is MujocoObject.BODY:
            names[binding.id] = name
    for name in (*objects, *fixtures):
        names[mujoco.mj_name2id(model, _BODY, name)] = name
    for body in range(model.nbody):  # unnamed or nested bodies: their nearest named ancestor
        ancestor = body
        while ancestor not in names:
            ancestor = int(model.body_parentid[ancestor])
        names[body] = names[ancestor]

    return MujocoPlanningScene(
        snapshot=snapshot,
        group=group,
        joints=joints,
        model=model,
        data=data,
        frames=bound,
        addresses=tuple(
            int(model.jnt_qposadr[mujoco.mj_name2id(model, _JOINT, j)]) for j in joints
        ),
        names=names,
        checker=checker,
    )
