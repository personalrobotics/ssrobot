"""Read-only views of a MujocoRuntime's state, for application viewers.

``render_handle(ctx)`` gives one viewer a ``RenderHandle``: its own copy of the compiled
model and an ``MjData`` of that copy. ``capture()`` copies the runtime's current state
into that data and returns the ``RenderFrame`` it now shows: the simulation time, scene
revision, and world identity (the model and scene, as snapshots name them).

Each handle owns its copies, so viewers cannot affect one another or the runtime. A
viewer may change anything it holds, and the next observation and command result are the
same. The application decides when to capture, typically after ``step()``. ssrobot
creates no viewer, thread, or synchronization loop.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any

import mujoco

from ssrobot.context import RobotContext
from ssrobot.conventions import Timestamp
from ssrobot.errors import CapabilityError, LifecycleError
from ssrobot.mujoco._runtime import MujocoRuntime


@dataclass(frozen=True, slots=True)
class RenderFrame:
    """What a handle's data shows after a capture."""

    time: Timestamp
    """The runtime time captured."""
    revision: int
    """The scene revision captured."""
    world: str
    """The compiled model and scene, as ``MujocoRuntime.world`` and snapshots name them."""


class RenderHandle:
    """One viewer's copies of a MujocoRuntime's model and state."""

    __slots__ = ("_ctx", "_runtime", "_source", "data", "model", "world")

    def __init__(self, ctx: RobotContext, runtime: MujocoRuntime) -> None:
        live = runtime._model
        if live is None:
            raise LifecycleError("not_open", "the runtime is not open")
        self._ctx = ctx
        self._runtime = runtime
        self._source = live  # the compiled model this handle's copy describes
        self.model: Any = copy.deepcopy(live)
        """This handle's own ``MjModel``: a copy of the compiled model."""
        self.data: Any = mujoco.MjData(self.model)
        """This handle's own ``MjData`` of ``model``: the last capture."""
        self.world: str = runtime.world()
        """The compiled model and scene this handle's model was copied from."""

    def capture(self) -> RenderFrame:
        """Copy the runtime's current state into ``data``. Fails with ``stale_render``
        if the runtime has reopened since this handle was made, since its model would no
        longer describe the runtime; make a new handle."""
        runtime = self._runtime
        live_model, live_data = runtime._model, runtime._data
        if live_model is None or live_data is None:
            raise LifecycleError("not_open", "the runtime is not open")
        if live_model is not self._source:
            raise CapabilityError("stale_render", "the runtime reopened; make a new render handle")
        now, revision = self._ctx.now, self._ctx.scene.revision
        mujoco.mj_copyData(self.data, live_model, live_data)
        mujoco.mj_forward(self.model, self.data)
        return RenderFrame(time=now, revision=revision, world=self.world)


def render_handle(ctx: RobotContext) -> RenderHandle:
    """A new viewer's handle on ``ctx``'s MujocoRuntime, with nothing captured yet. Fails
    with ``render_unavailable`` for any other runtime, and ``not_open`` before it opens."""
    runtime = getattr(ctx, "_runtime", None)
    if not isinstance(runtime, MujocoRuntime):
        raise CapabilityError(
            "render_unavailable", "only a MujocoRuntime's context can be rendered"
        )
    return RenderHandle(ctx, runtime)
