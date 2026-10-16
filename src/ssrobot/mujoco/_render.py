"""A read-only view of a MujocoRuntime's state, for application viewers.

``render_state(ctx)`` copies what a viewer needs to draw the current frame:
- the compiled model, copied once per open;
- the current ``MjData``, copied on every call.

It is labelled with the world identity (the model and scene, as snapshots name them),
simulation time, and scene revision it shows. A viewer owns these copies and may change
them freely; nothing it does reaches the runtime, so it cannot change the next
observation or command result. Every state of one open shares the model copy, so a
change to it shows in later states, never in the runtime.
The application decides when to call it, typically after ``step()``. ssrobot creates no
viewer, thread, or synchronization loop.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any

import mujoco

from ssrobot.context import RobotContext
from ssrobot.conventions import Timestamp
from ssrobot.errors import CapabilityError, ValidationError
from ssrobot.mujoco._runtime import MujocoRuntime


@dataclass(frozen=True, slots=True)
class RenderState:
    """One frame of a MujocoRuntime, as copies a viewer owns."""

    model: Any
    """The compiled ``MjModel``: a copy, shared by every state of this runtime's open."""
    data: Any
    """The ``MjData`` at ``time``: a copy made for this state alone."""
    time: Timestamp
    """The runtime time this frame shows."""
    revision: int
    """The scene revision this frame shows."""
    world: str
    """The compiled model and scene, as ``MujocoRuntime.world`` and snapshots name them."""

    def copy_into(self, data: Any) -> None:
        """Load this frame into ``data``, a viewer's ``MjData`` of ``model``."""
        mujoco.mj_copyData(data, self.model, self.data)


def render_state(ctx: RobotContext) -> RenderState:
    """The current frame of ``ctx``'s MujocoRuntime. Fails with ``render_unavailable``
    for any other runtime, and with ``not_open`` before the context opens."""
    runtime = getattr(ctx, "_runtime", None)
    if not isinstance(runtime, MujocoRuntime):
        raise CapabilityError(
            "render_unavailable", "only a MujocoRuntime's context has a render state"
        )
    now, revision = ctx.now, ctx.scene.revision
    live_model, live_data = runtime._model, runtime._data
    if live_model is None or live_data is None:
        raise ValidationError("not_open", "the runtime is not open")
    if runtime._render_model is None:
        runtime._render_model = copy.deepcopy(live_model)
    model = runtime._render_model
    data = mujoco.MjData(model)
    mujoco.mj_copyData(data, live_model, live_data)
    mujoco.mj_forward(model, data)
    return RenderState(model=model, data=data, time=now, revision=revision, world=runtime.world())
