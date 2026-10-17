"""A MuJoCo scene compiled from its closure's bytes; see ``ssrobot._world`` for how the
closure is read and identified."""

from __future__ import annotations

from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

import mujoco

from ssrobot._world import SceneClosure, SceneFile, read_scene

__all__ = ["SceneFile", "SceneSource", "load_scene"]


@dataclass(frozen=True, slots=True)
class SceneSource(SceneClosure):
    """A scene's closure, compilable."""

    def spec(self) -> Any:
        """The scene's ``MjSpec``, compiled from these bytes and nothing on disk."""
        return mujoco.MjSpec.from_string(
            self.xml.decode(), include=dict(self.includes), assets=dict(self.assets)
        )


def load_scene(scene: Path, root: Path | None = None) -> SceneSource:
    """``read_scene``, compilable: fails with ``scene_escape`` for a file outside the root
    and ``invalid_scene`` for anything unreadable or unresolvable, before anything
    compiles."""
    closure = read_scene(scene, root)
    return SceneSource(**{f.name: getattr(closure, f.name) for f in fields(closure)})
