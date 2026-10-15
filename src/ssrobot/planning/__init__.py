"""The planning integration: sscbirrt, ssik, and sstsr behind ssrobot's neutral planning
contracts (#22). Installed with the ``planning`` extra.

``CBiRRT`` plans a ``PlanningRequest`` in a ``PlanningScene`` and returns sscbirrt's path
and its timed trajectory. ``SsikKinematics`` gives it analytic IK for a manipulator from
the package's own model. ``CBiRRT.validator`` is the planner's own validator, an
``SscbirrtValidator``, which ``RobotContext.execute`` and ``ssrobot.prepare`` ask about
every motion. ``RestToRestProcessor`` times a geometric
path without leaving it. See docs/planning.md.
"""

from ssrobot.planning._cbirrt import CBiRRT
from ssrobot.planning._kinematics import SsikKinematics
from ssrobot.planning._processor import RestToRestProcessor
from ssrobot.planning._validator import SscbirrtValidator

__all__ = [
    "CBiRRT",
    "RestToRestProcessor",
    "SscbirrtValidator",
    "SsikKinematics",
]
