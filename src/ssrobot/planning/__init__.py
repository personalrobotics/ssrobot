"""The planning integration: sscbirrt, ssik, and sstsr behind ssrobot's neutral planning
contracts (#22). Installed with the ``planning`` extra.

``CBiRRT`` plans a ``PlanningRequest`` in a ``PlanningScene`` and returns sscbirrt's path
and its timed trajectory. ``SsikKinematics`` gives it analytic IK for a manipulator from
the package's own model. ``prepare`` samples a plan for a runtime and checks the command
independently before it is submitted. ``RestToRestProcessor`` times a geometric path
without leaving it. See docs/planning.md.
"""

from ssrobot.planning._cbirrt import CBiRRT
from ssrobot.planning._kinematics import SsikKinematics
from ssrobot.planning._prepare import GOAL_TOLERANCE, prepare
from ssrobot.planning._processor import RestToRestProcessor

__all__ = ["GOAL_TOLERANCE", "CBiRRT", "RestToRestProcessor", "SsikKinematics", "prepare"]
