"""Expected operational and input failures.

Every error carries a stable ``code``. Callers and tests branch on the code; the
message is for people and may change.
"""

from __future__ import annotations


class SsrobotError(Exception):
    """Base class for expected ssrobot failures."""

    def __init__(self, code: str, message: str, *, path: str = "") -> None:
        self.code = code
        self.message = message
        self.path = path
        super().__init__(f"{code}: {path + ': ' if path else ''}{message}")


class ValidationError(SsrobotError):
    """Malformed, ambiguous, or inconsistent input rejected at ingress."""


class CapabilityError(SsrobotError):
    """A request needs a capability that is not declared or not available."""


class StaleRevisionError(SsrobotError):
    """A value refers to a robot description or scene revision that no longer applies."""


class LifecycleError(SsrobotError):
    """An operation is not allowed in the current lifecycle state."""
