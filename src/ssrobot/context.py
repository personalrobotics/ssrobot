"""``RobotContext``: the public session through which users observe and command a robot."""

from __future__ import annotations

import enum
from types import TracebackType

from ssrobot.commands import Command
from ssrobot.conventions import ClockMode
from ssrobot.description import RobotDescription
from ssrobot.errors import CapabilityError, LifecycleError, StaleRevisionError
from ssrobot.execution import ExecutionStatus
from ssrobot.observations import Observation, ObservationRequest
from ssrobot.runtime import Runtime, RuntimeInfo
from ssrobot.validation import check_command, check_observation, check_request


class _State(enum.Enum):
    CREATED = "created"
    OPEN = "open"
    CLOSED = "closed"


class RobotContext:
    """Binds one immutable description to one injected runtime for one session.

    Use as a context manager. Entering opens the runtime; leaving closes it, including
    after a failed or partial open. A context is opened at most once. Every command and
    request is validated before it reaches the runtime, and every observation the
    runtime returns is validated before it reaches the caller.
    """

    def __init__(self, description: RobotDescription, runtime: Runtime) -> None:
        self._description = description
        self._fingerprint = description.fingerprint()
        self._runtime = runtime
        self._info: RuntimeInfo | None = None
        self._state = _State.CREATED

    @property
    def description(self) -> RobotDescription:
        return self._description

    @property
    def info(self) -> RuntimeInfo:
        """What the runtime reported when it opened."""
        return self._open_info()

    def __enter__(self) -> RobotContext:
        if self._state is not _State.CREATED:
            raise LifecycleError("already_opened", "a context can be opened only once")
        try:
            info = self._runtime.open(self._description)
            self._check_info(info)
        except BaseException as error:
            self._state = _State.CLOSED
            try:
                self._runtime.close()
            except Exception as close_error:
                error.add_note(f"closing the runtime also failed: {close_error!r}")
            raise
        self._info = info
        self._state = _State.OPEN
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()

    def close(self) -> None:
        """Close the runtime. Idempotent."""
        if self._state is _State.CLOSED:
            return
        self._state = _State.CLOSED
        self._runtime.close()

    def observe(self, request: ObservationRequest) -> Observation:
        info = self._open_info()
        check_request(self._description, request, info)
        observation = self._runtime.observe(request)
        check_observation(self._description, request, observation, info)
        return observation

    def submit(self, command: Command) -> ExecutionStatus:
        """Validate and hand a command to the runtime. Does not advance time."""
        info = self._open_info()
        check_command(self._description, command, info)
        return self._runtime.submit(command)

    def status(self, execution: str) -> ExecutionStatus:
        self._open_info()
        return self._runtime.status(execution)

    def cancel(self, execution: str) -> ExecutionStatus:
        self._open_info()
        return self._runtime.cancel(execution)

    def step(self) -> None:
        """Advance one control tick of a manually clocked runtime."""
        if self._open_info().clock_mode is not ClockMode.MANUAL:
            raise CapabilityError("manual_clock_required", "this runtime is externally clocked")
        self._runtime.step()

    def _open_info(self) -> RuntimeInfo:
        if self._state is not _State.OPEN or self._info is None:
            raise LifecycleError("not_open", f"context is {self._state.value}")
        return self._info

    def _check_info(self, info: RuntimeInfo) -> None:
        if info.description != self._fingerprint:
            raise StaleRevisionError(
                "stale_description",
                f"runtime bound description {info.description[:12]}, "
                f"context has {self._fingerprint[:12]}",
            )
        for cap in info.commands:
            if cap not in self._description.commands:
                raise CapabilityError(
                    "undeclared_capability",
                    f"runtime offers undeclared {cap.kind} on {cap.component!r}",
                )
        declared = {c.name for c in self._description.channels}
        for name in info.channels:
            if name not in declared:
                raise CapabilityError(
                    "undeclared_capability", f"runtime offers undeclared channel {name!r}"
                )
