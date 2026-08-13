"""Hardware-independent planar-stage interfaces and reference control logic.

All positions are in metres, velocities in metres per second, accelerations in
metres per second squared, and timestamps in seconds.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Protocol, Sequence, runtime_checkable

import numpy as np


class StageError(ValueError):
    """Raised for invalid stage data or unsafe stage commands."""


def _vector2(value: Sequence[float] | np.ndarray, name: str) -> np.ndarray:
    array = np.array(value, dtype=np.float64, copy=True)
    if array.shape != (2,):
        raise StageError(f"{name} must have shape (2,), got {array.shape}")
    if not np.all(np.isfinite(array)):
        raise StageError(f"{name} must contain only finite values")
    array.setflags(write=False)
    return array


def _positive_vector2(
    value: float | Sequence[float] | np.ndarray,
    name: str,
) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.shape == ():
        array = np.full(2, float(array), dtype=np.float64)
    return _vector2(array, name)


@dataclass(frozen=True)
class StageLimits:
    """Planar travel and motion limits."""

    minimum_position_m: np.ndarray
    maximum_position_m: np.ndarray
    maximum_speed_m_s: np.ndarray
    maximum_acceleration_m_s2: np.ndarray

    def __post_init__(self) -> None:
        minimum = _vector2(self.minimum_position_m, "minimum_position_m")
        maximum = _vector2(self.maximum_position_m, "maximum_position_m")
        speed = _positive_vector2(self.maximum_speed_m_s, "maximum_speed_m_s")
        acceleration = _positive_vector2(
            self.maximum_acceleration_m_s2,
            "maximum_acceleration_m_s2",
        )
        if np.any(maximum <= minimum):
            raise StageError(
                "maximum_position_m must exceed minimum_position_m on both axes"
            )
        if np.any(speed <= 0.0):
            raise StageError("maximum_speed_m_s must be positive on both axes")
        if np.any(acceleration <= 0.0):
            raise StageError(
                "maximum_acceleration_m_s2 must be positive on both axes"
            )
        object.__setattr__(self, "minimum_position_m", minimum)
        object.__setattr__(self, "maximum_position_m", maximum)
        object.__setattr__(self, "maximum_speed_m_s", speed)
        object.__setattr__(self, "maximum_acceleration_m_s2", acceleration)

    @staticmethod
    def _copy_vector(value: np.ndarray) -> np.ndarray:
        copied = np.array(value, copy=True)
        copied.setflags(write=False)
        return copied

    def __getattribute__(self, name: str):
        # Frozen dataclasses do not make owning NumPy arrays immutable: callers
        # can re-enable their write flag.  Return copies for the public limit
        # vectors so external code cannot silently change the safety envelope.
        value = object.__getattribute__(self, name)
        if name in {
            "minimum_position_m",
            "maximum_position_m",
            "maximum_speed_m_s",
            "maximum_acceleration_m_s2",
        }:
            return StageLimits._copy_vector(value)
        return value

    def contains(self, position_m: Sequence[float] | np.ndarray) -> bool:
        position = _vector2(position_m, "position_m")
        return bool(
            np.all(position >= self.minimum_position_m)
            and np.all(position <= self.maximum_position_m)
        )

    def clip(self, position_m: Sequence[float] | np.ndarray) -> np.ndarray:
        position = _vector2(position_m, "position_m")
        clipped = np.clip(
            position,
            self.minimum_position_m,
            self.maximum_position_m,
        )
        clipped.setflags(write=False)
        return clipped


@dataclass(frozen=True)
class StageState:
    """One timestamped planar-stage state sample."""

    position_m: np.ndarray
    velocity_m_s: np.ndarray
    timestamp_s: float
    homed: bool
    fault: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "position_m", _vector2(self.position_m, "position_m"))
        object.__setattr__(
            self,
            "velocity_m_s",
            _vector2(self.velocity_m_s, "velocity_m_s"),
        )
        timestamp = float(self.timestamp_s)
        if not np.isfinite(timestamp) or timestamp < 0.0:
            raise StageError("timestamp_s must be finite and non-negative")
        object.__setattr__(self, "timestamp_s", timestamp)
        if not isinstance(self.homed, (bool, np.bool_)):
            raise StageError("homed must be a boolean")
        object.__setattr__(self, "homed", bool(self.homed))
        if self.fault is not None:
            if not isinstance(self.fault, str) or not self.fault.strip():
                raise StageError("fault must be None or a non-empty string")
            object.__setattr__(self, "fault", self.fault.strip())

    @property
    def moving(self) -> bool:
        return bool(np.linalg.norm(self.velocity_m_s) > 0.0)


@runtime_checkable
class XYStage(Protocol):
    """Minimal interface required from a physical or simulated XY stage."""

    @property
    def limits(self) -> StageLimits:
        """Return the configured travel and motion limits."""

    def read_state(self) -> StageState:
        """Return the latest timestamped state."""

    def home(self) -> None:
        """Home the stage and establish its coordinate frame."""

    def command_position(
        self,
        position_m: Sequence[float] | np.ndarray,
        *,
        maximum_speed_m_s: float | Sequence[float] | np.ndarray | None = None,
    ) -> None:
        """Command an absolute planar position."""

    def stop(self) -> None:
        """Stop planar motion."""


class SimulatedXYStage:
    """Deterministic logic simulation with explicit time advancement.

    The reference plant applies nominal speed and acceleration limits while
    tracking a target, but ``stop()`` is an instantaneous logical stop.  It is
    not a safety-rated dynamic model for selecting real stage hardware.
    """

    def __init__(
        self,
        limits: StageLimits,
        *,
        initial_position_m: Sequence[float] | np.ndarray = (0.0, 0.0),
        home_position_m: Sequence[float] | np.ndarray = (0.0, 0.0),
        initially_homed: bool = False,
    ) -> None:
        if not isinstance(limits, StageLimits):
            raise StageError("limits must be a StageLimits instance")
        initial = _vector2(initial_position_m, "initial_position_m")
        home = _vector2(home_position_m, "home_position_m")
        if not limits.contains(initial):
            raise StageError("initial_position_m is outside the stage travel limits")
        if not limits.contains(home):
            raise StageError("home_position_m is outside the stage travel limits")
        if not isinstance(initially_homed, bool):
            raise StageError("initially_homed must be a boolean")

        self._limits = limits
        self._position_m = np.array(initial, copy=True)
        self._velocity_m_s = np.zeros(2, dtype=np.float64)
        self._target_position_m = np.array(initial, copy=True)
        self._home_position_m = home
        self._command_speed_m_s = np.array(limits.maximum_speed_m_s, copy=True)
        self._timestamp_s = 0.0
        self._homed = initially_homed
        self._fault: str | None = None

    @property
    def limits(self) -> StageLimits:
        return self._limits

    @property
    def target_position_m(self) -> np.ndarray:
        target = np.array(self._target_position_m, copy=True)
        target.setflags(write=False)
        return target

    def read_state(self) -> StageState:
        return StageState(
            position_m=self._position_m,
            velocity_m_s=self._velocity_m_s,
            timestamp_s=self._timestamp_s,
            homed=self._homed,
            fault=self._fault,
        )

    def home(self) -> None:
        if self._fault is not None:
            raise StageError(f"cannot home stage while faulted: {self._fault}")
        self._position_m = np.array(self._home_position_m, copy=True)
        self._target_position_m = np.array(self._home_position_m, copy=True)
        self._velocity_m_s.fill(0.0)
        self._homed = True

    def command_position(
        self,
        position_m: Sequence[float] | np.ndarray,
        *,
        maximum_speed_m_s: float | Sequence[float] | np.ndarray | None = None,
    ) -> None:
        target = _vector2(position_m, "position_m")
        if self._fault is not None:
            raise StageError(f"cannot move stage while faulted: {self._fault}")
        if not self._homed:
            raise StageError("stage must be homed before accepting motion commands")
        if not self._limits.contains(target):
            raise StageError("position_m is outside the stage travel limits")

        if maximum_speed_m_s is None:
            speed = self._limits.maximum_speed_m_s
        else:
            speed = _positive_vector2(maximum_speed_m_s, "maximum_speed_m_s")
            if np.any(speed <= 0.0):
                raise StageError("maximum_speed_m_s must be positive")
            if np.any(speed > self._limits.maximum_speed_m_s):
                raise StageError(
                    "maximum_speed_m_s exceeds the configured stage limit"
                )
        self._command_speed_m_s = np.array(speed, copy=True)
        self._target_position_m = np.array(target, copy=True)

    def stop(self) -> None:
        self._target_position_m = np.array(self._position_m, copy=True)
        self._velocity_m_s.fill(0.0)

    def set_fault(self, message: str) -> None:
        if not isinstance(message, str) or not message.strip():
            raise StageError("fault message must be a non-empty string")
        self.stop()
        self._fault = message.strip()

    def clear_fault(self) -> None:
        self._fault = None
        self._homed = False

    def advance(self, duration_s: float) -> StageState:
        """Advance the simulated plant by ``duration_s`` seconds."""

        duration = float(duration_s)
        if not np.isfinite(duration) or duration <= 0.0:
            raise StageError("duration_s must be finite and positive")
        if self._fault is not None:
            self._timestamp_s += duration
            return self.read_state()

        error = self._target_position_m - self._position_m
        distance = float(np.linalg.norm(error))
        if distance <= 16.0 * np.finfo(np.float64).eps:
            self._position_m = np.array(self._target_position_m, copy=True)
            self._velocity_m_s.fill(0.0)
            self._timestamp_s += duration
            return self.read_state()

        direction = error / distance
        speed_limit = float(
            np.min(
                self._command_speed_m_s[np.abs(direction) > 1e-15]
                / np.abs(direction[np.abs(direction) > 1e-15])
            )
        )
        acceleration_limit = float(
            np.min(
                self._limits.maximum_acceleration_m_s2[
                    np.abs(direction) > 1e-15
                ]
                / np.abs(direction[np.abs(direction) > 1e-15])
            )
        )
        current_speed = max(0.0, float(np.dot(self._velocity_m_s, direction)))
        braking_speed = np.sqrt(max(0.0, 2.0 * acceleration_limit * distance))
        desired_speed = min(speed_limit, braking_speed)
        speed_change = np.clip(
            desired_speed - current_speed,
            -acceleration_limit * duration,
            acceleration_limit * duration,
        )
        next_speed = max(0.0, current_speed + float(speed_change))
        displacement = next_speed * duration
        if displacement >= distance:
            self._position_m = np.array(self._target_position_m, copy=True)
            self._velocity_m_s.fill(0.0)
        else:
            self._position_m += direction * displacement
            self._velocity_m_s = direction * next_speed

        self._timestamp_s += duration
        return self.read_state()


class RecenterAction(str, Enum):
    HOLD = "hold"
    MOVE = "move"
    STOP = "stop"
    BLOCKED = "blocked"


@dataclass(frozen=True)
class RecenterDecision:
    """Observable result of one deadband-controller update."""

    action: RecenterAction
    local_error_m: np.ndarray
    target_stage_position_m: np.ndarray | None
    active: bool
    saturated: bool
    reason: str

    def __post_init__(self) -> None:
        if not isinstance(self.action, RecenterAction):
            raise StageError("action must be a RecenterAction")
        object.__setattr__(
            self,
            "local_error_m",
            _vector2(self.local_error_m, "local_error_m"),
        )
        if self.target_stage_position_m is not None:
            object.__setattr__(
                self,
                "target_stage_position_m",
                _vector2(
                    self.target_stage_position_m,
                    "target_stage_position_m",
                ),
            )
        if not isinstance(self.active, bool) or not isinstance(self.saturated, bool):
            raise StageError("active and saturated must be booleans")
        if not isinstance(self.reason, str):
            raise StageError("reason must be a string")


class DeadbandRecenterController:
    """Hysteretic local-workspace recentering for a slow planar stage.

    The controller starts moving when local error exceeds ``trigger_radius_m``
    and stops only after it falls below ``stop_radius_m``.  The default mapping
    assumes positive stage motion produces the same correction in stage
    coordinates as the observed robot-local displacement.  Set
    ``local_to_stage`` to reflect the measured mechanism kinematics.
    """

    def __init__(
        self,
        *,
        trigger_radius_m: float,
        stop_radius_m: float,
        target_local_position_m: Sequence[float] | np.ndarray = (0.0, 0.0),
        local_to_stage: Sequence[Sequence[float]] | np.ndarray = (
            (1.0, 0.0),
            (0.0, 1.0),
        ),
        maximum_speed_m_s: float | Sequence[float] | np.ndarray | None = None,
    ) -> None:
        trigger = float(trigger_radius_m)
        stop = float(stop_radius_m)
        if not np.isfinite(trigger) or trigger <= 0.0:
            raise StageError("trigger_radius_m must be finite and positive")
        if not np.isfinite(stop) or stop < 0.0 or stop >= trigger:
            raise StageError(
                "stop_radius_m must satisfy 0 <= stop < trigger_radius_m"
            )
        transform = np.array(local_to_stage, dtype=np.float64, copy=True)
        if transform.shape != (2, 2):
            raise StageError("local_to_stage must have shape (2, 2)")
        if not np.all(np.isfinite(transform)):
            raise StageError("local_to_stage must contain only finite values")
        if np.linalg.matrix_rank(transform) < 2:
            raise StageError("local_to_stage must have rank 2")
        transform.setflags(write=False)

        self.trigger_radius_m = trigger
        self.stop_radius_m = stop
        self.target_local_position_m = _vector2(
            target_local_position_m,
            "target_local_position_m",
        )
        self.local_to_stage = transform
        self.maximum_speed_m_s = (
            None
            if maximum_speed_m_s is None
            else _positive_vector2(maximum_speed_m_s, "maximum_speed_m_s")
        )
        if self.maximum_speed_m_s is not None and np.any(
            self.maximum_speed_m_s <= 0.0
        ):
            raise StageError("maximum_speed_m_s must be positive")
        self._active = False

    @property
    def active(self) -> bool:
        return self._active

    def reset(self) -> None:
        self._active = False

    def update(
        self,
        stage: XYStage,
        robot_position_local_m: Sequence[float] | np.ndarray,
    ) -> RecenterDecision:
        """Evaluate the hysteresis state and, when safe, command the stage."""

        robot_position = _vector2(
            robot_position_local_m,
            "robot_position_local_m",
        )
        error = robot_position - self.target_local_position_m
        error_norm = float(np.linalg.norm(error))
        state = stage.read_state()

        if state.fault is not None:
            self._active = False
            return RecenterDecision(
                action=RecenterAction.BLOCKED,
                local_error_m=error,
                target_stage_position_m=None,
                active=False,
                saturated=False,
                reason=f"stage fault: {state.fault}",
            )
        if not state.homed:
            self._active = False
            return RecenterDecision(
                action=RecenterAction.BLOCKED,
                local_error_m=error,
                target_stage_position_m=None,
                active=False,
                saturated=False,
                reason="stage is not homed",
            )

        if not self._active:
            if error_norm <= self.trigger_radius_m:
                return RecenterDecision(
                    action=RecenterAction.HOLD,
                    local_error_m=error,
                    target_stage_position_m=None,
                    active=False,
                    saturated=False,
                    reason="robot remains inside the trigger deadband",
                )
            self._active = True

        if error_norm <= self.stop_radius_m:
            stage.stop()
            self._active = False
            return RecenterDecision(
                action=RecenterAction.STOP,
                local_error_m=error,
                target_stage_position_m=state.position_m,
                active=False,
                saturated=False,
                reason="robot entered the stop deadband",
            )

        requested_target = (
            state.position_m + self.local_to_stage @ error
        )
        target = stage.limits.clip(requested_target)
        saturated = not np.allclose(target, requested_target, rtol=0.0, atol=1e-15)
        if np.allclose(target, state.position_m, rtol=0.0, atol=1e-15):
            self._active = False
            stage.stop()
            return RecenterDecision(
                action=RecenterAction.BLOCKED,
                local_error_m=error,
                target_stage_position_m=target,
                active=False,
                saturated=saturated,
                reason="stage travel limit prevents further recentering",
            )

        stage.command_position(target, maximum_speed_m_s=self.maximum_speed_m_s)
        return RecenterDecision(
            action=RecenterAction.MOVE,
            local_error_m=error,
            target_stage_position_m=target,
            active=True,
            saturated=saturated,
            reason=(
                "recenter target clipped to stage travel limits"
                if saturated
                else "recenter motion commanded"
            ),
        )


__all__ = [
    "DeadbandRecenterController",
    "RecenterAction",
    "RecenterDecision",
    "SimulatedXYStage",
    "StageError",
    "StageLimits",
    "StageState",
    "XYStage",
]
