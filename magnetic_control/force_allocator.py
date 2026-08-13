"""Convert calibrated force targets into rotating-field commands."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np

from .calculator import RotatingFieldCommand
from .force_command import PropulsionCommand, PropulsionMode
from .propulsion_model import (
    InfeasiblePropulsionTargetError,
    PropulsionModelError,
    RobotProfile,
    UnverifiedPropulsionModelError,
)


class AllocationReason(str, Enum):
    """Machine-readable reason attached to an allocation result."""

    PROJECTED_TO_FEASIBLE = "projected_to_feasible"
    TARGET_OUTSIDE_CALIBRATED_RANGE = "target_outside_calibrated_range"
    UNSUPPORTED_MODE = "unsupported_mode"
    PROFILE_MISMATCH = "profile_mismatch"
    MODEL_NOT_AVAILABLE = "model_not_available"
    MODEL_NOT_VERIFIED = "model_not_verified"
    CONTACT_NOT_CONFIRMED = "contact_not_confirmed"
    UNSUPPORTED_FRAME = "unsupported_frame"
    ZERO_TARGET_REQUIRES_STOP = "zero_target_requires_stop"
    INVALID_MODEL = "invalid_model"


def _readonly_vector(value: np.ndarray) -> np.ndarray:
    vector = np.array(value, dtype=np.float64, copy=True)
    vector.setflags(write=False)
    return vector


@dataclass(frozen=True)
class ForceAllocationResult:
    """Accepted command or an explicit rejection with no hardware command."""

    accepted: bool
    requested_vector_si: np.ndarray
    predicted_vector_si: np.ndarray | None
    rotating_field_command: RotatingFieldCommand | None
    projected: bool
    reasons: tuple[AllocationReason, ...]
    message: str
    stepout_lower_hz: float | None = None
    safe_frequency_limit_hz: float | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "requested_vector_si",
            _readonly_vector(self.requested_vector_si),
        )
        if self.predicted_vector_si is not None:
            object.__setattr__(
                self,
                "predicted_vector_si",
                _readonly_vector(self.predicted_vector_si),
            )

    @property
    def stepout_margin_hz(self) -> float | None:
        if self.stepout_lower_hz is None or self.rotating_field_command is None:
            return None
        return self.stepout_lower_hz - self.rotating_field_command.frequency_hz


class ForceAllocator:
    """Allocate force modes using a verified robot-specific LUT."""

    def __init__(self, *, preferred_rotation_sense: int = 1) -> None:
        if preferred_rotation_sense not in (-1, 1):
            raise ValueError("preferred_rotation_sense must be -1 or +1")
        self.preferred_rotation_sense = int(preferred_rotation_sense)

    @staticmethod
    def _rejected(
        command: PropulsionCommand,
        reason: AllocationReason,
        message: str,
    ) -> ForceAllocationResult:
        return ForceAllocationResult(
            accepted=False,
            requested_vector_si=command.target_xyz_si,
            predicted_vector_si=None,
            rotating_field_command=None,
            projected=False,
            reasons=(reason,),
            message=message,
        )

    def allocate(
        self,
        command: PropulsionCommand,
        profile: RobotProfile,
    ) -> ForceAllocationResult:
        if not isinstance(command, PropulsionCommand):
            raise TypeError("command must be a PropulsionCommand")
        if not isinstance(profile, RobotProfile):
            raise TypeError("profile must be a RobotProfile")

        target_magnitude = command.target_magnitude_si
        if target_magnitude == 0.0:
            return self._rejected(
                command,
                AllocationReason.ZERO_TARGET_REQUIRES_STOP,
                "a zero force target must be handled by the supervisor stop path",
            )
        if command.robot_profile_id != profile.profile_id:
            return self._rejected(
                command,
                AllocationReason.PROFILE_MISMATCH,
                "command robot_profile_id does not match the supplied profile",
            )
        if command.frame != "lab":
            return self._rejected(
                command,
                AllocationReason.UNSUPPORTED_FRAME,
                "ForceAllocator currently accepts lab-frame vectors only",
            )
        if command.mode is PropulsionMode.FREE_SWIM_VELOCITY:
            return self._rejected(
                command,
                AllocationReason.UNSUPPORTED_MODE,
                "ForceAllocator accepts force modes only; use a velocity allocator for m/s targets",
            )
        if (
            command.mode is PropulsionMode.BLOCKED_THRUST
            and (not command.contact_confirmed or command.contact_id is None)
        ):
            return self._rejected(
                command,
                AllocationReason.CONTACT_NOT_CONFIRMED,
                "blocked thrust requires a confirmed, identified contact",
            )

        model = profile.model_for(command.mode)
        if model is None:
            return self._rejected(
                command,
                AllocationReason.MODEL_NOT_AVAILABLE,
                f"profile has no {command.mode.value} calibration",
            )

        try:
            inverse = model.inverse(
                target_magnitude,
                project_to_feasible=command.project_to_feasible,
            )
        except UnverifiedPropulsionModelError as exc:
            return self._rejected(
                command,
                AllocationReason.MODEL_NOT_VERIFIED,
                str(exc),
            )
        except InfeasiblePropulsionTargetError as exc:
            return self._rejected(
                command,
                AllocationReason.TARGET_OUTSIDE_CALIBRATED_RANGE,
                str(exc),
            )
        except PropulsionModelError as exc:
            return self._rejected(
                command,
                AllocationReason.INVALID_MODEL,
                str(exc),
            )

        requested_direction = command.target_xyz_si / target_magnitude
        sense = self.preferred_rotation_sense
        axis = requested_direction * profile.propulsion_sign * sense
        field_command = RotatingFieldCommand(
            axis_xyz=axis,
            magnitude_T=inverse.field_magnitude_T,
            frequency_hz=inverse.frequency_hz,
            rotation_sense=sense,
        )
        predicted_direction = axis * profile.propulsion_sign * sense
        predicted_vector = predicted_direction * inverse.achieved_si
        reasons = (
            (AllocationReason.PROJECTED_TO_FEASIBLE,) if inverse.projected else ()
        )
        message = (
            "target projected to the calibrated safe boundary"
            if inverse.projected
            else "target allocated within the calibrated safe range"
        )
        return ForceAllocationResult(
            accepted=True,
            requested_vector_si=command.target_xyz_si,
            predicted_vector_si=predicted_vector,
            rotating_field_command=field_command,
            projected=inverse.projected,
            reasons=reasons,
            message=message,
            stepout_lower_hz=inverse.stepout_lower_hz,
            safe_frequency_limit_hz=inverse.safe_frequency_limit_hz,
        )
