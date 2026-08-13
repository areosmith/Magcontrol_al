"""High-level coordination for calibrated propulsive-force commands."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .calculator import CalculatorStatus, RotatingFieldCalculator, SignalPreview
from .force_allocator import ForceAllocationResult, ForceAllocator
from .force_command import PropulsionCommand
from .propulsion_model import RobotProfile


@dataclass(frozen=True)
class SupervisorResult:
    """Result of previewing or scheduling one high-level command."""

    accepted: bool
    message: str
    allocation: ForceAllocationResult
    preview: SignalPreview | None = None
    calculator_status: CalculatorStatus | None = None

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "accepted": self.accepted,
            "message": self.message,
            "allocation": {
                "accepted": self.allocation.accepted,
                "requested_vector_si": self.allocation.requested_vector_si.tolist(),
                "predicted_vector_si": (
                    None
                    if self.allocation.predicted_vector_si is None
                    else self.allocation.predicted_vector_si.tolist()
                ),
                "projected": self.allocation.projected,
                "reasons": [reason.value for reason in self.allocation.reasons],
                "message": self.allocation.message,
                "stepout_margin_hz": self.allocation.stepout_margin_hz,
            },
        }
        if self.preview is not None:
            result["preview"] = self.preview.to_dict()
        if self.calculator_status is not None:
            result["calculator_status"] = self.calculator_status.to_dict()
        return result


class MagneticControlSupervisor:
    """Connect force allocation to the existing DAQ-voltage calculator.

    Rejected allocations never mutate calculator state.  A zero target is the
    one allocator rejection handled specially: it schedules the calculator's
    controlled stop path instead of constructing an invalid zero-frequency
    rotating-field command.
    """

    def __init__(
        self,
        allocator: ForceAllocator,
        profile: RobotProfile,
        calculator: RotatingFieldCalculator,
    ) -> None:
        if not isinstance(allocator, ForceAllocator):
            raise TypeError("allocator must be a ForceAllocator")
        if not isinstance(profile, RobotProfile):
            raise TypeError("profile must be a RobotProfile")
        if not isinstance(calculator, RotatingFieldCalculator):
            raise TypeError("calculator must be a RotatingFieldCalculator")
        self.allocator = allocator
        self.profile = profile
        self.calculator = calculator

    def _allocate(self, command: PropulsionCommand) -> ForceAllocationResult:
        return self.allocator.allocate(command, self.profile)

    @staticmethod
    def _is_stop(command: PropulsionCommand) -> bool:
        # Keep this identical to ForceAllocator's zero-target rule.  Treating a
        # merely small non-zero force as stop would make the nested allocation
        # and the action reported by the supervisor disagree.
        return command.target_magnitude_si == 0.0

    def preview_force(self, command: PropulsionCommand) -> SupervisorResult:
        allocation = self._allocate(command)
        if self._is_stop(command):
            return SupervisorResult(
                accepted=True,
                message="zero target would schedule a controlled stop",
                allocation=allocation,
                calculator_status=self.calculator.status(),
            )
        if not allocation.accepted:
            return SupervisorResult(
                accepted=False,
                message=allocation.message,
                allocation=allocation,
                calculator_status=self.calculator.status(),
            )
        try:
            preview = self.calculator.preview(allocation.rotating_field_command)
        except Exception as exc:
            return SupervisorResult(
                accepted=False,
                message=f"electrical preview rejected command: {exc}",
                allocation=allocation,
                calculator_status=self.calculator.status(),
            )
        return SupervisorResult(
            accepted=True,
            message="force request is feasible",
            allocation=allocation,
            preview=preview,
            calculator_status=self.calculator.status(),
        )

    def set_force(
        self,
        command: PropulsionCommand,
        *,
        transition_s: float | None = None,
    ) -> SupervisorResult:
        allocation = self._allocate(command)
        if self._is_stop(command):
            status = self.calculator.stop(transition_s=transition_s)
            return SupervisorResult(
                accepted=True,
                message="controlled stop scheduled",
                allocation=allocation,
                calculator_status=status,
            )
        if not allocation.accepted:
            return SupervisorResult(
                accepted=False,
                message=allocation.message,
                allocation=allocation,
                calculator_status=self.calculator.status(),
            )
        try:
            preview = self.calculator.preview(allocation.rotating_field_command)
            status = self.calculator.set_command(
                allocation.rotating_field_command,
                transition_s=transition_s,
            )
        except Exception as exc:
            return SupervisorResult(
                accepted=False,
                message=f"electrical calculator rejected command: {exc}",
                allocation=allocation,
                calculator_status=self.calculator.status(),
            )
        return SupervisorResult(
            accepted=True,
            message="force request scheduled",
            allocation=allocation,
            preview=preview,
            calculator_status=status,
        )

    def generate_voltage_block(self, samples: int) -> np.ndarray:
        """DAQ-compatible ``source(samples) -> (3, samples)`` callback."""

        return self.calculator.generate_voltage_block(samples)

    def stop(self, *, transition_s: float | None = None) -> CalculatorStatus:
        return self.calculator.stop(transition_s=transition_s)

    def status(self) -> CalculatorStatus:
        return self.calculator.status()
