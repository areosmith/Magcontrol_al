"""Open-loop rotating uniform-field control for a three-axis coil system."""

from .calibration import (
    ActuatorCalibration,
    CalibrationError,
    FieldMatrixFit,
    RLFit,
    fit_field_matrix,
    fit_rl_channel,
)
from .calculator import (
    CalculatorFault,
    CalculatorStatus,
    CommandValidationError,
    RotatingFieldCalculator,
    RotatingFieldCommand,
    SafetyLimitError,
    SignalPreview,
)

__all__ = [
    "ActuatorCalibration",
    "CalibrationError",
    "CalculatorFault",
    "CalculatorStatus",
    "CommandValidationError",
    "FieldMatrixFit",
    "RLFit",
    "RotatingFieldCalculator",
    "RotatingFieldCommand",
    "SafetyLimitError",
    "SignalPreview",
    "fit_field_matrix",
    "fit_rl_channel",
]
