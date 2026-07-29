"""Calibration models for a three-axis magnetic-field actuator.

All calculations use SI units:

* magnetic field: tesla (T)
* current: ampere (A)
* voltage: volt (V)
* resistance: ohm
* inductance: henry (H)

The magnetic model is

    B_total = C_BI @ I_model + B_ambient

where the columns of ``C_BI`` follow DAQ channel order.  ``channel_signs``
maps electrically positive coil current to the current sign used by the
magnetic model.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


class CalibrationError(ValueError):
    """Raised when calibration data is missing, malformed, or unsafe."""


def _readonly_array(
    value: Sequence[float] | Sequence[Sequence[float]] | np.ndarray,
    shape: tuple[int, ...],
    name: str,
) -> np.ndarray:
    array = np.array(value, dtype=np.float64, copy=True)
    if array.shape != shape:
        raise CalibrationError(f"{name} must have shape {shape}, got {array.shape}")
    if not np.all(np.isfinite(array)):
        raise CalibrationError(f"{name} must contain only finite values")
    array.setflags(write=False)
    return array


@dataclass(frozen=True)
class ActuatorCalibration:
    """Versioned calibration for three coil channels and their voltage drive."""

    field_matrix_t_per_a: np.ndarray
    ambient_field_t: np.ndarray
    channel_axes: tuple[str, str, str]
    channel_signs: np.ndarray
    amplifier_gain_v_per_v: np.ndarray
    coil_resistance_ohm: np.ndarray
    coil_inductance_h: np.ndarray
    daq_voltage_limit_v: np.ndarray
    peak_current_limit_a: np.ndarray
    rms_current_limit_a: np.ndarray
    valid_frequency_hz: tuple[float, float]
    verified: bool = False
    calibration_date: str = ""
    fit_metrics: Mapping[str, Any] = field(default_factory=dict)
    condition_number_limit: float = 100.0
    config_version: int = 1

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "field_matrix_t_per_a",
            _readonly_array(
                self.field_matrix_t_per_a,
                (3, 3),
                "field_matrix_t_per_a",
            ),
        )
        object.__setattr__(
            self,
            "ambient_field_t",
            _readonly_array(self.ambient_field_t, (3,), "ambient_field_t"),
        )
        object.__setattr__(
            self,
            "channel_signs",
            _readonly_array(self.channel_signs, (3,), "channel_signs"),
        )
        object.__setattr__(
            self,
            "amplifier_gain_v_per_v",
            _readonly_array(
                self.amplifier_gain_v_per_v,
                (3,),
                "amplifier_gain_v_per_v",
            ),
        )
        object.__setattr__(
            self,
            "coil_resistance_ohm",
            _readonly_array(
                self.coil_resistance_ohm,
                (3,),
                "coil_resistance_ohm",
            ),
        )
        object.__setattr__(
            self,
            "coil_inductance_h",
            _readonly_array(self.coil_inductance_h, (3,), "coil_inductance_h"),
        )
        object.__setattr__(
            self,
            "daq_voltage_limit_v",
            _readonly_array(self.daq_voltage_limit_v, (3,), "daq_voltage_limit_v"),
        )
        object.__setattr__(
            self,
            "peak_current_limit_a",
            _readonly_array(
                self.peak_current_limit_a,
                (3,),
                "peak_current_limit_a",
            ),
        )
        object.__setattr__(
            self,
            "rms_current_limit_a",
            _readonly_array(
                self.rms_current_limit_a,
                (3,),
                "rms_current_limit_a",
            ),
        )

        axes = tuple(str(axis).lower() for axis in self.channel_axes)
        if len(axes) != 3 or set(axes) != {"x", "y", "z"}:
            raise CalibrationError(
                "channel_axes must contain x, y, and z exactly once"
            )
        object.__setattr__(self, "channel_axes", axes)

        if not np.all(np.isin(self.channel_signs, (-1.0, 1.0))):
            raise CalibrationError("channel_signs must contain only -1 or +1")
        if np.any(self.amplifier_gain_v_per_v <= 0):
            raise CalibrationError("amplifier gains must be positive")
        if np.any(self.coil_resistance_ohm <= 0):
            raise CalibrationError("coil resistances must be positive")
        if np.any(self.coil_inductance_h < 0):
            raise CalibrationError("coil inductances must be non-negative")
        if np.any(self.daq_voltage_limit_v <= 0):
            raise CalibrationError("DAQ voltage limits must be positive")
        if np.any(self.peak_current_limit_a <= 0):
            raise CalibrationError("peak-current limits must be positive")
        if np.any(self.rms_current_limit_a <= 0):
            raise CalibrationError("RMS-current limits must be positive")
        if np.any(self.rms_current_limit_a > self.peak_current_limit_a):
            raise CalibrationError(
                "each RMS-current limit must not exceed its peak-current limit"
            )

        if len(self.valid_frequency_hz) != 2:
            raise CalibrationError("valid_frequency_hz must be [minimum, maximum]")
        frequency_range = tuple(float(v) for v in self.valid_frequency_hz)
        if not np.all(np.isfinite(frequency_range)):
            raise CalibrationError("valid_frequency_hz must be finite")
        if frequency_range[0] < 0 or frequency_range[1] <= frequency_range[0]:
            raise CalibrationError(
                "valid_frequency_hz must satisfy 0 <= minimum < maximum"
            )
        object.__setattr__(self, "valid_frequency_hz", frequency_range)

        if not np.isfinite(self.condition_number_limit):
            raise CalibrationError("condition_number_limit must be finite")
        if self.condition_number_limit <= 1.0:
            raise CalibrationError("condition_number_limit must be greater than 1")
        if self.config_version != 1:
            raise CalibrationError(
                f"unsupported calibration config version: {self.config_version}"
            )
        if not isinstance(self.verified, bool):
            raise CalibrationError("verified must be a boolean")
        if not isinstance(self.fit_metrics, Mapping):
            raise CalibrationError("fit_metrics must be a mapping")
        object.__setattr__(self, "fit_metrics", dict(self.fit_metrics))

    @property
    def matrix_rank(self) -> int:
        return int(np.linalg.matrix_rank(self.field_matrix_t_per_a))

    @property
    def condition_number(self) -> float:
        return float(np.linalg.cond(self.field_matrix_t_per_a))

    def readiness_errors(self, require_verified: bool = True) -> list[str]:
        """Return reasons why this calibration must not drive real hardware."""

        errors: list[str] = []
        if require_verified and not self.verified:
            errors.append("calibration is not marked verified")
        if self.matrix_rank != 3:
            errors.append(
                f"field matrix rank is {self.matrix_rank}; full rank 3 is required"
            )
        if not np.isfinite(self.condition_number):
            errors.append("field matrix condition number is not finite")
        elif self.condition_number > self.condition_number_limit:
            errors.append(
                "field matrix condition number "
                f"{self.condition_number:.3g} exceeds "
                f"{self.condition_number_limit:.3g}"
            )
        return errors

    def assert_ready(self, require_verified: bool = True) -> None:
        errors = self.readiness_errors(require_verified=require_verified)
        if errors:
            raise CalibrationError("; ".join(errors))

    @property
    def inverse_field_matrix(self) -> np.ndarray:
        self.assert_ready(require_verified=False)
        return np.linalg.inv(self.field_matrix_t_per_a)

    def model_currents_for_coil_field(self, coil_field_t: np.ndarray) -> np.ndarray:
        """Convert desired coil-generated field to model-current channel order."""

        field_array = np.asarray(coil_field_t, dtype=np.float64)
        if field_array.shape[0] != 3:
            raise CalibrationError(
                "coil_field_t must have first dimension 3 (world X/Y/Z)"
            )
        return self.inverse_field_matrix @ field_array

    def coil_field_for_model_currents(self, model_current_a: np.ndarray) -> np.ndarray:
        current_array = np.asarray(model_current_a, dtype=np.float64)
        if current_array.shape[0] != 3:
            raise CalibrationError(
                "model_current_a must have first dimension 3 (channel order)"
            )
        return self.field_matrix_t_per_a @ current_array

    def model_to_electrical_current(self, model_current_a: np.ndarray) -> np.ndarray:
        current_array = np.asarray(model_current_a, dtype=np.float64)
        if current_array.shape[0] != 3:
            raise CalibrationError("model_current_a must have first dimension 3")
        reshape = (3,) + (1,) * (current_array.ndim - 1)
        return self.channel_signs.reshape(reshape) * current_array

    def electrical_to_model_current(
        self,
        electrical_current_a: np.ndarray,
    ) -> np.ndarray:
        current_array = np.asarray(electrical_current_a, dtype=np.float64)
        if current_array.shape[0] != 3:
            raise CalibrationError(
                "electrical_current_a must have first dimension 3"
            )
        reshape = (3,) + (1,) * (current_array.ndim - 1)
        return self.channel_signs.reshape(reshape) * current_array

    def rl_discrete_coefficients(self, dt_s: float) -> tuple[np.ndarray, np.ndarray]:
        """Return ``a, b`` for ``i[k+1] = a*i[k] + b*v_daq[k]``."""

        dt_s = float(dt_s)
        if not np.isfinite(dt_s) or dt_s <= 0:
            raise CalibrationError("dt_s must be a positive finite value")

        resistance = self.coil_resistance_ohm
        inductance = self.coil_inductance_h
        gain = self.amplifier_gain_v_per_v

        dynamic = inductance > 1e-15
        a = np.zeros(3, dtype=np.float64)
        a[dynamic] = np.exp(
            -resistance[dynamic] * dt_s / inductance[dynamic]
        )
        b = np.empty(3, dtype=np.float64)
        b[dynamic] = (
            gain[dynamic]
            / resistance[dynamic]
            * (1.0 - a[dynamic])
        )
        b[~dynamic] = gain[~dynamic] / resistance[~dynamic]
        return a, b

    def daq_voltage_for_current_step(
        self,
        current_now_a: np.ndarray,
        current_next_a: np.ndarray,
        dt_s: float,
    ) -> np.ndarray:
        """Invert the per-channel discrete RL model for one or many samples."""

        current_now = np.asarray(current_now_a, dtype=np.float64)
        current_next = np.asarray(current_next_a, dtype=np.float64)
        if current_now.shape != current_next.shape or current_now.shape[0] != 3:
            raise CalibrationError(
                "current arrays must have equal shape with first dimension 3"
            )
        a, b = self.rl_discrete_coefficients(dt_s)
        reshape = (3,) + (1,) * (current_now.ndim - 1)
        return (
            current_next - a.reshape(reshape) * current_now
        ) / b.reshape(reshape)

    def electrical_current_step(
        self,
        current_now_a: np.ndarray,
        daq_voltage_v: np.ndarray,
        dt_s: float,
    ) -> np.ndarray:
        """Apply the forward discrete RL model, primarily for verification."""

        current_now = np.asarray(current_now_a, dtype=np.float64)
        voltage = np.asarray(daq_voltage_v, dtype=np.float64)
        if current_now.shape != voltage.shape or current_now.shape[0] != 3:
            raise CalibrationError(
                "current and voltage arrays must have equal shape with first dimension 3"
            )
        a, b = self.rl_discrete_coefficients(dt_s)
        reshape = (3,) + (1,) * (current_now.ndim - 1)
        return (
            a.reshape(reshape) * current_now
            + b.reshape(reshape) * voltage
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "config_version": self.config_version,
            "verified": self.verified,
            "calibration_date": self.calibration_date,
            "channel_axes": list(self.channel_axes),
            "channel_signs": self.channel_signs.tolist(),
            "field_matrix_t_per_a": self.field_matrix_t_per_a.tolist(),
            "ambient_field_t": self.ambient_field_t.tolist(),
            "amplifier_gain_v_per_v": self.amplifier_gain_v_per_v.tolist(),
            "coil_resistance_ohm": self.coil_resistance_ohm.tolist(),
            "coil_inductance_h": self.coil_inductance_h.tolist(),
            "daq_voltage_limit_v": self.daq_voltage_limit_v.tolist(),
            "peak_current_limit_a": self.peak_current_limit_a.tolist(),
            "rms_current_limit_a": self.rms_current_limit_a.tolist(),
            "valid_frequency_hz": list(self.valid_frequency_hz),
            "condition_number_limit": self.condition_number_limit,
            "fit_metrics": dict(self.fit_metrics),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ActuatorCalibration":
        try:
            return cls(
                field_matrix_t_per_a=data["field_matrix_t_per_a"],
                ambient_field_t=data["ambient_field_t"],
                channel_axes=tuple(data["channel_axes"]),
                channel_signs=data["channel_signs"],
                amplifier_gain_v_per_v=data["amplifier_gain_v_per_v"],
                coil_resistance_ohm=data["coil_resistance_ohm"],
                coil_inductance_h=data["coil_inductance_h"],
                daq_voltage_limit_v=data["daq_voltage_limit_v"],
                peak_current_limit_a=data["peak_current_limit_a"],
                rms_current_limit_a=data["rms_current_limit_a"],
                valid_frequency_hz=tuple(data["valid_frequency_hz"]),
                verified=data.get("verified", False),
                calibration_date=str(data.get("calibration_date", "")),
                fit_metrics=data.get("fit_metrics", {}),
                condition_number_limit=float(
                    data.get("condition_number_limit", 100.0)
                ),
                config_version=int(data.get("config_version", 1)),
            )
        except KeyError as exc:
            raise CalibrationError(
                f"missing calibration field: {exc.args[0]}"
            ) from exc

    @classmethod
    def from_json(cls, path: str | Path) -> "ActuatorCalibration":
        with Path(path).open("r", encoding="utf-8") as file:
            data = json.load(file)
        if not isinstance(data, Mapping):
            raise CalibrationError("calibration JSON root must be an object")
        return cls.from_dict(data)

    def to_json(self, path: str | Path) -> None:
        with Path(path).open("w", encoding="utf-8") as file:
            json.dump(self.to_dict(), file, indent=2, ensure_ascii=False)
            file.write("\n")


@dataclass(frozen=True)
class FieldMatrixFit:
    """Result of an affine least-squares magnetic-field calibration."""

    field_matrix_t_per_a: np.ndarray
    ambient_field_t: np.ndarray
    rms_error_t: np.ndarray
    max_vector_error_t: float
    condition_number: float


def fit_field_matrix(
    model_currents_a: np.ndarray,
    measured_fields_t: np.ndarray,
) -> FieldMatrixFit:
    """Fit ``B = C @ I + B_ambient`` from N three-axis measurements."""

    currents = np.asarray(model_currents_a, dtype=np.float64)
    fields = np.asarray(measured_fields_t, dtype=np.float64)
    if currents.ndim != 2 or currents.shape[1] != 3:
        raise CalibrationError("model_currents_a must have shape (N, 3)")
    if fields.shape != currents.shape:
        raise CalibrationError(
            "measured_fields_t must have the same (N, 3) shape as currents"
        )
    if currents.shape[0] < 4:
        raise CalibrationError("at least four calibration measurements are required")
    if not np.all(np.isfinite(currents)) or not np.all(np.isfinite(fields)):
        raise CalibrationError("calibration samples must be finite")

    design = np.column_stack((currents, np.ones(currents.shape[0])))
    coefficients, _, design_rank, _ = np.linalg.lstsq(design, fields, rcond=None)
    if design_rank < 4:
        raise CalibrationError(
            "calibration current samples do not span all three current axes"
        )

    matrix = coefficients[:3, :].T
    ambient = coefficients[3, :]
    predicted = design @ coefficients
    residual = predicted - fields
    rms = np.sqrt(np.mean(residual**2, axis=0))
    max_vector = float(np.max(np.linalg.norm(residual, axis=1)))
    condition = float(np.linalg.cond(matrix))

    matrix.setflags(write=False)
    ambient.setflags(write=False)
    rms.setflags(write=False)
    return FieldMatrixFit(
        field_matrix_t_per_a=matrix,
        ambient_field_t=ambient,
        rms_error_t=rms,
        max_vector_error_t=max_vector,
        condition_number=condition,
    )


@dataclass(frozen=True)
class RLFit:
    """Result of fitting the series model ``Z(f) = R + j*2*pi*f*L``."""

    resistance_ohm: float
    inductance_h: float
    rms_impedance_error_ohm: float
    max_relative_error: float


def fit_rl_channel(
    frequency_hz: np.ndarray,
    daq_voltage_phasor_v: np.ndarray,
    current_phasor_a: np.ndarray,
    amplifier_gain_v_per_v: float,
) -> RLFit:
    """Fit one channel from complex DAQ-voltage and current phasors."""

    frequency = np.asarray(frequency_hz, dtype=np.float64)
    voltage = np.asarray(daq_voltage_phasor_v, dtype=np.complex128)
    current = np.asarray(current_phasor_a, dtype=np.complex128)
    if frequency.ndim != 1 or voltage.shape != frequency.shape:
        raise CalibrationError("frequency and voltage phasors must be 1-D and equal")
    if current.shape != frequency.shape:
        raise CalibrationError("current phasors must match frequency shape")
    if frequency.size < 2:
        raise CalibrationError("at least two impedance samples are required")
    if np.any(frequency < 0) or not np.all(np.isfinite(frequency)):
        raise CalibrationError("frequencies must be finite and non-negative")
    if not np.all(np.isfinite(voltage)) or not np.all(np.isfinite(current)):
        raise CalibrationError("phasors must be finite")
    if np.any(np.abs(current) < 1e-12):
        raise CalibrationError("current phasors must be non-zero")
    gain = float(amplifier_gain_v_per_v)
    if not np.isfinite(gain) or gain <= 0:
        raise CalibrationError("amplifier gain must be positive and finite")

    impedance = gain * voltage / current
    omega = 2.0 * np.pi * frequency
    resistance = float(np.mean(impedance.real))
    denominator = float(np.dot(omega, omega))
    if denominator <= 0:
        raise CalibrationError("at least one non-zero frequency is required")
    inductance = float(np.dot(omega, impedance.imag) / denominator)
    if resistance <= 0 or inductance < 0:
        raise CalibrationError(
            "fitted R/L values are non-physical; check phasor sign conventions"
        )

    predicted = resistance + 1j * omega * inductance
    error = predicted - impedance
    rms_error = float(np.sqrt(np.mean(np.abs(error) ** 2)))
    relative = np.abs(error) / np.maximum(np.abs(impedance), 1e-12)
    return RLFit(
        resistance_ohm=resistance,
        inductance_h=inductance,
        rms_impedance_error_ohm=rms_error,
        max_relative_error=float(np.max(relative)),
    )
