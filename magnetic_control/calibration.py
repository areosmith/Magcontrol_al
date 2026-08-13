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

import csv
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


class CalibrationError(ValueError):
    """Raised when calibration data is missing, malformed, or unsafe."""


VOLTAGE_AMPLIFIER = "voltage_amplifier"
CURRENT_DRIVER = "current_driver"
_SUPPORTED_DRIVE_MODES = frozenset((VOLTAGE_AMPLIFIER, CURRENT_DRIVER))

_GAUSSMETER_CURRENT_COLUMNS = (
    "current_ch0_a",
    "current_ch1_a",
    "current_ch2_a",
)
_GAUSSMETER_FIELD_COLUMNS_T = (
    "field_x_t",
    "field_y_t",
    "field_z_t",
)
_GAUSSMETER_FIELD_COLUMNS_GAUSS = (
    "field_x_gauss",
    "field_y_gauss",
    "field_z_gauss",
)


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
    channel_drive_modes: tuple[str, str, str] = (
        VOLTAGE_AMPLIFIER,
        VOLTAGE_AMPLIFIER,
        VOLTAGE_AMPLIFIER,
    )
    current_gain_a_per_v: np.ndarray | None = None
    current_driver_bandwidth_hz: np.ndarray | None = None

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
        current_gain = (
            np.zeros(3, dtype=np.float64)
            if self.current_gain_a_per_v is None
            else self.current_gain_a_per_v
        )
        object.__setattr__(
            self,
            "current_gain_a_per_v",
            _readonly_array(
                current_gain,
                (3,),
                "current_gain_a_per_v",
            ),
        )
        current_bandwidth = (
            np.zeros(3, dtype=np.float64)
            if self.current_driver_bandwidth_hz is None
            else self.current_driver_bandwidth_hz
        )
        object.__setattr__(
            self,
            "current_driver_bandwidth_hz",
            _readonly_array(
                current_bandwidth,
                (3,),
                "current_driver_bandwidth_hz",
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

        drive_modes = tuple(str(mode).lower() for mode in self.channel_drive_modes)
        if len(drive_modes) != 3:
            raise CalibrationError("channel_drive_modes must contain three entries")
        unsupported = sorted(set(drive_modes) - _SUPPORTED_DRIVE_MODES)
        if unsupported:
            raise CalibrationError(
                "unsupported channel drive mode(s): " + ", ".join(unsupported)
            )
        object.__setattr__(self, "channel_drive_modes", drive_modes)

        if not np.all(np.isin(self.channel_signs, (-1.0, 1.0))):
            raise CalibrationError("channel_signs must contain only -1 or +1")
        voltage_channels = np.array(
            [mode == VOLTAGE_AMPLIFIER for mode in drive_modes],
            dtype=bool,
        )
        current_channels = ~voltage_channels
        if np.any(self.amplifier_gain_v_per_v[voltage_channels] <= 0):
            raise CalibrationError(
                "voltage-amplifier gains must be positive on voltage-driven channels"
            )
        if np.any(self.amplifier_gain_v_per_v[current_channels] < 0):
            raise CalibrationError("unused voltage-amplifier gains must be non-negative")
        if np.any(self.current_gain_a_per_v[current_channels] <= 0):
            raise CalibrationError(
                "current-driver gains must be positive on current-driven channels"
            )
        if np.any(self.current_gain_a_per_v[voltage_channels] < 0):
            raise CalibrationError("unused current-driver gains must be non-negative")
        if np.any(self.current_driver_bandwidth_hz < 0):
            raise CalibrationError("current-driver bandwidths must be non-negative")
        if np.any(self.coil_resistance_ohm <= 0):
            raise CalibrationError("coil resistances must be positive")
        if np.any(self.coil_inductance_h < 0):
            raise CalibrationError("coil inductances must be non-negative")
        if np.any(self.daq_voltage_limit_v <= 0):
            raise CalibrationError("DAQ voltage limits must be positive")
        if np.any(self.daq_voltage_limit_v > 10.0):
            raise CalibrationError(
                "DAQ voltage limits must not exceed the current output-layer "
                "range of 10 V"
            )
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
        if self.config_version not in (1, 2):
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

    @property
    def voltage_amplifier_mask(self) -> np.ndarray:
        """Boolean channel mask; returned arrays are safe for caller mutation."""

        return np.array(
            [mode == VOLTAGE_AMPLIFIER for mode in self.channel_drive_modes],
            dtype=bool,
        )

    @property
    def current_driver_mask(self) -> np.ndarray:
        return ~self.voltage_amplifier_mask

    def steady_electrical_current_for_daq_voltage(
        self,
        daq_voltage_v: np.ndarray,
    ) -> np.ndarray:
        """Convert static DAQ voltage to steady electrical coil current.

        This helper is intended for centre-field calibration runs.  It uses the
        configured DC voltage gain and coil resistance for voltage amplifiers,
        or the configured A/V gain for current drivers.  Dynamic waveforms must
        still use :meth:`electrical_current_step`.
        """

        voltage = np.asarray(daq_voltage_v, dtype=np.float64)
        if voltage.ndim == 0 or voltage.shape[0] != 3:
            raise CalibrationError(
                "daq_voltage_v must have first dimension 3 (channel order)"
            )
        if not np.all(np.isfinite(voltage)):
            raise CalibrationError("daq_voltage_v must contain only finite values")
        reshape = (3,) + (1,) * (voltage.ndim - 1)
        current_per_volt = np.empty(3, dtype=np.float64)
        voltage_mask = self.voltage_amplifier_mask
        current_mask = self.current_driver_mask
        current_per_volt[voltage_mask] = (
            self.amplifier_gain_v_per_v[voltage_mask]
            / self.coil_resistance_ohm[voltage_mask]
        )
        current_per_volt[current_mask] = self.current_gain_a_per_v[current_mask]
        return current_per_volt.reshape(reshape) * voltage

    def steady_model_current_for_daq_voltage(
        self,
        daq_voltage_v: np.ndarray,
    ) -> np.ndarray:
        """Convert static DAQ voltage directly to signed field-model current."""

        return self.electrical_to_model_current(
            self.steady_electrical_current_for_daq_voltage(daq_voltage_v)
        )

    @property
    def voltage_gain_v_per_v(self) -> np.ndarray:
        """Unambiguous alias for the legacy ``amplifier_gain_v_per_v`` field."""

        return self.amplifier_gain_v_per_v

    def daq_voltage_components_for_current(
        self,
        dc_current_a: np.ndarray,
        cosine_current_a: np.ndarray,
        sine_current_a: np.ndarray,
        signed_frequency_hz: float,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Convert sinusoidal current components to DAQ-voltage components.

        Currents follow ``dc + cosine*cos(theta) + sine*sin(theta)``.  A
        voltage-amplifier channel uses the measured amplifier voltage gain and
        the channel's series R/L model.  A current-driver channel uses A/V gain;
        a positive bandwidth applies a first-order closed-loop model, while a
        zero bandwidth selects an ideal memoryless driver.
        """

        components = []
        for name, value in (
            ("dc_current_a", dc_current_a),
            ("cosine_current_a", cosine_current_a),
            ("sine_current_a", sine_current_a),
        ):
            component = np.asarray(value, dtype=np.float64)
            if component.shape != (3,) or not np.all(np.isfinite(component)):
                raise CalibrationError(f"{name} must be a finite three-vector")
            components.append(component)
        dc_current, cosine_current, sine_current = components
        frequency = float(signed_frequency_hz)
        if not np.isfinite(frequency):
            raise CalibrationError("signed_frequency_hz must be finite")
        omega = 2.0 * np.pi * frequency

        dc_voltage = np.empty(3, dtype=np.float64)
        cosine_voltage = np.empty(3, dtype=np.float64)
        sine_voltage = np.empty(3, dtype=np.float64)

        voltage_mask = self.voltage_amplifier_mask
        voltage_gain = self.amplifier_gain_v_per_v[voltage_mask]
        resistance = self.coil_resistance_ohm[voltage_mask]
        inductance = self.coil_inductance_h[voltage_mask]
        dc_voltage[voltage_mask] = (
            resistance * dc_current[voltage_mask] / voltage_gain
        )
        cosine_voltage[voltage_mask] = (
            resistance * cosine_current[voltage_mask]
            + inductance * omega * sine_current[voltage_mask]
        ) / voltage_gain
        sine_voltage[voltage_mask] = (
            resistance * sine_current[voltage_mask]
            - inductance * omega * cosine_current[voltage_mask]
        ) / voltage_gain

        current_mask = self.current_driver_mask
        current_gain = self.current_gain_a_per_v[current_mask]
        bandwidth = self.current_driver_bandwidth_hz[current_mask]
        inverse_corner = np.zeros_like(bandwidth)
        finite_bandwidth = bandwidth > 0.0
        inverse_corner[finite_bandwidth] = (
            1.0 / (2.0 * np.pi * bandwidth[finite_bandwidth])
        )
        dc_voltage[current_mask] = dc_current[current_mask] / current_gain
        cosine_voltage[current_mask] = (
            cosine_current[current_mask]
            + omega * inverse_corner * sine_current[current_mask]
        ) / current_gain
        sine_voltage[current_mask] = (
            sine_current[current_mask]
            - omega * inverse_corner * cosine_current[current_mask]
        ) / current_gain
        return dc_voltage, cosine_voltage, sine_voltage

    def rl_discrete_coefficients(self, dt_s: float) -> tuple[np.ndarray, np.ndarray]:
        """Return ``a, b`` for ``i[k+1] = a*i[k] + b*v_daq[k]``.

        The historical method name is retained for compatibility.  Returned
        coefficients also cover current-driver channels.
        """

        dt_s = float(dt_s)
        if not np.isfinite(dt_s) or dt_s <= 0:
            raise CalibrationError("dt_s must be a positive finite value")

        a = np.zeros(3, dtype=np.float64)
        b = np.empty(3, dtype=np.float64)

        voltage_mask = self.voltage_amplifier_mask
        inductance = self.coil_inductance_h
        resistance = self.coil_resistance_ohm
        voltage_dynamic = voltage_mask & (inductance > 1e-15)
        voltage_static = voltage_mask & ~voltage_dynamic
        a[voltage_dynamic] = np.exp(
            -resistance[voltage_dynamic]
            * dt_s
            / inductance[voltage_dynamic]
        )
        b[voltage_dynamic] = (
            self.amplifier_gain_v_per_v[voltage_dynamic]
            / resistance[voltage_dynamic]
            * (1.0 - a[voltage_dynamic])
        )
        b[voltage_static] = (
            self.amplifier_gain_v_per_v[voltage_static]
            / resistance[voltage_static]
        )

        current_mask = self.current_driver_mask
        bandwidth = self.current_driver_bandwidth_hz
        current_dynamic = current_mask & (bandwidth > 0.0)
        current_static = current_mask & ~current_dynamic
        a[current_dynamic] = np.exp(
            -2.0 * np.pi * bandwidth[current_dynamic] * dt_s
        )
        b[current_dynamic] = (
            self.current_gain_a_per_v[current_dynamic]
            * (1.0 - a[current_dynamic])
        )
        b[current_static] = self.current_gain_a_per_v[current_static]
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
        data = {
            "config_version": self.config_version,
            "verified": self.verified,
            "calibration_date": self.calibration_date,
            "channel_axes": list(self.channel_axes),
            "channel_signs": self.channel_signs.tolist(),
            "field_matrix_t_per_a": self.field_matrix_t_per_a.tolist(),
            "ambient_field_t": self.ambient_field_t.tolist(),
            "channel_drive_modes": list(self.channel_drive_modes),
            "current_gain_a_per_v": self.current_gain_a_per_v.tolist(),
            "current_driver_bandwidth_hz": (
                self.current_driver_bandwidth_hz.tolist()
            ),
            "coil_resistance_ohm": self.coil_resistance_ohm.tolist(),
            "coil_inductance_h": self.coil_inductance_h.tolist(),
            "daq_voltage_limit_v": self.daq_voltage_limit_v.tolist(),
            "peak_current_limit_a": self.peak_current_limit_a.tolist(),
            "rms_current_limit_a": self.rms_current_limit_a.tolist(),
            "valid_frequency_hz": list(self.valid_frequency_hz),
            "condition_number_limit": self.condition_number_limit,
            "fit_metrics": dict(self.fit_metrics),
        }
        gain_key = (
            "amplifier_gain_v_per_v"
            if self.config_version == 1
            else "voltage_gain_v_per_v"
        )
        data[gain_key] = self.amplifier_gain_v_per_v.tolist()
        return data

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ActuatorCalibration":
        try:
            drive_modes = tuple(
                data.get(
                    "channel_drive_modes",
                    (VOLTAGE_AMPLIFIER,) * 3,
                )
            )
            amplifier_gain = data.get(
                "voltage_gain_v_per_v",
                data.get("amplifier_gain_v_per_v"),
            )
            if amplifier_gain is None:
                if any(
                    str(mode).lower() == VOLTAGE_AMPLIFIER
                    for mode in drive_modes
                ):
                    raise KeyError("amplifier_gain_v_per_v")
                amplifier_gain = [0.0, 0.0, 0.0]
            return cls(
                field_matrix_t_per_a=data["field_matrix_t_per_a"],
                ambient_field_t=data["ambient_field_t"],
                channel_axes=tuple(data["channel_axes"]),
                channel_signs=data["channel_signs"],
                amplifier_gain_v_per_v=amplifier_gain,
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
                channel_drive_modes=drive_modes,
                current_gain_a_per_v=data.get("current_gain_a_per_v"),
                current_driver_bandwidth_hz=data.get(
                    "current_driver_bandwidth_hz"
                ),
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


@dataclass(frozen=True)
class GaussmeterDataset:
    """Offline three-axis gaussmeter samples in SI units.

    ``model_currents_a`` uses the same signed channel order as the columns of
    ``field_matrix_t_per_a``.  If a logger records electrical current instead,
    convert it with ``ActuatorCalibration.electrical_to_model_current`` before
    constructing or exporting this dataset.
    """

    model_currents_a: np.ndarray
    measured_fields_t: np.ndarray
    source: str = ""

    def __post_init__(self) -> None:
        currents = np.array(self.model_currents_a, dtype=np.float64, copy=True)
        fields = np.array(self.measured_fields_t, dtype=np.float64, copy=True)
        if currents.ndim != 2 or currents.shape[1] != 3:
            raise CalibrationError("model_currents_a must have shape (N, 3)")
        if fields.shape != currents.shape:
            raise CalibrationError(
                "measured_fields_t must have the same (N, 3) shape as currents"
            )
        if currents.shape[0] == 0:
            raise CalibrationError("gaussmeter dataset must contain at least one row")
        if not np.all(np.isfinite(currents)) or not np.all(np.isfinite(fields)):
            raise CalibrationError("gaussmeter samples must be finite")
        currents.setflags(write=False)
        fields.setflags(write=False)
        object.__setattr__(self, "model_currents_a", currents)
        object.__setattr__(self, "measured_fields_t", fields)
        object.__setattr__(self, "source", str(self.source))

    @classmethod
    def from_csv(cls, path: str | Path) -> "GaussmeterDataset":
        """Load the stable offline CSV format (field columns in T or gauss)."""

        csv_path = Path(path)
        try:
            with csv_path.open("r", encoding="utf-8-sig", newline="") as file:
                reader = csv.DictReader(file)
                fieldnames = tuple(reader.fieldnames or ())
                missing_current = [
                    name for name in _GAUSSMETER_CURRENT_COLUMNS
                    if name not in fieldnames
                ]
                if missing_current:
                    raise CalibrationError(
                        "gaussmeter CSV is missing column(s): "
                        + ", ".join(missing_current)
                    )
                has_tesla = all(
                    name in fieldnames for name in _GAUSSMETER_FIELD_COLUMNS_T
                )
                has_gauss = all(
                    name in fieldnames for name in _GAUSSMETER_FIELD_COLUMNS_GAUSS
                )
                if has_tesla == has_gauss:
                    raise CalibrationError(
                        "gaussmeter CSV must contain exactly one complete field "
                        "column set: field_[xyz]_t or field_[xyz]_gauss"
                    )
                field_columns = (
                    _GAUSSMETER_FIELD_COLUMNS_T
                    if has_tesla
                    else _GAUSSMETER_FIELD_COLUMNS_GAUSS
                )
                field_scale = 1.0 if has_tesla else 1e-4
                current_rows: list[list[float]] = []
                field_rows: list[list[float]] = []
                for row_number, row in enumerate(reader, start=2):
                    try:
                        current_rows.append(
                            [float(row[name]) for name in _GAUSSMETER_CURRENT_COLUMNS]
                        )
                        field_rows.append(
                            [float(row[name]) * field_scale for name in field_columns]
                        )
                    except (TypeError, ValueError) as exc:
                        raise CalibrationError(
                            f"gaussmeter CSV row {row_number} contains a non-number"
                        ) from exc
        except OSError as exc:
            raise CalibrationError(
                f"cannot read gaussmeter CSV {csv_path}: {exc}"
            ) from exc
        return cls(current_rows, field_rows, source=str(csv_path))

    def to_csv(self, path: str | Path, field_unit: str = "tesla") -> None:
        """Write samples using explicit SI or gauss field column names."""

        normalized_unit = str(field_unit).strip().lower()
        if normalized_unit in ("tesla", "t"):
            field_columns = _GAUSSMETER_FIELD_COLUMNS_T
            field_scale = 1.0
        elif normalized_unit in ("gauss", "g"):
            field_columns = _GAUSSMETER_FIELD_COLUMNS_GAUSS
            field_scale = 1e4
        else:
            raise CalibrationError("field_unit must be 'tesla' or 'gauss'")
        csv_path = Path(path)
        try:
            with csv_path.open("w", encoding="utf-8", newline="") as file:
                writer = csv.writer(file)
                writer.writerow((*_GAUSSMETER_CURRENT_COLUMNS, *field_columns))
                for currents, fields in zip(
                    self.model_currents_a,
                    self.measured_fields_t,
                    strict=True,
                ):
                    writer.writerow((*currents.tolist(), *(fields * field_scale).tolist()))
        except OSError as exc:
            raise CalibrationError(
                f"cannot write gaussmeter CSV {csv_path}: {exc}"
            ) from exc

    def fit_field_matrix(self) -> FieldMatrixFit:
        """Fit the affine center-field model from these samples."""

        return fit_field_matrix(self.model_currents_a, self.measured_fields_t)

    @classmethod
    def from_static_daq_voltages(
        cls,
        calibration: ActuatorCalibration,
        daq_voltages_v: np.ndarray,
        measured_fields_t: np.ndarray,
        *,
        source: str = "",
    ) -> "GaussmeterDataset":
        """Build samples when a calibration run logged static DAQ voltages.

        The configured amplifier/driver DC gains must already be measured.
        This method does not replace direct current measurement when driver
        gain, saturation, or load regulation is uncertain.
        """

        voltages = np.asarray(daq_voltages_v, dtype=np.float64)
        fields = np.asarray(measured_fields_t, dtype=np.float64)
        if voltages.ndim != 2 or voltages.shape[1] != 3:
            raise CalibrationError("daq_voltages_v must have shape (N, 3)")
        model_currents = calibration.steady_model_current_for_daq_voltage(
            voltages.T
        ).T
        return cls(model_currents, fields, source=source)


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


def fit_gaussmeter_csv(path: str | Path) -> FieldMatrixFit:
    """Load an offline gaussmeter CSV and fit its affine field matrix."""

    return GaussmeterDataset.from_csv(path).fit_field_matrix()


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
