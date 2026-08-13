"""Continuous rotating-field synthesis and voltage-drive calculation."""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Any

import numpy as np

from .calibration import ActuatorCalibration, CalibrationError


class CommandValidationError(ValueError):
    """Raised when a rotating-field command is malformed or out of range."""


class SafetyLimitError(RuntimeError):
    """Raised when a command would exceed a calibrated hardware limit."""


class CalculatorFault(RuntimeError):
    """Raised after a runtime fault has latched the calculator off."""


def _readonly_vector(value: Any, name: str) -> np.ndarray:
    vector = np.array(value, dtype=np.float64, copy=True)
    if vector.shape != (3,):
        raise CommandValidationError(f"{name} must have shape (3,), got {vector.shape}")
    if not np.all(np.isfinite(vector)):
        raise CommandValidationError(f"{name} must contain only finite values")
    vector.setflags(write=False)
    return vector


@dataclass(frozen=True)
class RotatingFieldCommand:
    """Desired circular magnetic field at the calibrated workspace center."""

    axis_xyz: np.ndarray
    magnitude_T: float
    frequency_hz: float
    rotation_sense: int = 1

    def __post_init__(self) -> None:
        axis = _readonly_vector(self.axis_xyz, "axis_xyz")
        norm = float(np.linalg.norm(axis))
        if norm < 1e-12:
            raise CommandValidationError("axis_xyz must be non-zero")
        normalized = np.array(axis / norm, copy=True)
        normalized.setflags(write=False)
        object.__setattr__(self, "axis_xyz", normalized)

        magnitude = float(self.magnitude_T)
        frequency = float(self.frequency_hz)
        if not np.isfinite(magnitude) or magnitude <= 0:
            raise CommandValidationError("magnitude_T must be positive and finite")
        if not np.isfinite(frequency) or frequency <= 0:
            raise CommandValidationError("frequency_hz must be positive and finite")
        if self.rotation_sense not in (-1, 1):
            raise CommandValidationError("rotation_sense must be -1 or +1")
        object.__setattr__(self, "magnitude_T", magnitude)
        object.__setattr__(self, "frequency_hz", frequency)
        object.__setattr__(self, "rotation_sense", int(self.rotation_sense))


@dataclass(frozen=True)
class SignalPreview:
    """Steady-state current and voltage requirements for one command."""

    command: RotatingFieldCommand
    peak_current_a: np.ndarray
    rms_current_a: np.ndarray
    peak_daq_voltage_v: np.ndarray
    rms_daq_voltage_v: np.ndarray
    field_condition_number: float
    max_reconstruction_error_t: float

    def __post_init__(self) -> None:
        for name in (
            "peak_current_a",
            "rms_current_a",
            "peak_daq_voltage_v",
            "rms_daq_voltage_v",
        ):
            array = np.array(getattr(self, name), dtype=np.float64, copy=True)
            array.setflags(write=False)
            object.__setattr__(self, name, array)

    def to_dict(self) -> dict[str, Any]:
        return {
            "axis_xyz": self.command.axis_xyz.tolist(),
            "magnitude_T": self.command.magnitude_T,
            "frequency_hz": self.command.frequency_hz,
            "rotation_sense": self.command.rotation_sense,
            "peak_current_a": self.peak_current_a.tolist(),
            "rms_current_a": self.rms_current_a.tolist(),
            "peak_daq_voltage_v": self.peak_daq_voltage_v.tolist(),
            "rms_daq_voltage_v": self.rms_daq_voltage_v.tolist(),
            "field_condition_number": self.field_condition_number,
            "max_reconstruction_error_t": self.max_reconstruction_error_t,
        }


@dataclass(frozen=True)
class CalculatorStatus:
    """Thread-safe snapshot of the calculator output state."""

    active: bool
    axis_xyz: np.ndarray
    magnitude_T: float
    frequency_hz: float
    rotation_sense: int
    phase_rad: float
    activation: float
    transition_remaining_s: float
    electrical_current_setpoint_a: np.ndarray
    last_peak_daq_voltage_v: np.ndarray
    fault: str | None

    def __post_init__(self) -> None:
        for name in ("axis_xyz", "electrical_current_setpoint_a", "last_peak_daq_voltage_v"):
            array = np.array(getattr(self, name), dtype=np.float64, copy=True)
            array.setflags(write=False)
            object.__setattr__(self, name, array)

    def to_dict(self) -> dict[str, Any]:
        return {
            "active": self.active,
            "axis_xyz": self.axis_xyz.tolist(),
            "magnitude_T": self.magnitude_T,
            "frequency_hz": self.frequency_hz,
            "rotation_sense": self.rotation_sense,
            "phase_rad": self.phase_rad,
            "activation": self.activation,
            "transition_remaining_s": self.transition_remaining_s,
            "electrical_current_setpoint_a": (
                self.electrical_current_setpoint_a.tolist()
            ),
            "last_peak_daq_voltage_v": self.last_peak_daq_voltage_v.tolist(),
            "fault": self.fault,
        }


@dataclass
class _Transition:
    start_q: np.ndarray
    target_q: np.ndarray
    start_magnitude_t: float
    target_magnitude_t: float
    start_signed_frequency_hz: float
    target_signed_frequency_hz: float
    start_activation: float
    target_activation: float
    duration_s: float
    elapsed_s: float = 0.0


@dataclass(frozen=True)
class _StateSnapshot:
    frame_q: np.ndarray
    phase_rad: float
    magnitude_t: float
    signed_frequency_hz: float
    activation: float


def _smoothstep(value: np.ndarray) -> np.ndarray:
    clipped = np.clip(value, 0.0, 1.0)
    return clipped * clipped * (3.0 - 2.0 * clipped)


def _quaternion_normalize(quaternion: np.ndarray) -> np.ndarray:
    result = np.asarray(quaternion, dtype=np.float64)
    norm = np.linalg.norm(result, axis=-1, keepdims=True)
    if np.any(norm < 1e-15):
        raise CommandValidationError("invalid zero quaternion")
    return result / norm


def _quaternion_multiply(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    lw, lx, ly, lz = left
    rw, rx, ry, rz = right
    return np.array(
        [
            lw * rw - lx * rx - ly * ry - lz * rz,
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
        ],
        dtype=np.float64,
    )


def _matrix_to_quaternion(matrix: np.ndarray) -> np.ndarray:
    """Convert a proper rotation matrix to a [w, x, y, z] quaternion."""

    matrix = np.asarray(matrix, dtype=np.float64)
    trace = float(np.trace(matrix))
    if trace > 0:
        scale = np.sqrt(trace + 1.0) * 2.0
        quaternion = np.array(
            [
                0.25 * scale,
                (matrix[2, 1] - matrix[1, 2]) / scale,
                (matrix[0, 2] - matrix[2, 0]) / scale,
                (matrix[1, 0] - matrix[0, 1]) / scale,
            ]
        )
    else:
        diagonal = np.diag(matrix)
        index = int(np.argmax(diagonal))
        if index == 0:
            scale = np.sqrt(1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2]) * 2.0
            quaternion = np.array(
                [
                    (matrix[2, 1] - matrix[1, 2]) / scale,
                    0.25 * scale,
                    (matrix[0, 1] + matrix[1, 0]) / scale,
                    (matrix[0, 2] + matrix[2, 0]) / scale,
                ]
            )
        elif index == 1:
            scale = np.sqrt(1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2]) * 2.0
            quaternion = np.array(
                [
                    (matrix[0, 2] - matrix[2, 0]) / scale,
                    (matrix[0, 1] + matrix[1, 0]) / scale,
                    0.25 * scale,
                    (matrix[1, 2] + matrix[2, 1]) / scale,
                ]
            )
        else:
            scale = np.sqrt(1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1]) * 2.0
            quaternion = np.array(
                [
                    (matrix[1, 0] - matrix[0, 1]) / scale,
                    (matrix[0, 2] + matrix[2, 0]) / scale,
                    (matrix[1, 2] + matrix[2, 1]) / scale,
                    0.25 * scale,
                ]
            )
    return _quaternion_normalize(quaternion)


def _quaternion_to_matrix(quaternion: np.ndarray) -> np.ndarray:
    w, x, y, z = _quaternion_normalize(quaternion)
    return np.array(
        [
            [
                1.0 - 2.0 * (y * y + z * z),
                2.0 * (x * y - z * w),
                2.0 * (x * z + y * w),
            ],
            [
                2.0 * (x * y + z * w),
                1.0 - 2.0 * (x * x + z * z),
                2.0 * (y * z - x * w),
            ],
            [
                2.0 * (x * z - y * w),
                2.0 * (y * z + x * w),
                1.0 - 2.0 * (x * x + y * y),
            ],
        ],
        dtype=np.float64,
    )


def _quaternion_slerp(
    start: np.ndarray,
    target: np.ndarray,
    fraction: np.ndarray,
) -> np.ndarray:
    """Vectorized shortest-path quaternion SLERP."""

    q0 = _quaternion_normalize(start)
    q1 = _quaternion_normalize(target)
    dot = float(np.dot(q0, q1))
    if dot < 0:
        q1 = -q1
        dot = -dot
    dot = float(np.clip(dot, -1.0, 1.0))

    fraction = np.asarray(fraction, dtype=np.float64).reshape(-1, 1)
    if dot > 0.9995:
        result = (1.0 - fraction) * q0 + fraction * q1
        return _quaternion_normalize(result)

    angle = np.arccos(dot)
    denominator = np.sin(angle)
    start_weight = np.sin((1.0 - fraction) * angle) / denominator
    target_weight = np.sin(fraction * angle) / denominator
    return _quaternion_normalize(start_weight * q0 + target_weight * q1)


def _frame_for_axis(axis_xyz: np.ndarray) -> np.ndarray:
    """Create a deterministic right-handed [u, v, n] rotation frame."""

    axis = np.asarray(axis_xyz, dtype=np.float64)
    axis = axis / np.linalg.norm(axis)
    if abs(axis[2]) > 0.9:
        reference = np.array([1.0, 0.0, 0.0])
    else:
        reference = np.array([0.0, 0.0, 1.0])
    u_axis = reference - axis * np.dot(reference, axis)
    u_axis /= np.linalg.norm(u_axis)
    v_axis = np.cross(axis, u_axis)
    return np.column_stack((u_axis, v_axis, axis))


def _target_frame_quaternion(
    current_quaternion: np.ndarray,
    target_axis_xyz: np.ndarray,
) -> np.ndarray:
    """Apply the minimum rotation that aligns the current axis to a new axis."""

    current_matrix = _quaternion_to_matrix(current_quaternion)
    current_axis = current_matrix[:, 2]
    current_u = current_matrix[:, 0]
    target_axis = np.asarray(target_axis_xyz, dtype=np.float64)
    target_axis = target_axis / np.linalg.norm(target_axis)

    cosine = float(np.clip(np.dot(current_axis, target_axis), -1.0, 1.0))
    if cosine > 1.0 - 1e-12:
        return np.array(current_quaternion, copy=True)
    if cosine < -1.0 + 1e-12:
        delta = np.array([0.0, *current_u], dtype=np.float64)
    else:
        cross = np.cross(current_axis, target_axis)
        sine = float(np.linalg.norm(cross))
        rotation_axis = cross / sine
        angle = np.arctan2(sine, cosine)
        delta = np.array(
            [
                np.cos(angle / 2.0),
                *(rotation_axis * np.sin(angle / 2.0)),
            ],
            dtype=np.float64,
        )
    return _quaternion_normalize(
        _quaternion_multiply(delta, current_quaternion)
    )


def _quaternion_basis(quaternions: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return u and v columns for an array of [w, x, y, z] quaternions."""

    q = _quaternion_normalize(quaternions)
    w, x, y, z = (q[:, index] for index in range(4))
    u_axis = np.column_stack(
        (
            1.0 - 2.0 * (y * y + z * z),
            2.0 * (x * y + z * w),
            2.0 * (x * z - y * w),
        )
    )
    v_axis = np.column_stack(
        (
            2.0 * (x * y - z * w),
            1.0 - 2.0 * (x * x + z * z),
            2.0 * (y * z + x * w),
        )
    )
    return u_axis, v_axis


class RotatingFieldCalculator:
    """Convert rotating-field commands into safe three-channel DAQ voltages."""

    def __init__(
        self,
        calibration: ActuatorCalibration,
        sample_rate_hz: float,
        *,
        allow_unverified: bool = False,
        default_transition_s: float = 0.25,
        max_transition_s: float = 2.0,
    ) -> None:
        if not isinstance(calibration, ActuatorCalibration):
            raise TypeError("calibration must be an ActuatorCalibration")
        calibration.assert_ready(require_verified=not allow_unverified)

        sample_rate_hz = float(sample_rate_hz)
        if not np.isfinite(sample_rate_hz) or sample_rate_hz <= 0:
            raise ValueError("sample_rate_hz must be positive and finite")
        default_transition_s = float(default_transition_s)
        max_transition_s = float(max_transition_s)
        if not np.isfinite(default_transition_s) or default_transition_s <= 0:
            raise ValueError("default_transition_s must be positive and finite")
        if not np.isfinite(max_transition_s) or max_transition_s <= 0:
            raise ValueError("max_transition_s must be positive and finite")
        if default_transition_s > max_transition_s:
            raise ValueError(
                "default_transition_s must not exceed max_transition_s"
            )

        self.calibration = calibration
        self.sample_rate_hz = sample_rate_hz
        self.dt_s = 1.0 / sample_rate_hz
        self.default_transition_s = default_transition_s
        self.max_transition_s = max_transition_s

        self._inverse_field_matrix = calibration.inverse_field_matrix
        self._lock = threading.RLock()
        self._frame_q = _matrix_to_quaternion(
            _frame_for_axis(np.array([0.0, 0.0, 1.0]))
        )
        self._phase_rad = 0.0
        self._magnitude_t = 0.0
        self._signed_frequency_hz = 0.0
        self._activation = 0.0
        self._transition: _Transition | None = None
        self._target_command: RotatingFieldCommand | None = None
        self._fault_reason: str | None = None
        self._electrical_current_setpoint_a = np.zeros(3, dtype=np.float64)
        self._last_peak_voltage_v = np.zeros(3, dtype=np.float64)

    def _validate_command_range(self, command: RotatingFieldCommand) -> None:
        if not isinstance(command, RotatingFieldCommand):
            raise TypeError("command must be a RotatingFieldCommand")
        minimum, maximum = self.calibration.valid_frequency_hz
        if command.frequency_hz < minimum or command.frequency_hz > maximum:
            raise CommandValidationError(
                f"frequency {command.frequency_hz:g} Hz is outside calibrated "
                f"range [{minimum:g}, {maximum:g}] Hz"
            )

    def preview(self, command: RotatingFieldCommand) -> SignalPreview:
        """Validate and report steady-state requirements without changing state."""

        self._validate_command_range(command)
        frame = _frame_for_axis(command.axis_xyz)
        u_axis = frame[:, 0]
        v_axis = frame[:, 1]
        signed_omega = (
            command.rotation_sense * 2.0 * np.pi * command.frequency_hz
        )

        dc_model = self._inverse_field_matrix @ (
            -self.calibration.ambient_field_t
        )
        cosine_model = self._inverse_field_matrix @ (
            command.magnitude_T * u_axis
        )
        sine_model = self._inverse_field_matrix @ (
            command.magnitude_T * v_axis
        )

        signs = self.calibration.channel_signs
        dc_current = signs * dc_model
        cosine_current = signs * cosine_model
        sine_current = signs * sine_model

        dc_voltage, cosine_voltage, sine_voltage = (
            self.calibration.daq_voltage_components_for_current(
                dc_current,
                cosine_current,
                sine_current,
                signed_frequency_hz=(signed_omega / (2.0 * np.pi)),
            )
        )

        current_ac_amplitude = np.hypot(cosine_current, sine_current)
        voltage_ac_amplitude = np.hypot(cosine_voltage, sine_voltage)
        peak_current = np.abs(dc_current) + current_ac_amplitude
        rms_current = np.sqrt(dc_current**2 + current_ac_amplitude**2 / 2.0)
        peak_voltage = np.abs(dc_voltage) + voltage_ac_amplitude
        rms_voltage = np.sqrt(dc_voltage**2 + voltage_ac_amplitude**2 / 2.0)

        reconstructed_dc = (
            self.calibration.field_matrix_t_per_a @ dc_model
            + self.calibration.ambient_field_t
        )
        reconstructed_cosine = (
            self.calibration.field_matrix_t_per_a @ cosine_model
        )
        reconstructed_sine = (
            self.calibration.field_matrix_t_per_a @ sine_model
        )
        errors = (
            np.linalg.norm(reconstructed_dc),
            np.linalg.norm(
                reconstructed_cosine - command.magnitude_T * u_axis
            ),
            np.linalg.norm(
                reconstructed_sine - command.magnitude_T * v_axis
            ),
        )
        preview = SignalPreview(
            command=command,
            peak_current_a=peak_current,
            rms_current_a=rms_current,
            peak_daq_voltage_v=peak_voltage,
            rms_daq_voltage_v=rms_voltage,
            field_condition_number=self.calibration.condition_number,
            max_reconstruction_error_t=float(max(errors)),
        )
        self._validate_preview_limits(preview)
        return preview

    def _validate_preview_limits(self, preview: SignalPreview) -> None:
        messages: list[str] = []
        for channel in range(3):
            peak_limit = self.calibration.peak_current_limit_a[channel]
            rms_limit = self.calibration.rms_current_limit_a[channel]
            voltage_limit = self.calibration.daq_voltage_limit_v[channel]
            if preview.peak_current_a[channel] > peak_limit * (1.0 + 1e-12):
                messages.append(
                    f"ch{channel} peak current "
                    f"{preview.peak_current_a[channel]:.6g} A > {peak_limit:.6g} A"
                )
            if preview.rms_current_a[channel] > rms_limit * (1.0 + 1e-12):
                messages.append(
                    f"ch{channel} RMS current "
                    f"{preview.rms_current_a[channel]:.6g} A > {rms_limit:.6g} A"
                )
            if preview.peak_daq_voltage_v[channel] > voltage_limit * (1.0 + 1e-12):
                messages.append(
                    f"ch{channel} DAQ voltage "
                    f"{preview.peak_daq_voltage_v[channel]:.6g} V > "
                    f"{voltage_limit:.6g} V"
                )
        if messages:
            raise SafetyLimitError("; ".join(messages))

    def _parameter_arrays(
        self,
        points: int,
        transition: _Transition | None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if transition is None:
            quaternions = np.repeat(self._frame_q[None, :], points, axis=0)
            magnitude = np.full(points, self._magnitude_t)
            frequency = np.full(points, self._signed_frequency_hz)
            activation = np.full(points, self._activation)
            return quaternions, magnitude, frequency, activation

        times = transition.elapsed_s + np.arange(points) * self.dt_s
        fraction = _smoothstep(times / transition.duration_s)
        quaternions = _quaternion_slerp(
            transition.start_q,
            transition.target_q,
            fraction,
        )
        magnitude = (
            transition.start_magnitude_t
            + fraction
            * (
                transition.target_magnitude_t
                - transition.start_magnitude_t
            )
        )
        frequency = (
            transition.start_signed_frequency_hz
            + fraction
            * (
                transition.target_signed_frequency_hz
                - transition.start_signed_frequency_hz
            )
        )
        activation = (
            transition.start_activation
            + fraction
            * (
                transition.target_activation
                - transition.start_activation
            )
        )
        return quaternions, magnitude, frequency, activation

    def _build_waveforms(
        self,
        samples: int,
        transition: _Transition | None,
        phase_start_rad: float,
    ) -> tuple[
        np.ndarray,
        np.ndarray,
        float,
        np.ndarray,
        float,
        float,
        float,
    ]:
        points = samples + 1
        quaternions, magnitude, frequency, activation = self._parameter_arrays(
            points,
            transition,
        )
        phase_increment = (
            2.0
            * np.pi
            * 0.5
            * (frequency[:-1] + frequency[1:])
            * self.dt_s
        )
        phases = np.empty(points, dtype=np.float64)
        phases[0] = phase_start_rad
        phases[1:] = phase_start_rad + np.cumsum(phase_increment)

        u_axis, v_axis = _quaternion_basis(quaternions)
        rotating_field = magnitude[:, None] * (
            u_axis * np.cos(phases)[:, None]
            + v_axis * np.sin(phases)[:, None]
        )
        coil_field = activation[:, None] * (
            rotating_field - self.calibration.ambient_field_t[None, :]
        )
        model_current = self._inverse_field_matrix @ coil_field.T
        electrical_current = self.calibration.model_to_electrical_current(
            model_current
        )
        voltage = self.calibration.daq_voltage_for_current_step(
            electrical_current[:, :-1],
            electrical_current[:, 1:],
            self.dt_s,
        )
        return (
            voltage,
            electrical_current,
            float(phases[-1] % (2.0 * np.pi)),
            quaternions[-1],
            float(magnitude[-1]),
            float(frequency[-1]),
            float(activation[-1]),
        )

    def _transient_limit_error(
        self,
        electrical_current: np.ndarray,
        voltage: np.ndarray,
    ) -> str | None:
        if not np.all(np.isfinite(electrical_current)) or not np.all(
            np.isfinite(voltage)
        ):
            return "generated current or voltage contains a non-finite value"
        current_peak = np.max(np.abs(electrical_current), axis=1)
        voltage_peak = np.max(np.abs(voltage), axis=1)
        messages: list[str] = []
        for channel in range(3):
            current_limit = self.calibration.peak_current_limit_a[channel]
            voltage_limit = self.calibration.daq_voltage_limit_v[channel]
            if current_peak[channel] > current_limit * (1.0 + 1e-12):
                messages.append(
                    f"ch{channel} transient current {current_peak[channel]:.6g} A "
                    f"> {current_limit:.6g} A"
                )
            if voltage_peak[channel] > voltage_limit * (1.0 + 1e-12):
                messages.append(
                    f"ch{channel} transient voltage {voltage_peak[channel]:.6g} V "
                    f"> {voltage_limit:.6g} V"
                )
        return "; ".join(messages) if messages else None

    def _snapshot_locked(self) -> _StateSnapshot:
        return _StateSnapshot(
            frame_q=np.array(self._frame_q, copy=True),
            phase_rad=self._phase_rad,
            magnitude_t=self._magnitude_t,
            signed_frequency_hz=self._signed_frequency_hz,
            activation=self._activation,
        )

    @staticmethod
    def _make_transition(
        state: _StateSnapshot,
        target_q: np.ndarray,
        target_magnitude_t: float,
        target_signed_frequency_hz: float,
        target_activation: float,
        duration_s: float,
    ) -> _Transition:
        return _Transition(
            start_q=np.array(state.frame_q, copy=True),
            target_q=np.array(target_q, copy=True),
            start_magnitude_t=state.magnitude_t,
            target_magnitude_t=float(target_magnitude_t),
            start_signed_frequency_hz=state.signed_frequency_hz,
            target_signed_frequency_hz=float(target_signed_frequency_hz),
            start_activation=state.activation,
            target_activation=float(target_activation),
            duration_s=float(duration_s),
        )

    def _select_safe_transition(
        self,
        state: _StateSnapshot,
        target_q: np.ndarray,
        target_magnitude_t: float,
        target_signed_frequency_hz: float,
        target_activation: float,
        requested_s: float,
    ) -> float:
        duration = max(float(requested_s), self.dt_s)
        if not np.isfinite(duration) or duration <= 0:
            raise CommandValidationError("transition_s must be positive and finite")
        if duration > self.max_transition_s:
            raise CommandValidationError(
                f"transition_s exceeds maximum {self.max_transition_s:g} s"
            )

        last_error = ""
        while True:
            transition = self._make_transition(
                state,
                target_q,
                target_magnitude_t,
                target_signed_frequency_hz,
                target_activation,
                duration,
            )
            samples = max(1, int(np.ceil(duration * self.sample_rate_hz)))
            voltage, current, *_ = self._build_waveforms(
                samples,
                transition,
                state.phase_rad,
            )
            last_error = self._transient_limit_error(current, voltage) or ""
            if not last_error:
                return duration
            if duration >= self.max_transition_s - 1e-12:
                break
            duration = min(
                self.max_transition_s,
                max(duration * 1.5, duration + 0.05),
            )
        raise SafetyLimitError(
            "no safe transition within "
            f"{self.max_transition_s:g} s: {last_error}"
        )

    def set_command(
        self,
        command: RotatingFieldCommand,
        transition_s: float | None = None,
    ) -> CalculatorStatus:
        """Schedule a phase-continuous, limit-checked transition."""

        self.preview(command)
        requested = (
            self.default_transition_s
            if transition_s is None
            else float(transition_s)
        )
        with self._lock:
            if self._fault_reason is not None:
                raise CalculatorFault(
                    f"calculator is faulted: {self._fault_reason}"
                )
            state = self._snapshot_locked()

        target_q = _target_frame_quaternion(state.frame_q, command.axis_xyz)
        signed_frequency = command.rotation_sense * command.frequency_hz
        duration = self._select_safe_transition(
            state=state,
            target_q=target_q,
            target_magnitude_t=command.magnitude_T,
            target_signed_frequency_hz=signed_frequency,
            target_activation=1.0,
            requested_s=requested,
        )

        with self._lock:
            if self._fault_reason is not None:
                raise CalculatorFault(
                    f"calculator is faulted: {self._fault_reason}"
                )
            current_state = self._snapshot_locked()
            current_target_q = _target_frame_quaternion(
                current_state.frame_q,
                command.axis_xyz,
            )
            self._transition = self._make_transition(
                current_state,
                current_target_q,
                command.magnitude_T,
                signed_frequency,
                1.0,
                duration,
            )
            self._target_command = command
            return self._status_locked()

    def stop(self, transition_s: float | None = None) -> CalculatorStatus:
        """Ramp all requested coil currents to zero."""

        requested = (
            self.default_transition_s
            if transition_s is None
            else float(transition_s)
        )
        with self._lock:
            if self._fault_reason is not None:
                return self._status_locked()
            if self._activation <= 1e-15 and self._transition is None:
                self._magnitude_t = 0.0
                self._signed_frequency_hz = 0.0
                self._electrical_current_setpoint_a.fill(0.0)
                self._target_command = None
                return self._status_locked()
            state = self._snapshot_locked()

        duration = self._select_safe_transition(
            state=state,
            target_q=state.frame_q,
            target_magnitude_t=0.0,
            target_signed_frequency_hz=0.0,
            target_activation=0.0,
            requested_s=requested,
        )

        with self._lock:
            if self._fault_reason is not None:
                return self._status_locked()
            current_state = self._snapshot_locked()
            self._transition = self._make_transition(
                current_state,
                current_state.frame_q,
                0.0,
                0.0,
                0.0,
                duration,
            )
            self._target_command = None
            return self._status_locked()

    def generate_voltage_block(self, samples: int) -> np.ndarray:
        """Generate a ``(3, samples)`` float64 voltage block for the DAQ."""

        if isinstance(samples, bool) or not isinstance(samples, (int, np.integer)):
            raise TypeError("samples must be an integer")
        samples = int(samples)
        if samples <= 0:
            raise ValueError("samples must be positive")

        with self._lock:
            if self._fault_reason is not None:
                raise CalculatorFault(
                    f"calculator is faulted: {self._fault_reason}"
                )
            try:
                (
                    voltage,
                    electrical_current,
                    phase_end,
                    frame_end,
                    magnitude_end,
                    frequency_end,
                    activation_end,
                ) = self._build_waveforms(
                    samples,
                    self._transition,
                    self._phase_rad,
                )
                limit_error = self._transient_limit_error(
                    electrical_current,
                    voltage,
                )
                if limit_error:
                    raise SafetyLimitError(limit_error)
            except Exception as exc:
                self._fault_reason = str(exc)
                raise

            self._phase_rad = phase_end
            self._frame_q = np.array(frame_end, copy=True)
            self._magnitude_t = magnitude_end
            self._signed_frequency_hz = frequency_end
            self._activation = activation_end
            self._electrical_current_setpoint_a = np.array(
                electrical_current[:, -1],
                copy=True,
            )
            self._last_peak_voltage_v = np.max(np.abs(voltage), axis=1)

            if self._transition is not None:
                self._transition.elapsed_s += samples * self.dt_s
                if (
                    self._transition.elapsed_s
                    >= self._transition.duration_s - 0.5 * self.dt_s
                ):
                    target = self._transition
                    self._frame_q = np.array(target.target_q, copy=True)
                    self._magnitude_t = target.target_magnitude_t
                    self._signed_frequency_hz = target.target_signed_frequency_hz
                    self._activation = target.target_activation
                    self._transition = None
                    if self._activation <= 1e-15:
                        self._magnitude_t = 0.0
                        self._signed_frequency_hz = 0.0
                        self._electrical_current_setpoint_a.fill(0.0)

            return np.ascontiguousarray(voltage, dtype=np.float64)

    def status(self) -> CalculatorStatus:
        with self._lock:
            return self._status_locked()

    def _status_locked(self) -> CalculatorStatus:
        matrix = _quaternion_to_matrix(self._frame_q)
        remaining = 0.0
        if self._transition is not None:
            remaining = max(
                0.0,
                self._transition.duration_s - self._transition.elapsed_s,
            )
        if abs(self._signed_frequency_hz) > 1e-15:
            rotation_sense = 1 if self._signed_frequency_hz > 0 else -1
        elif self._target_command is not None:
            rotation_sense = self._target_command.rotation_sense
        else:
            rotation_sense = 1
        active = bool(
            self._activation > 1e-12
            or (
                self._transition is not None
                and self._transition.target_activation > 1e-12
            )
        )
        return CalculatorStatus(
            active=active,
            axis_xyz=matrix[:, 2],
            magnitude_T=self._magnitude_t,
            frequency_hz=abs(self._signed_frequency_hz),
            rotation_sense=rotation_sense,
            phase_rad=self._phase_rad,
            activation=self._activation,
            transition_remaining_s=remaining,
            electrical_current_setpoint_a=self._electrical_current_setpoint_a,
            last_peak_daq_voltage_v=self._last_peak_voltage_v,
            fault=self._fault_reason,
        )

    def clear_fault(self) -> None:
        """Reset the calculator to zero; call only while DAQ output is stopped."""

        with self._lock:
            self._fault_reason = None
            self._transition = None
            self._target_command = None
            self._phase_rad = 0.0
            self._magnitude_t = 0.0
            self._signed_frequency_hz = 0.0
            self._activation = 0.0
            self._electrical_current_setpoint_a.fill(0.0)
            self._last_peak_voltage_v.fill(0.0)
            self._frame_q = _matrix_to_quaternion(
                _frame_for_axis(np.array([0.0, 0.0, 1.0]))
            )
