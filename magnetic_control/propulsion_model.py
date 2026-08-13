"""Validated monotonic lookup models for rotating-robot propulsion."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

import numpy as np

from .force_command import PropulsionMode


class PropulsionModelError(ValueError):
    """Base class for invalid or unusable propulsion models."""


class UnverifiedPropulsionModelError(PropulsionModelError):
    """Raised when control is attempted with an unverified calibration."""


class InfeasiblePropulsionTargetError(PropulsionModelError):
    """Raised when a target lies outside the calibrated safe range."""

    def __init__(self, requested_si: float, maximum_si: float, unit: str) -> None:
        self.requested_si = float(requested_si)
        self.maximum_si = float(maximum_si)
        self.unit = unit
        super().__init__(
            f"requested {requested_si:.6g} {unit} exceeds calibrated safe "
            f"maximum {maximum_si:.6g} {unit}"
        )


def _readonly_1d(value, name: str, *, minimum_size: int = 2) -> np.ndarray:
    array = np.array(value, dtype=np.float64, copy=True)
    if array.ndim != 1 or array.size < minimum_size:
        raise PropulsionModelError(
            f"{name} must be a one-dimensional array with at least {minimum_size} values"
        )
    if not np.all(np.isfinite(array)):
        raise PropulsionModelError(f"{name} must contain only finite values")
    array.setflags(write=False)
    return array


@dataclass(frozen=True)
class PropulsionInverse:
    """One safe inverse-LUT solution."""

    requested_si: float
    achieved_si: float
    field_magnitude_T: float
    frequency_hz: float
    stepout_lower_hz: float
    safe_frequency_limit_hz: float
    projected: bool

    @property
    def stepout_margin_hz(self) -> float:
        return self.stepout_lower_hz - self.frequency_hz

    @property
    def safe_frequency_margin_hz(self) -> float:
        return self.safe_frequency_limit_hz - self.frequency_hz


@dataclass(frozen=True)
class MonotonicPropulsionLUT:
    """Monotonic output map over field magnitude and rotation frequency.

    Rows of ``output_si`` correspond to ``field_magnitudes_T`` and columns to
    ``frequencies_hz``.  The zero-frequency column must be zero.  Interpolation
    is linear and never extrapolates beyond the calibrated grid.
    """

    mode: PropulsionMode
    field_magnitudes_T: np.ndarray
    frequencies_hz: np.ndarray
    output_si: np.ndarray
    stepout_frequency_lower_hz: np.ndarray
    validated: bool
    stepout_safety_factor: float = 0.75

    def __post_init__(self) -> None:
        try:
            mode = PropulsionMode(self.mode)
        except (TypeError, ValueError) as exc:
            raise PropulsionModelError(f"unsupported propulsion mode: {self.mode!r}") from exc
        object.__setattr__(self, "mode", mode)

        fields = _readonly_1d(self.field_magnitudes_T, "field_magnitudes_T")
        frequencies = _readonly_1d(self.frequencies_hz, "frequencies_hz")
        stepout = _readonly_1d(
            self.stepout_frequency_lower_hz,
            "stepout_frequency_lower_hz",
        )
        output = np.array(self.output_si, dtype=np.float64, copy=True)

        if np.any(fields <= 0.0) or np.any(np.diff(fields) <= 0.0):
            raise PropulsionModelError(
                "field_magnitudes_T must be positive and strictly increasing"
            )
        if frequencies[0] != 0.0 or np.any(np.diff(frequencies) <= 0.0):
            raise PropulsionModelError(
                "frequencies_hz must start at zero and be strictly increasing"
            )
        if stepout.shape != fields.shape or np.any(stepout <= 0.0):
            raise PropulsionModelError(
                "stepout_frequency_lower_hz must have one positive value per field"
            )
        if np.any(np.diff(stepout) < -1e-12):
            raise PropulsionModelError(
                "stepout_frequency_lower_hz must be non-decreasing with field"
            )
        if output.shape != (fields.size, frequencies.size):
            raise PropulsionModelError(
                "output_si shape must be (number of fields, number of frequencies)"
            )
        if not np.all(np.isfinite(output)) or np.any(output < 0.0):
            raise PropulsionModelError("output_si must be finite and non-negative")
        scale = max(float(np.max(output)), np.finfo(np.float64).tiny)
        tolerance = 1e-12 * scale
        if np.any(np.abs(output[:, 0]) > tolerance):
            raise PropulsionModelError("output_si must be zero at zero frequency")
        if np.any(np.diff(output, axis=0) < -tolerance):
            raise PropulsionModelError("output_si must be non-decreasing with field")
        if np.any(np.diff(output, axis=1) < -tolerance):
            raise PropulsionModelError("output_si must be non-decreasing with frequency")

        if not isinstance(self.validated, (bool, np.bool_)):
            raise PropulsionModelError("validated must be boolean")
        safety_factor = float(self.stepout_safety_factor)
        if not np.isfinite(safety_factor) or not 0.0 < safety_factor < 1.0:
            raise PropulsionModelError("stepout_safety_factor must be between zero and one")

        output.setflags(write=False)
        object.__setattr__(self, "field_magnitudes_T", fields)
        object.__setattr__(self, "frequencies_hz", frequencies)
        object.__setattr__(self, "output_si", output)
        object.__setattr__(self, "stepout_frequency_lower_hz", stepout)
        object.__setattr__(self, "validated", bool(self.validated))
        object.__setattr__(self, "stepout_safety_factor", safety_factor)

    @property
    def unit(self) -> str:
        return self.mode.unit

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode.value,
            "field_magnitudes_T": self.field_magnitudes_T.tolist(),
            "frequencies_hz": self.frequencies_hz.tolist(),
            "output_si": self.output_si.tolist(),
            "stepout_frequency_lower_hz": (
                self.stepout_frequency_lower_hz.tolist()
            ),
            "validated": self.validated,
            "stepout_safety_factor": self.stepout_safety_factor,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "MonotonicPropulsionLUT":
        if not isinstance(data, Mapping):
            raise PropulsionModelError("propulsion LUT must be an object")
        try:
            return cls(
                mode=data["mode"],
                field_magnitudes_T=data["field_magnitudes_T"],
                frequencies_hz=data["frequencies_hz"],
                output_si=data["output_si"],
                stepout_frequency_lower_hz=data[
                    "stepout_frequency_lower_hz"
                ],
                validated=data.get("validated", False),
                stepout_safety_factor=data.get("stepout_safety_factor", 0.75),
            )
        except KeyError as exc:
            raise PropulsionModelError(
                f"missing propulsion LUT field: {exc.args[0]}"
            ) from exc

    def _check_field(self, field_magnitude_T: float) -> float:
        field = float(field_magnitude_T)
        if not np.isfinite(field):
            raise PropulsionModelError("field_magnitude_T must be finite")
        if field < self.field_magnitudes_T[0] or field > self.field_magnitudes_T[-1]:
            raise PropulsionModelError("field_magnitude_T is outside the calibrated grid")
        return field

    def stepout_lower_hz(self, field_magnitude_T: float) -> float:
        field = self._check_field(field_magnitude_T)
        return float(
            np.interp(field, self.field_magnitudes_T, self.stepout_frequency_lower_hz)
        )

    def safe_frequency_limit_hz(self, field_magnitude_T: float) -> float:
        return min(
            float(self.frequencies_hz[-1]),
            self.stepout_safety_factor * self.stepout_lower_hz(field_magnitude_T),
        )

    def predict(self, field_magnitude_T: float, frequency_hz: float) -> float:
        field = self._check_field(field_magnitude_T)
        frequency = float(frequency_hz)
        if not np.isfinite(frequency):
            raise PropulsionModelError("frequency_hz must be finite")
        if frequency < 0.0 or frequency > self.frequencies_hz[-1]:
            raise PropulsionModelError("frequency_hz is outside the calibrated grid")

        values_at_frequency = np.array(
            [
                np.interp(frequency, self.frequencies_hz, row)
                for row in self.output_si
            ],
            dtype=np.float64,
        )
        return float(np.interp(field, self.field_magnitudes_T, values_at_frequency))

    def safe_capacity(self, field_magnitude_T: float) -> float:
        field = self._check_field(field_magnitude_T)
        return self.predict(field, self.safe_frequency_limit_hz(field))

    @property
    def maximum_safe_output_si(self) -> float:
        return self.safe_capacity(float(self.field_magnitudes_T[-1]))

    def inverse(
        self,
        target_si: float,
        *,
        project_to_feasible: bool = False,
    ) -> PropulsionInverse:
        """Find the lowest calibrated field that can produce ``target_si``."""

        if not self.validated:
            raise UnverifiedPropulsionModelError(
                "propulsion LUT is not validated for hardware control"
            )
        target = float(target_si)
        if not np.isfinite(target) or target <= 0.0:
            raise PropulsionModelError("target_si must be positive and finite")

        maximum = self.maximum_safe_output_si
        projected = target > maximum
        if projected and not project_to_feasible:
            raise InfeasiblePropulsionTargetError(target, maximum, self.unit)
        effective_target = min(target, maximum)
        if effective_target <= 0.0:
            raise InfeasiblePropulsionTargetError(target, maximum, self.unit)

        field_low = float(self.field_magnitudes_T[0])
        field_high = float(self.field_magnitudes_T[-1])
        if self.safe_capacity(field_low) < effective_target:
            for _ in range(64):
                midpoint = 0.5 * (field_low + field_high)
                if self.safe_capacity(midpoint) >= effective_target:
                    field_high = midpoint
                else:
                    field_low = midpoint
            field = field_high
        else:
            field = field_low

        safe_frequency = self.safe_frequency_limit_hz(field)
        frequency_low = 0.0
        frequency_high = safe_frequency
        for _ in range(64):
            midpoint = 0.5 * (frequency_low + frequency_high)
            if self.predict(field, midpoint) >= effective_target:
                frequency_high = midpoint
            else:
                frequency_low = midpoint
        frequency = frequency_high
        achieved = self.predict(field, frequency)

        return PropulsionInverse(
            requested_si=target,
            achieved_si=achieved,
            field_magnitude_T=field,
            frequency_hz=frequency,
            stepout_lower_hz=self.stepout_lower_hz(field),
            safe_frequency_limit_hz=safe_frequency,
            projected=projected,
        )


@dataclass(frozen=True)
class RobotProfile:
    """Robot-specific direction convention and calibrated propulsion models."""

    profile_id: str
    chirality_sign: int
    magnetization_sign: int
    models: Mapping[PropulsionMode, MonotonicPropulsionLUT]

    def __post_init__(self) -> None:
        profile_id = str(self.profile_id).strip()
        if not profile_id:
            raise PropulsionModelError("profile_id must not be empty")
        object.__setattr__(self, "profile_id", profile_id)

        for name in ("chirality_sign", "magnetization_sign"):
            value = getattr(self, name)
            if isinstance(value, (bool, np.bool_)) or value not in (-1, 1):
                raise PropulsionModelError(f"{name} must be -1 or +1")
            object.__setattr__(self, name, int(value))

        normalized: dict[PropulsionMode, MonotonicPropulsionLUT] = {}
        for raw_mode, model in self.models.items():
            try:
                mode = PropulsionMode(raw_mode)
            except (TypeError, ValueError) as exc:
                raise PropulsionModelError(f"unsupported model key: {raw_mode!r}") from exc
            if not isinstance(model, MonotonicPropulsionLUT):
                raise PropulsionModelError("models must contain MonotonicPropulsionLUT values")
            if model.mode is not mode:
                raise PropulsionModelError("model key does not match model.mode")
            if mode in normalized:
                raise PropulsionModelError(f"duplicate model for {mode.value}")
            normalized[mode] = model
        if not normalized:
            raise PropulsionModelError("models must not be empty")
        object.__setattr__(self, "models", MappingProxyType(normalized))

    @property
    def propulsion_sign(self) -> int:
        return self.chirality_sign * self.magnetization_sign

    def model_for(self, mode: PropulsionMode) -> MonotonicPropulsionLUT | None:
        return self.models.get(PropulsionMode(mode))

    def to_dict(self) -> dict[str, Any]:
        return {
            "profile_version": 1,
            "profile_id": self.profile_id,
            "chirality_sign": self.chirality_sign,
            "magnetization_sign": self.magnetization_sign,
            "models": {
                mode.value: model.to_dict()
                for mode, model in self.models.items()
            },
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "RobotProfile":
        if not isinstance(data, Mapping):
            raise PropulsionModelError("robot profile must be an object")
        raw_version = data.get("profile_version", 1)
        if isinstance(raw_version, (bool, np.bool_)) or not isinstance(
            raw_version, (int, np.integer)
        ):
            raise PropulsionModelError("profile_version must be the integer 1")
        version = int(raw_version)
        if version != 1:
            raise PropulsionModelError(
                f"unsupported robot profile version: {version}"
            )
        try:
            raw_models = data["models"]
            if not isinstance(raw_models, Mapping):
                raise PropulsionModelError("models must be an object")
            models = {
                PropulsionMode(mode): MonotonicPropulsionLUT.from_dict(model)
                for mode, model in raw_models.items()
            }
            return cls(
                profile_id=data["profile_id"],
                chirality_sign=data["chirality_sign"],
                magnetization_sign=data["magnetization_sign"],
                models=models,
            )
        except KeyError as exc:
            raise PropulsionModelError(
                f"missing robot profile field: {exc.args[0]}"
            ) from exc
        except (TypeError, ValueError) as exc:
            if isinstance(exc, PropulsionModelError):
                raise
            raise PropulsionModelError(str(exc)) from exc

    @classmethod
    def from_json(cls, path: str | Path) -> "RobotProfile":
        try:
            with Path(path).open("r", encoding="utf-8") as file:
                data = json.load(file)
        except (OSError, json.JSONDecodeError) as exc:
            raise PropulsionModelError(f"cannot load robot profile: {exc}") from exc
        return cls.from_dict(data)

    def to_json(self, path: str | Path) -> None:
        try:
            with Path(path).open("w", encoding="utf-8") as file:
                json.dump(self.to_dict(), file, indent=2, ensure_ascii=False)
                file.write("\n")
        except OSError as exc:
            raise PropulsionModelError(f"cannot write robot profile: {exc}") from exc
