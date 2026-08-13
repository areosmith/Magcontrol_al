"""Spatial magnetic-field calibration maps.

All positions are expressed in metres, magnetic fields in tesla, and coil
currents in ampere.  A map query returns the local affine magnetic model

    B = C_BI @ I + B_ambient

at a measured position.  The classes in this module deliberately do not
extrapolate beyond their stated calibration domain.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Protocol, Sequence, TYPE_CHECKING, runtime_checkable

import numpy as np

if TYPE_CHECKING:
    from .calibration import ActuatorCalibration


class SpatialFieldMapError(ValueError):
    """Raised when a spatial map or query is malformed."""


def _readonly_array(
    value: Sequence[float] | np.ndarray,
    *,
    shape: tuple[int, ...],
    name: str,
) -> np.ndarray:
    array = np.array(value, dtype=np.float64, copy=True)
    if array.shape != shape:
        raise SpatialFieldMapError(
            f"{name} must have shape {shape}, got {array.shape}"
        )
    if not np.all(np.isfinite(array)):
        raise SpatialFieldMapError(f"{name} must contain only finite values")
    array.setflags(write=False)
    return array


def _position_array(
    value: Sequence[float] | np.ndarray,
    *,
    dimensions: int,
    name: str = "position_m",
) -> np.ndarray:
    if dimensions not in (2, 3):
        raise SpatialFieldMapError("field maps must have two or three dimensions")
    return _readonly_array(value, shape=(dimensions,), name=name)


@dataclass(frozen=True)
class FieldMapResult:
    """Result of one field-map query.

    Invalid results contain no matrix or background field.  Returned arrays
    are independent, read-only copies so callers cannot alter map data.
    """

    position_m: np.ndarray
    valid: bool
    field_matrix_t_per_a: np.ndarray | None = None
    ambient_field_t: np.ndarray | None = None
    reason: str = ""

    def __post_init__(self) -> None:
        position = np.array(self.position_m, dtype=np.float64, copy=True)
        if position.shape not in ((2,), (3,)):
            raise SpatialFieldMapError(
                "position_m must have shape (2,) or (3,)"
            )
        if not np.all(np.isfinite(position)):
            raise SpatialFieldMapError("position_m must contain only finite values")
        position.setflags(write=False)
        object.__setattr__(self, "position_m", position)

        if not isinstance(self.valid, (bool, np.bool_)):
            raise SpatialFieldMapError("valid must be a boolean")
        object.__setattr__(self, "valid", bool(self.valid))

        if self.valid:
            if self.field_matrix_t_per_a is None or self.ambient_field_t is None:
                raise SpatialFieldMapError(
                    "a valid result requires a field matrix and ambient field"
                )
            matrix = _readonly_array(
                self.field_matrix_t_per_a,
                shape=(3, 3),
                name="field_matrix_t_per_a",
            )
            ambient = _readonly_array(
                self.ambient_field_t,
                shape=(3,),
                name="ambient_field_t",
            )
            object.__setattr__(self, "field_matrix_t_per_a", matrix)
            object.__setattr__(self, "ambient_field_t", ambient)
        elif self.field_matrix_t_per_a is not None or self.ambient_field_t is not None:
            raise SpatialFieldMapError(
                "an invalid result must not provide field calibration values"
            )

        if not isinstance(self.reason, str):
            raise SpatialFieldMapError("reason must be a string")


@runtime_checkable
class SpatialFieldMap(Protocol):
    """Interface implemented by spatial field-model providers."""

    @property
    def dimensions(self) -> int:
        """Number of position coordinates accepted by :meth:`query`."""

    def query(self, position_m: Sequence[float] | np.ndarray) -> FieldMapResult:
        """Return the local affine field model or an invalid result."""


class CenterFieldMap:
    """A constant model valid only near one explicitly calibrated centre.

    ``valid_radius_m`` defaults to zero.  This makes the calibration valid at
    the centre only; users must provide a measured radius before claiming a
    finite local workspace.
    """

    def __init__(
        self,
        field_matrix_t_per_a: Sequence[Sequence[float]] | np.ndarray,
        ambient_field_t: Sequence[float] | np.ndarray,
        *,
        center_position_m: Sequence[float] | np.ndarray = (0.0, 0.0),
        valid_radius_m: float = 0.0,
    ) -> None:
        center = np.array(center_position_m, dtype=np.float64, copy=True)
        if center.shape not in ((2,), (3,)):
            raise SpatialFieldMapError(
                "center_position_m must have shape (2,) or (3,)"
            )
        if not np.all(np.isfinite(center)):
            raise SpatialFieldMapError(
                "center_position_m must contain only finite values"
            )
        center.setflags(write=False)
        self._center_position_m = center
        self._field_matrix_t_per_a = _readonly_array(
            field_matrix_t_per_a,
            shape=(3, 3),
            name="field_matrix_t_per_a",
        )
        self._ambient_field_t = _readonly_array(
            ambient_field_t,
            shape=(3,),
            name="ambient_field_t",
        )
        radius = float(valid_radius_m)
        if not np.isfinite(radius) or radius < 0.0:
            raise SpatialFieldMapError(
                "valid_radius_m must be finite and non-negative"
            )
        self._valid_radius_m = radius

    @classmethod
    def from_calibration(
        cls,
        calibration: ActuatorCalibration,
        *,
        center_position_m: Sequence[float] | np.ndarray = (0.0, 0.0),
        valid_radius_m: float = 0.0,
    ) -> CenterFieldMap:
        """Build a centre map from an actuator calibration."""

        return cls(
            calibration.field_matrix_t_per_a,
            calibration.ambient_field_t,
            center_position_m=center_position_m,
            valid_radius_m=valid_radius_m,
        )

    @property
    def dimensions(self) -> int:
        return int(self._center_position_m.size)

    @property
    def center_position_m(self) -> np.ndarray:
        center = np.array(self._center_position_m, copy=True)
        center.setflags(write=False)
        return center

    @property
    def valid_radius_m(self) -> float:
        return self._valid_radius_m

    def query(self, position_m: Sequence[float] | np.ndarray) -> FieldMapResult:
        position = _position_array(position_m, dimensions=self.dimensions)
        distance = float(np.linalg.norm(position - self._center_position_m))
        tolerance = 16.0 * np.finfo(np.float64).eps * max(
            1.0,
            self._valid_radius_m,
            float(np.linalg.norm(self._center_position_m)),
        )
        if distance > self._valid_radius_m + tolerance:
            return FieldMapResult(
                position_m=position,
                valid=False,
                reason=(
                    "position is outside the explicitly calibrated centre "
                    f"radius ({distance:.6g} m > {self._valid_radius_m:.6g} m)"
                ),
            )
        return FieldMapResult(
            position_m=position,
            valid=True,
            field_matrix_t_per_a=self._field_matrix_t_per_a,
            ambient_field_t=self._ambient_field_t,
        )


class RegularGridFieldMap:
    """Measured 2-D or 3-D regular-grid field map with multilinear interpolation.

    Grid axes must be strictly increasing and use metres.  ``valid_mask`` can
    mark rejected measurement nodes.  A query is valid only when every node
    with non-zero interpolation weight is valid.  No extrapolation is done.
    """

    def __init__(
        self,
        axes_m: Sequence[Sequence[float] | np.ndarray],
        field_matrices_t_per_a: Sequence[float] | np.ndarray,
        ambient_fields_t: Sequence[float] | np.ndarray,
        *,
        valid_mask: Sequence[bool] | np.ndarray | None = None,
    ) -> None:
        if len(axes_m) not in (2, 3):
            raise SpatialFieldMapError("axes_m must contain two or three axes")

        axes: list[np.ndarray] = []
        for index, values in enumerate(axes_m):
            axis = np.array(values, dtype=np.float64, copy=True)
            if axis.ndim != 1 or axis.size < 2:
                raise SpatialFieldMapError(
                    f"axes_m[{index}] must be a one-dimensional array with "
                    "at least two points"
                )
            if not np.all(np.isfinite(axis)):
                raise SpatialFieldMapError(
                    f"axes_m[{index}] must contain only finite values"
                )
            if np.any(np.diff(axis) <= 0.0):
                raise SpatialFieldMapError(
                    f"axes_m[{index}] must be strictly increasing"
                )
            axis.setflags(write=False)
            axes.append(axis)

        grid_shape = tuple(int(axis.size) for axis in axes)
        matrices = np.array(field_matrices_t_per_a, dtype=np.float64, copy=True)
        expected_matrix_shape = grid_shape + (3, 3)
        if matrices.shape != expected_matrix_shape:
            raise SpatialFieldMapError(
                "field_matrices_t_per_a must have shape "
                f"{expected_matrix_shape}, got {matrices.shape}"
            )
        if not np.all(np.isfinite(matrices)):
            raise SpatialFieldMapError(
                "field_matrices_t_per_a must contain only finite values"
            )
        matrices.setflags(write=False)

        ambient = np.array(ambient_fields_t, dtype=np.float64, copy=True)
        expected_ambient_shape = grid_shape + (3,)
        if ambient.shape != expected_ambient_shape:
            raise SpatialFieldMapError(
                f"ambient_fields_t must have shape {expected_ambient_shape}, "
                f"got {ambient.shape}"
            )
        if not np.all(np.isfinite(ambient)):
            raise SpatialFieldMapError(
                "ambient_fields_t must contain only finite values"
            )
        ambient.setflags(write=False)

        if valid_mask is None:
            mask = np.ones(grid_shape, dtype=np.bool_)
        else:
            raw_mask = np.asarray(valid_mask)
            if raw_mask.shape != grid_shape:
                raise SpatialFieldMapError(
                    f"valid_mask must have shape {grid_shape}, got {raw_mask.shape}"
                )
            if not np.issubdtype(raw_mask.dtype, np.bool_):
                raise SpatialFieldMapError(
                    "valid_mask must contain boolean values, without implicit conversion"
                )
            mask = np.array(raw_mask, dtype=np.bool_, copy=True)
        mask.setflags(write=False)

        self._axes_m = tuple(axes)
        self._field_matrices_t_per_a = matrices
        self._ambient_fields_t = ambient
        self._valid_mask = mask

    @property
    def dimensions(self) -> int:
        return len(self._axes_m)

    @property
    def axes_m(self) -> tuple[np.ndarray, ...]:
        copies: list[np.ndarray] = []
        for axis in self._axes_m:
            copied = np.array(axis, copy=True)
            copied.setflags(write=False)
            copies.append(copied)
        return tuple(copies)

    @property
    def valid_mask(self) -> np.ndarray:
        mask = np.array(self._valid_mask, copy=True)
        mask.setflags(write=False)
        return mask

    def query(self, position_m: Sequence[float] | np.ndarray) -> FieldMapResult:
        position = _position_array(position_m, dimensions=self.dimensions)

        lower_indices: list[int] = []
        fractions: list[float] = []
        for coordinate, axis in zip(position, self._axes_m):
            tolerance = 16.0 * np.finfo(np.float64).eps * max(
                1.0,
                abs(float(axis[0])),
                abs(float(axis[-1])),
            )
            if coordinate < axis[0] - tolerance or coordinate > axis[-1] + tolerance:
                return FieldMapResult(
                    position_m=position,
                    valid=False,
                    reason="position is outside the measured field-map bounds",
                )

            bounded = float(np.clip(coordinate, axis[0], axis[-1]))
            if bounded == axis[-1]:
                lower = axis.size - 2
                fraction = 1.0
            else:
                lower = int(np.searchsorted(axis, bounded, side="right") - 1)
                lower = max(0, min(lower, axis.size - 2))
                interval = float(axis[lower + 1] - axis[lower])
                fraction = (bounded - float(axis[lower])) / interval
            lower_indices.append(lower)
            fractions.append(float(np.clip(fraction, 0.0, 1.0)))

        matrix = np.zeros((3, 3), dtype=np.float64)
        ambient = np.zeros(3, dtype=np.float64)
        active_corners: list[tuple[int, ...]] = []

        for corner in product((0, 1), repeat=self.dimensions):
            weight = 1.0
            index: list[int] = []
            for dimension, upper in enumerate(corner):
                fraction = fractions[dimension]
                weight *= fraction if upper else 1.0 - fraction
                index.append(lower_indices[dimension] + upper)
            if weight <= 8.0 * np.finfo(np.float64).eps:
                continue
            grid_index = tuple(index)
            active_corners.append(grid_index)
            if not self._valid_mask[grid_index]:
                return FieldMapResult(
                    position_m=position,
                    valid=False,
                    reason=(
                        "interpolation cell includes a rejected calibration node "
                        f"at index {grid_index}"
                    ),
                )
            matrix += weight * self._field_matrices_t_per_a[grid_index]
            ambient += weight * self._ambient_fields_t[grid_index]

        if not active_corners:
            raise RuntimeError("field-map interpolation produced no active corners")

        return FieldMapResult(
            position_m=position,
            valid=True,
            field_matrix_t_per_a=matrix,
            ambient_field_t=ambient,
        )


__all__ = [
    "CenterFieldMap",
    "FieldMapResult",
    "RegularGridFieldMap",
    "SpatialFieldMap",
    "SpatialFieldMapError",
]
