"""High-level propulsion targets expressed in SI units."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

import numpy as np


class PropulsionMode(str, Enum):
    """Meaning and unit of a propulsion target."""

    ESTIMATED_PROPULSIVE_COMPONENT = "estimated_propulsive_component"
    BLOCKED_THRUST = "blocked_thrust"
    FREE_SWIM_VELOCITY = "free_swim_velocity"

    @property
    def unit(self) -> str:
        if self is PropulsionMode.FREE_SWIM_VELOCITY:
            return "m/s"
        return "N"

    @property
    def is_force(self) -> bool:
        return self is not PropulsionMode.FREE_SWIM_VELOCITY


class PropulsionCommandError(ValueError):
    """Raised when a propulsion command is malformed."""


def _readonly_vector(value: Any) -> np.ndarray:
    vector = np.array(value, dtype=np.float64, copy=True)
    if vector.shape != (3,):
        raise PropulsionCommandError(
            f"target_xyz_si must have shape (3,), got {vector.shape}"
        )
    if not np.all(np.isfinite(vector)):
        raise PropulsionCommandError("target_xyz_si must contain only finite values")
    vector.setflags(write=False)
    return vector


@dataclass(frozen=True)
class PropulsionCommand:
    """Requested axial propulsion vector.

    ``target_xyz_si`` is in newtons for the two force modes and metres per
    second for free-swim velocity.  A zero vector is valid at this layer so a
    supervisor can interpret it as a stop request; a force allocator cannot
    turn it into a :class:`RotatingFieldCommand`.
    """

    target_xyz_si: np.ndarray
    mode: PropulsionMode
    robot_profile_id: str
    frame: str = "lab"
    project_to_feasible: bool = False
    contact_confirmed: bool = False
    contact_id: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "target_xyz_si", _readonly_vector(self.target_xyz_si))

        try:
            mode = PropulsionMode(self.mode)
        except (TypeError, ValueError) as exc:
            raise PropulsionCommandError(f"unsupported propulsion mode: {self.mode!r}") from exc
        object.__setattr__(self, "mode", mode)

        profile_id = str(self.robot_profile_id).strip()
        if not profile_id:
            raise PropulsionCommandError("robot_profile_id must not be empty")
        object.__setattr__(self, "robot_profile_id", profile_id)

        frame = str(self.frame).strip()
        if not frame:
            raise PropulsionCommandError("frame must not be empty")
        object.__setattr__(self, "frame", frame)

        if not isinstance(self.project_to_feasible, (bool, np.bool_)):
            raise PropulsionCommandError("project_to_feasible must be boolean")
        if not isinstance(self.contact_confirmed, (bool, np.bool_)):
            raise PropulsionCommandError("contact_confirmed must be boolean")
        object.__setattr__(self, "project_to_feasible", bool(self.project_to_feasible))
        object.__setattr__(self, "contact_confirmed", bool(self.contact_confirmed))

        if self.contact_id is not None:
            contact_id = str(self.contact_id).strip()
            if not contact_id:
                raise PropulsionCommandError("contact_id must not be blank")
            object.__setattr__(self, "contact_id", contact_id)

    @property
    def target_magnitude_si(self) -> float:
        return float(np.linalg.norm(self.target_xyz_si))

    @property
    def unit(self) -> str:
        return self.mode.unit

