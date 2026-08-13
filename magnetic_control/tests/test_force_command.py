from __future__ import annotations

import unittest

import numpy as np

from magnetic_control.force_command import (
    PropulsionCommand,
    PropulsionCommandError,
    PropulsionMode,
)


class PropulsionCommandTests(unittest.TestCase):
    def test_modes_report_their_si_units(self):
        self.assertEqual(PropulsionMode.ESTIMATED_PROPULSIVE_COMPONENT.unit, "N")
        self.assertEqual(PropulsionMode.BLOCKED_THRUST.unit, "N")
        self.assertEqual(PropulsionMode.FREE_SWIM_VELOCITY.unit, "m/s")

    def test_command_copies_and_freezes_target(self):
        source = np.array([1e-9, 2e-9, 0.0])
        command = PropulsionCommand(
            source,
            "estimated_propulsive_component",
            "robot-1",
        )
        source[0] = 10.0
        np.testing.assert_allclose(command.target_xyz_si, [1e-9, 2e-9, 0.0])
        self.assertFalse(command.target_xyz_si.flags.writeable)
        self.assertAlmostEqual(command.target_magnitude_si, np.sqrt(5.0) * 1e-9)

    def test_zero_target_is_valid_for_supervisor_stop_semantics(self):
        command = PropulsionCommand(
            [0.0, 0.0, 0.0],
            PropulsionMode.ESTIMATED_PROPULSIVE_COMPONENT,
            "robot-1",
        )
        self.assertEqual(command.target_magnitude_si, 0.0)

    def test_invalid_vector_and_metadata_are_rejected(self):
        with self.assertRaises(PropulsionCommandError):
            PropulsionCommand([1.0, 2.0], "blocked_thrust", "robot-1")
        with self.assertRaises(PropulsionCommandError):
            PropulsionCommand([1.0, np.nan, 0.0], "blocked_thrust", "robot-1")
        with self.assertRaises(PropulsionCommandError):
            PropulsionCommand([1.0, 0.0, 0.0], "unknown", "robot-1")
        with self.assertRaises(PropulsionCommandError):
            PropulsionCommand([1.0, 0.0, 0.0], "blocked_thrust", " ")
        with self.assertRaises(PropulsionCommandError):
            PropulsionCommand(
                [1.0, 0.0, 0.0],
                "blocked_thrust",
                "robot-1",
                contact_id=" ",
            )


if __name__ == "__main__":
    unittest.main()

