from __future__ import annotations

import unittest

import numpy as np

from magnetic_control import ActuatorCalibration, RotatingFieldCalculator
from magnetic_control.force_allocator import AllocationReason, ForceAllocator
from magnetic_control.force_command import PropulsionCommand, PropulsionMode
from magnetic_control.propulsion_model import MonotonicPropulsionLUT, RobotProfile
from magnetic_control.supervisor import MagneticControlSupervisor


def make_supervisor() -> MagneticControlSupervisor:
    calibration = ActuatorCalibration(
        field_matrix_t_per_a=np.diag([0.01, 0.012, 0.009]),
        ambient_field_t=np.zeros(3),
        channel_axes=("x", "y", "z"),
        channel_signs=np.ones(3),
        amplifier_gain_v_per_v=np.array([4.0, 5.0, 6.0]),
        coil_resistance_ohm=np.array([2.0, 2.5, 3.0]),
        coil_inductance_h=np.array([0.01, 0.012, 0.008]),
        daq_voltage_limit_v=np.ones(3) * 10.0,
        peak_current_limit_a=np.ones(3) * 2.0,
        rms_current_limit_a=np.ones(3),
        valid_frequency_hz=(0.0, 100.0),
        verified=True,
    )
    calculator = RotatingFieldCalculator(
        calibration,
        sample_rate_hz=1000.0,
        default_transition_s=0.05,
        max_transition_s=0.25,
    )
    mode = PropulsionMode.ESTIMATED_PROPULSIVE_COMPONENT
    model = MonotonicPropulsionLUT(
        mode=mode,
        field_magnitudes_T=[1e-3, 2e-3],
        frequencies_hz=[0.0, 10.0, 20.0],
        output_si=[[0.0, 1e-9, 2e-9], [0.0, 2e-9, 4e-9]],
        stepout_frequency_lower_hz=[20.0, 30.0],
        validated=True,
    )
    profile = RobotProfile(
        profile_id="helix-001",
        chirality_sign=1,
        magnetization_sign=1,
        models={mode: model},
    )
    return MagneticControlSupervisor(ForceAllocator(), profile, calculator)


def command(force) -> PropulsionCommand:
    return PropulsionCommand(
        force,
        PropulsionMode.ESTIMATED_PROPULSIVE_COMPONENT,
        "helix-001",
    )


class SupervisorTests(unittest.TestCase):
    def test_preview_does_not_activate_calculator(self):
        supervisor = make_supervisor()
        result = supervisor.preview_force(command([1e-9, 0.0, 0.0]))
        self.assertTrue(result.accepted, result.message)
        self.assertIsNotNone(result.preview)
        self.assertFalse(supervisor.status().active)

    def test_set_force_generates_daq_compatible_block(self):
        supervisor = make_supervisor()
        result = supervisor.set_force(
            command([0.0, 1e-9, 0.0]),
            transition_s=0.05,
        )
        self.assertTrue(result.accepted, result.message)
        block = supervisor.generate_voltage_block(100)
        self.assertEqual(block.shape, (3, 100))
        self.assertEqual(block.dtype, np.float64)
        self.assertTrue(block.flags.c_contiguous)

    def test_zero_target_uses_calculator_stop(self):
        supervisor = make_supervisor()
        self.assertTrue(
            supervisor.set_force(
                command([1e-9, 0.0, 0.0]), transition_s=0.05
            ).accepted
        )
        supervisor.generate_voltage_block(100)
        stopped = supervisor.set_force(
            command([0.0, 0.0, 0.0]),
            transition_s=0.05,
        )
        self.assertTrue(stopped.accepted, stopped.message)
        supervisor.generate_voltage_block(100)
        np.testing.assert_array_equal(
            supervisor.generate_voltage_block(10),
            np.zeros((3, 10)),
        )

    def test_small_nonzero_target_is_not_misreported_as_stop(self):
        supervisor = make_supervisor()
        result = supervisor.preview_force(command([1e-19, 0.0, 0.0]))
        self.assertTrue(result.accepted, result.message)
        self.assertIsNotNone(result.preview)
        self.assertIsNotNone(result.allocation.rotating_field_command)
        self.assertGreater(
            result.allocation.rotating_field_command.frequency_hz,
            0.0,
        )
        self.assertFalse(supervisor.status().active)

    def test_zero_target_stops_even_with_stale_profile_metadata(self):
        supervisor = make_supervisor()
        stale = PropulsionCommand(
            [0.0, 0.0, 0.0],
            PropulsionMode.ESTIMATED_PROPULSIVE_COMPONENT,
            "stale-profile-id",
        )
        result = supervisor.set_force(stale)
        self.assertTrue(result.accepted)
        self.assertEqual(
            result.allocation.reasons,
            (AllocationReason.ZERO_TARGET_REQUIRES_STOP,),
        )

    def test_infeasible_command_does_not_change_state(self):
        supervisor = make_supervisor()
        result = supervisor.set_force(command([20e-9, 0.0, 0.0]))
        self.assertFalse(result.accepted)
        self.assertFalse(supervisor.status().active)


if __name__ == "__main__":
    unittest.main()
