from __future__ import annotations

import unittest

import numpy as np

from magnetic_control import (
    ActuatorCalibration,
    CalibrationError,
    CommandValidationError,
    RotatingFieldCalculator,
    RotatingFieldCommand,
    SafetyLimitError,
)


def make_calibration(**overrides) -> ActuatorCalibration:
    values = {
        "field_matrix_t_per_a": np.eye(3) * 0.01,
        "ambient_field_t": np.zeros(3),
        "channel_axes": ("x", "y", "z"),
        "channel_signs": np.ones(3),
        "amplifier_gain_v_per_v": np.ones(3) * 4.0,
        "coil_resistance_ohm": np.ones(3) * 2.0,
        "coil_inductance_h": np.ones(3) * 0.01,
        "daq_voltage_limit_v": np.ones(3) * 10.0,
        "peak_current_limit_a": np.ones(3) * 2.0,
        "rms_current_limit_a": np.ones(3),
        "valid_frequency_hz": (0.0, 100.0),
        "verified": True,
    }
    values.update(overrides)
    return ActuatorCalibration(**values)


def forward_currents(
    calibration: ActuatorCalibration,
    voltage: np.ndarray,
    current_start: np.ndarray,
    dt_s: float,
) -> np.ndarray:
    current = np.array(current_start, dtype=np.float64, copy=True)
    history = np.empty_like(voltage)
    for index in range(voltage.shape[1]):
        current = calibration.electrical_current_step(
            current,
            voltage[:, index],
            dt_s,
        )
        history[:, index] = current
    return history


def total_field(
    calibration: ActuatorCalibration,
    electrical_current: np.ndarray,
) -> np.ndarray:
    model_current = calibration.electrical_to_model_current(electrical_current)
    return (
        calibration.coil_field_for_model_currents(model_current)
        + calibration.ambient_field_t[:, None]
    )


class CalculatorTests(unittest.TestCase):
    def setUp(self):
        self.calibration = make_calibration()

    def make_calculator(self) -> RotatingFieldCalculator:
        return RotatingFieldCalculator(
            self.calibration,
            sample_rate_hz=1000.0,
            default_transition_s=0.05,
            max_transition_s=0.25,
        )

    def test_axis_field_is_circular_perpendicular_and_right_handed(self):
        for axis in (
            np.array([1.0, 0.0, 0.0]),
            np.array([0.0, 1.0, 0.0]),
            np.array([0.0, 0.0, 1.0]),
            np.array([1.0, -2.0, 3.0]),
        ):
            with self.subTest(axis=axis):
                calculator = self.make_calculator()
                command = RotatingFieldCommand(axis, 1e-3, 5.0, 1)
                calculator.set_command(command, transition_s=0.05)
                current_start = calculator.status().electrical_current_setpoint_a
                voltage = calculator.generate_voltage_block(300)
                currents = forward_currents(
                    self.calibration,
                    voltage,
                    current_start,
                    calculator.dt_s,
                )
                field = total_field(self.calibration, currents)[:, 100:]
                unit_axis = axis / np.linalg.norm(axis)

                np.testing.assert_allclose(
                    np.linalg.norm(field, axis=0),
                    1e-3,
                    atol=2e-12,
                )
                np.testing.assert_allclose(
                    unit_axis @ field,
                    0.0,
                    atol=2e-12,
                )
                handedness = np.dot(
                    np.cross(field[:, 20], field[:, 21]),
                    unit_axis,
                )
                self.assertGreater(handedness, 0.0)

    def test_negative_rotation_sense_reverses_handedness(self):
        calculator = self.make_calculator()
        axis = np.array([0.0, 0.0, 1.0])
        command = RotatingFieldCommand(axis, 1e-3, 5.0, -1)
        calculator.set_command(command, transition_s=0.05)
        voltage = calculator.generate_voltage_block(200)
        currents = forward_currents(
            self.calibration,
            voltage,
            np.zeros(3),
            calculator.dt_s,
        )
        field = total_field(self.calibration, currents)[:, 100:]
        handedness = np.dot(np.cross(field[:, 10], field[:, 11]), axis)
        self.assertLess(handedness, 0.0)

    def test_split_blocks_equal_one_combined_block(self):
        command = RotatingFieldCommand([0.2, 0.4, 0.8], 8e-4, 7.0)
        combined = self.make_calculator()
        split = self.make_calculator()
        combined.set_command(command, transition_s=0.05)
        split.set_command(command, transition_s=0.05)

        expected = combined.generate_voltage_block(400)
        actual = np.concatenate(
            (
                split.generate_voltage_block(137),
                split.generate_voltage_block(263),
            ),
            axis=1,
        )
        np.testing.assert_allclose(actual, expected, atol=2e-13)

    def test_live_axis_change_is_smooth_and_reaches_target(self):
        calculator = self.make_calculator()
        calculator.set_command(
            RotatingFieldCommand([0, 0, 1], 1e-3, 5.0),
            transition_s=0.05,
        )
        first_voltage = calculator.generate_voltage_block(200)
        first_currents = forward_currents(
            self.calibration,
            first_voltage,
            np.zeros(3),
            calculator.dt_s,
        )
        previous_field = total_field(
            self.calibration,
            first_currents[:, -1:],
        )[:, 0]

        target_axis = np.array([1.0, 1.0, 1.0])
        calculator.set_command(
            RotatingFieldCommand(target_axis, 8e-4, 7.0),
            transition_s=0.05,
        )
        start_current = first_currents[:, -1]
        next_voltage = calculator.generate_voltage_block(100)
        next_currents = forward_currents(
            self.calibration,
            next_voltage,
            start_current,
            calculator.dt_s,
        )
        next_field = total_field(
            self.calibration,
            next_currents[:, :1],
        )[:, 0]

        self.assertLess(np.linalg.norm(next_field - previous_field), 1e-4)
        np.testing.assert_allclose(
            calculator.status().axis_xyz,
            target_axis / np.linalg.norm(target_axis),
            atol=1e-12,
        )

    def test_stop_finishes_at_strict_zero_voltage(self):
        calculator = self.make_calculator()
        calculator.set_command(
            RotatingFieldCommand([0, 0, 1], 1e-3, 5.0),
            transition_s=0.05,
        )
        calculator.generate_voltage_block(100)
        calculator.stop(transition_s=0.05)
        calculator.generate_voltage_block(100)
        stopped = calculator.generate_voltage_block(20)

        np.testing.assert_array_equal(stopped, np.zeros((3, 20)))
        status = calculator.status()
        self.assertFalse(status.active)
        np.testing.assert_array_equal(
            status.electrical_current_setpoint_a,
            np.zeros(3),
        )

    def test_preview_rl_voltage_matches_analytic_result(self):
        calculator = self.make_calculator()
        command = RotatingFieldCommand([0, 0, 1], 1e-3, 5.0)
        preview = calculator.preview(command)
        current_amplitude = 0.1
        expected_voltage = (
            current_amplitude
            * np.hypot(2.0, 2.0 * np.pi * 5.0 * 0.01)
            / 4.0
        )
        np.testing.assert_allclose(
            preview.peak_daq_voltage_v[:2],
            expected_voltage,
            atol=1e-14,
        )

    def test_transition_is_automatically_extended_to_respect_voltage_limit(self):
        limited = make_calibration(daq_voltage_limit_v=np.ones(3) * 0.2)
        calculator = RotatingFieldCalculator(
            limited,
            sample_rate_hz=1000.0,
            default_transition_s=0.001,
            max_transition_s=0.25,
        )
        status = calculator.set_command(
            RotatingFieldCommand([0, 0, 1], 1e-3, 5.0),
            transition_s=0.001,
        )

        self.assertGreater(status.transition_remaining_s, 0.001)
        voltage = calculator.generate_voltage_block(
            int(np.ceil(status.transition_remaining_s * 1000.0))
        )
        self.assertLessEqual(np.max(np.abs(voltage)), 0.2 + 1e-12)

    def test_invalid_and_unsafe_commands_are_rejected(self):
        calculator = self.make_calculator()
        with self.assertRaises(CommandValidationError):
            RotatingFieldCommand([0, 0, 0], 1e-3, 5.0)
        with self.assertRaises(CommandValidationError):
            calculator.preview(
                RotatingFieldCommand([0, 0, 1], 1e-3, 101.0)
            )
        with self.assertRaises(SafetyLimitError):
            calculator.preview(
                RotatingFieldCommand([0, 0, 1], 0.1, 5.0)
            )

    def test_unverified_and_singular_calibrations_cannot_drive(self):
        with self.assertRaisesRegex(CalibrationError, "not marked verified"):
            RotatingFieldCalculator(
                make_calibration(verified=False),
                sample_rate_hz=1000,
            )
        with self.assertRaisesRegex(CalibrationError, "rank"):
            RotatingFieldCalculator(
                make_calibration(
                    field_matrix_t_per_a=np.diag([0.01, 0.01, 0.0])
                ),
                sample_rate_hz=1000,
            )

    def test_allow_unverified_is_explicit_simulation_escape_hatch(self):
        calculator = RotatingFieldCalculator(
            make_calibration(verified=False),
            sample_rate_hz=1000,
            allow_unverified=True,
        )
        self.assertEqual(calculator.generate_voltage_block(10).shape, (3, 10))


if __name__ == "__main__":
    unittest.main()
