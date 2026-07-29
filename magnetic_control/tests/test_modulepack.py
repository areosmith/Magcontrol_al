from __future__ import annotations

import unittest

import numpy as np

import modulepack as mp
from magnetic_control import (
    ActuatorCalibration,
    RotatingFieldCalculator,
    RotatingFieldCommand,
)


class ModulepackSourceTests(unittest.TestCase):
    def tearDown(self):
        try:
            mp.stop()
        except Exception:
            pass
        mp.use_manual_voltage_source()

    def test_custom_source_shape_and_values(self):
        mp.set_voltage_source(
            lambda samples: np.full((mp.NUM_CHANNELS, samples), 1.25)
        )
        data = mp.generate(17)
        self.assertEqual(data.shape, (3, 17))
        np.testing.assert_array_equal(data, np.full((3, 17), 1.25))
        self.assertIsNone(mp.get_output_fault())

    def test_invalid_source_latches_fault_and_returns_zero(self):
        mp.set_voltage_source(
            lambda samples: np.full((mp.NUM_CHANNELS, samples), np.nan)
        )
        first = mp.generate(8)
        second = mp.generate(8)

        np.testing.assert_array_equal(first, np.zeros((3, 8)))
        np.testing.assert_array_equal(second, np.zeros((3, 8)))
        self.assertIn("NaN", mp.get_output_fault())

    def test_overvoltage_source_latches_fault(self):
        mp.set_voltage_source(
            lambda samples: np.full((mp.NUM_CHANNELS, samples), 10.01)
        )
        data = mp.generate(4)

        np.testing.assert_array_equal(data, np.zeros((3, 4)))
        self.assertIn("AO range", mp.get_output_fault())

    def test_calculator_can_be_used_as_fake_daq_source(self):
        calibration = ActuatorCalibration(
            field_matrix_t_per_a=np.eye(3) * 0.01,
            ambient_field_t=np.zeros(3),
            channel_axes=("x", "y", "z"),
            channel_signs=np.ones(3),
            amplifier_gain_v_per_v=np.ones(3) * 4.0,
            coil_resistance_ohm=np.ones(3) * 2.0,
            coil_inductance_h=np.ones(3) * 0.01,
            daq_voltage_limit_v=np.ones(3) * 10.0,
            peak_current_limit_a=np.ones(3) * 2.0,
            rms_current_limit_a=np.ones(3),
            valid_frequency_hz=(0.0, 100.0),
            verified=True,
        )
        calculator = RotatingFieldCalculator(
            calibration,
            sample_rate_hz=mp.SAMPLE_RATE,
        )
        calculator.set_command(
            RotatingFieldCommand([0, 0, 1], 1e-3, 5.0)
        )
        mp.set_voltage_source(calculator.generate_voltage_block)

        data = mp.generate(mp.BUFFER_SIZE)
        self.assertEqual(data.shape, (3, mp.BUFFER_SIZE))
        self.assertTrue(np.all(np.isfinite(data)))
        self.assertLessEqual(np.max(np.abs(data)), 10.0)
        self.assertIsNone(mp.get_output_fault())


if __name__ == "__main__":
    unittest.main()
