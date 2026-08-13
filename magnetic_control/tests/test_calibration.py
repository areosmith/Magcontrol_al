from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from magnetic_control.calibration import (
    ActuatorCalibration,
    CalibrationError,
    CURRENT_DRIVER,
    GaussmeterDataset,
    VOLTAGE_AMPLIFIER,
    fit_field_matrix,
    fit_gaussmeter_csv,
    fit_rl_channel,
)


def make_calibration(**overrides) -> ActuatorCalibration:
    values = {
        "field_matrix_t_per_a": np.diag([0.010, 0.012, 0.009]),
        "ambient_field_t": np.array([20e-6, -10e-6, 5e-6]),
        "channel_axes": ("x", "y", "z"),
        "channel_signs": np.array([1.0, -1.0, 1.0]),
        "amplifier_gain_v_per_v": np.array([4.0, 4.0, 5.0]),
        "coil_resistance_ohm": np.array([2.0, 2.2, 1.8]),
        "coil_inductance_h": np.array([0.010, 0.012, 0.008]),
        "daq_voltage_limit_v": np.array([10.0, 10.0, 10.0]),
        "peak_current_limit_a": np.array([2.0, 2.0, 2.0]),
        "rms_current_limit_a": np.array([1.0, 1.0, 1.0]),
        "valid_frequency_hz": (0.0, 100.0),
        "verified": True,
        "calibration_date": "2026-07-29",
    }
    values.update(overrides)
    return ActuatorCalibration(**values)


class CalibrationTests(unittest.TestCase):
    def test_daq_limit_cannot_exceed_output_layer_range(self):
        with self.assertRaisesRegex(CalibrationError, "must not exceed"):
            make_calibration(daq_voltage_limit_v=[10.01, 10.0, 10.0])

    def test_json_round_trip(self):
        calibration = make_calibration()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "calibration.json"
            calibration.to_json(path)
            restored = ActuatorCalibration.from_json(path)

        np.testing.assert_allclose(
            restored.field_matrix_t_per_a,
            calibration.field_matrix_t_per_a,
        )
        np.testing.assert_allclose(restored.ambient_field_t, calibration.ambient_field_t)
        self.assertEqual(restored.channel_axes, ("x", "y", "z"))
        self.assertTrue(restored.verified)

    def test_legacy_json_defaults_to_voltage_amplifiers(self):
        legacy = make_calibration().to_dict()
        legacy.pop("channel_drive_modes")
        legacy.pop("current_gain_a_per_v")
        legacy.pop("current_driver_bandwidth_hz")
        restored = ActuatorCalibration.from_dict(legacy)

        self.assertEqual(
            restored.channel_drive_modes,
            (VOLTAGE_AMPLIFIER,) * 3,
        )
        np.testing.assert_allclose(
            restored.amplifier_gain_v_per_v,
            [4.0, 4.0, 5.0],
        )

    def test_mixed_driver_modes_use_independent_channel_models(self):
        calibration = make_calibration(
            config_version=2,
            channel_drive_modes=(
                VOLTAGE_AMPLIFIER,
                CURRENT_DRIVER,
                CURRENT_DRIVER,
            ),
            current_gain_a_per_v=np.array([0.0, 2.0, 4.0]),
            current_driver_bandwidth_hz=np.array([0.0, 0.0, 50.0]),
        )
        now = np.array([0.1, -0.2, 0.3])
        desired = np.array([0.11, -0.18, 0.28])
        voltage = calibration.daq_voltage_for_current_step(now, desired, 1e-4)
        reconstructed = calibration.electrical_current_step(now, voltage, 1e-4)

        np.testing.assert_allclose(reconstructed, desired, atol=1e-13)
        self.assertAlmostEqual(voltage[1], desired[1] / 2.0)

    def test_current_driver_sinusoidal_compensation(self):
        calibration = make_calibration(
            config_version=2,
            channel_drive_modes=(CURRENT_DRIVER,) * 3,
            amplifier_gain_v_per_v=np.zeros(3),
            current_gain_a_per_v=np.array([2.0, 4.0, 5.0]),
            current_driver_bandwidth_hz=np.zeros(3),
        )
        dc, cosine, sine = calibration.daq_voltage_components_for_current(
            np.array([0.2, 0.4, 0.5]),
            np.array([0.4, 0.8, 1.0]),
            np.array([0.6, 1.2, 1.5]),
            signed_frequency_hz=20.0,
        )

        np.testing.assert_allclose(dc, [0.1, 0.1, 0.1])
        np.testing.assert_allclose(cosine, [0.2, 0.2, 0.2])
        np.testing.assert_allclose(sine, [0.3, 0.3, 0.3])

    def test_unverified_and_singular_readiness(self):
        unverified = make_calibration(verified=False)
        with self.assertRaisesRegex(CalibrationError, "not marked verified"):
            unverified.assert_ready()

        singular = make_calibration(
            field_matrix_t_per_a=np.diag([0.01, 0.01, 0.0])
        )
        with self.assertRaisesRegex(CalibrationError, "rank"):
            singular.assert_ready(require_verified=False)

    def test_discrete_rl_inverse_matches_forward_model(self):
        calibration = make_calibration()
        current_now = np.array([0.1, -0.2, 0.3])
        current_next = np.array([0.11, -0.18, 0.28])
        voltage = calibration.daq_voltage_for_current_step(
            current_now,
            current_next,
            1e-4,
        )
        reconstructed = calibration.electrical_current_step(
            current_now,
            voltage,
            1e-4,
        )
        np.testing.assert_allclose(reconstructed, current_next, atol=1e-13)

    def test_static_daq_voltage_can_build_gaussmeter_dataset(self):
        calibration = make_calibration(
            channel_drive_modes=(
                "voltage_amplifier",
                "current_driver",
                "current_driver",
            ),
            current_gain_a_per_v=np.array([0.0, 2.0, 3.0]),
            current_driver_bandwidth_hz=np.array([0.0, 100.0, 200.0]),
        )
        voltages = np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [-1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, -1.0, 0.0],
                [0.0, 0.0, 1.0],
                [0.0, 0.0, -1.0],
            ]
        )
        model_currents = np.array(
            [
                [0.0, 0.0, 0.0],
                [2.0, 0.0, 0.0],
                [-2.0, 0.0, 0.0],
                [0.0, -2.0, 0.0],
                [0.0, 2.0, 0.0],
                [0.0, 0.0, 3.0],
                [0.0, 0.0, -3.0],
            ]
        )
        fields = (
            model_currents @ calibration.field_matrix_t_per_a.T
            + calibration.ambient_field_t
        )
        dataset = GaussmeterDataset.from_static_daq_voltages(
            calibration,
            voltages,
            fields,
        )
        np.testing.assert_allclose(dataset.model_currents_a, model_currents)
        np.testing.assert_allclose(
            dataset.fit_field_matrix().field_matrix_t_per_a,
            calibration.field_matrix_t_per_a,
            atol=1e-14,
        )

    def test_fit_field_matrix_recovers_affine_model(self):
        matrix = np.array(
            [
                [0.010, 0.0004, -0.0002],
                [0.0001, 0.012, 0.0003],
                [-0.0003, 0.0002, 0.009],
            ]
        )
        ambient = np.array([30e-6, -20e-6, 10e-6])
        currents = np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [-1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, -1.0, 0.0],
                [0.0, 0.0, 1.0],
                [0.0, 0.0, -1.0],
                [0.4, -0.6, 0.8],
            ]
        )
        fields = currents @ matrix.T + ambient
        fit = fit_field_matrix(currents, fields)

        np.testing.assert_allclose(fit.field_matrix_t_per_a, matrix, atol=1e-14)
        np.testing.assert_allclose(fit.ambient_field_t, ambient, atol=1e-14)
        self.assertLess(fit.max_vector_error_t, 1e-13)

    def test_gaussmeter_csv_round_trip_and_fit_in_gauss(self):
        matrix = np.array(
            [
                [0.010, 0.0004, -0.0002],
                [0.0001, 0.012, 0.0003],
                [-0.0003, 0.0002, 0.009],
            ]
        )
        ambient = np.array([30e-6, -20e-6, 10e-6])
        currents = np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [-1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, -1.0, 0.0],
                [0.0, 0.0, 1.0],
                [0.0, 0.0, -1.0],
            ]
        )
        fields = currents @ matrix.T + ambient
        dataset = GaussmeterDataset(currents, fields)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "gaussmeter.csv"
            dataset.to_csv(path, field_unit="gauss")
            restored = GaussmeterDataset.from_csv(path)
            fit = fit_gaussmeter_csv(path)

        np.testing.assert_allclose(restored.model_currents_a, currents)
        np.testing.assert_allclose(restored.measured_fields_t, fields)
        np.testing.assert_allclose(fit.field_matrix_t_per_a, matrix, atol=1e-14)
        np.testing.assert_allclose(fit.ambient_field_t, ambient, atol=1e-14)

    def test_gaussmeter_csv_rejects_ambiguous_field_units(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "ambiguous.csv"
            path.write_text(
                "current_ch0_a,current_ch1_a,current_ch2_a,"
                "field_x_t,field_y_t,field_z_t,"
                "field_x_gauss,field_y_gauss,field_z_gauss\n"
                "0,0,0,0,0,0,0,0,0\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(CalibrationError, "exactly one"):
                GaussmeterDataset.from_csv(path)

    def test_fit_rl_channel_recovers_parameters(self):
        frequency = np.array([0.0, 2.0, 10.0, 25.0, 50.0])
        resistance = 3.2
        inductance = 0.018
        gain = 5.0
        current = np.full(frequency.shape, 0.4 + 0.1j)
        impedance = resistance + 1j * 2.0 * np.pi * frequency * inductance
        voltage = impedance * current / gain

        fit = fit_rl_channel(frequency, voltage, current, gain)
        self.assertAlmostEqual(fit.resistance_ohm, resistance, places=12)
        self.assertAlmostEqual(fit.inductance_h, inductance, places=12)
        self.assertLess(fit.max_relative_error, 1e-12)


if __name__ == "__main__":
    unittest.main()
