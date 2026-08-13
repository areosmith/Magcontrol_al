from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from magnetic_control.force_command import PropulsionMode
from magnetic_control.propulsion_model import (
    InfeasiblePropulsionTargetError,
    MonotonicPropulsionLUT,
    PropulsionModelError,
    RobotProfile,
    UnverifiedPropulsionModelError,
)


def make_lut(**overrides) -> MonotonicPropulsionLUT:
    values = {
        "mode": PropulsionMode.ESTIMATED_PROPULSIVE_COMPONENT,
        "field_magnitudes_T": [1e-3, 2e-3, 3e-3],
        "frequencies_hz": [0.0, 10.0, 20.0, 30.0],
        "output_si": np.array(
            [
                [0.0, 1e-9, 2e-9, 3e-9],
                [0.0, 2e-9, 4e-9, 6e-9],
                [0.0, 3e-9, 6e-9, 9e-9],
            ]
        ),
        "stepout_frequency_lower_hz": [20.0, 30.0, 40.0],
        "stepout_safety_factor": 0.75,
        "validated": True,
    }
    values.update(overrides)
    return MonotonicPropulsionLUT(**values)


class PropulsionModelTests(unittest.TestCase):
    def test_bilinear_prediction_and_stepout_interpolation(self):
        lut = make_lut()
        self.assertAlmostEqual(lut.predict(1.5e-3, 15.0), 2.25e-9)
        self.assertAlmostEqual(lut.stepout_lower_hz(1.5e-3), 25.0)
        self.assertAlmostEqual(lut.safe_frequency_limit_hz(1.5e-3), 18.75)

    def test_inverse_uses_lowest_safe_field_and_respects_stepout(self):
        lut = make_lut()
        solution = lut.inverse(2.0e-9)
        self.assertAlmostEqual(solution.achieved_si, 2.0e-9, places=20)
        self.assertLessEqual(solution.frequency_hz, solution.safe_frequency_limit_hz)
        self.assertGreater(solution.stepout_margin_hz, 0.0)
        self.assertFalse(solution.projected)

    def test_out_of_range_rejects_by_default_and_projects_explicitly(self):
        lut = make_lut()
        with self.assertRaises(InfeasiblePropulsionTargetError):
            lut.inverse(20e-9)

        solution = lut.inverse(20e-9, project_to_feasible=True)
        self.assertTrue(solution.projected)
        self.assertAlmostEqual(solution.achieved_si, lut.maximum_safe_output_si)
        self.assertLessEqual(solution.frequency_hz, solution.safe_frequency_limit_hz)

    def test_unverified_lut_cannot_be_inverted(self):
        with self.assertRaises(UnverifiedPropulsionModelError):
            make_lut(validated=False).inverse(1e-9)

    def test_non_monotonic_or_badly_shaped_calibration_is_rejected(self):
        bad = np.array(
            [
                [0.0, 1e-9, 0.5e-9, 3e-9],
                [0.0, 2e-9, 4e-9, 6e-9],
                [0.0, 3e-9, 6e-9, 9e-9],
            ]
        )
        with self.assertRaises(PropulsionModelError):
            make_lut(output_si=bad)
        with self.assertRaises(PropulsionModelError):
            make_lut(stepout_frequency_lower_hz=[20.0, 10.0, 40.0])

    def test_robot_profile_combines_chirality_and_magnetization_signs(self):
        lut = make_lut()
        profile = RobotProfile("robot-1", -1, 1, {lut.mode: lut})
        self.assertEqual(profile.propulsion_sign, -1)
        self.assertIs(profile.model_for(lut.mode), lut)
        with self.assertRaises(TypeError):
            profile.models[lut.mode] = lut

    def test_robot_profile_json_round_trip(self):
        mode = PropulsionMode.ESTIMATED_PROPULSIVE_COMPONENT
        profile = RobotProfile(
            profile_id="robot-json",
            chirality_sign=-1,
            magnetization_sign=1,
            models={mode: make_lut(mode=mode)},
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "robot.json"
            profile.to_json(path)
            restored = RobotProfile.from_json(path)
        self.assertEqual(restored.profile_id, profile.profile_id)
        self.assertEqual(restored.propulsion_sign, profile.propulsion_sign)
        restored_model = restored.model_for(mode)
        self.assertIsNotNone(restored_model)
        np.testing.assert_allclose(
            restored_model.output_si,
            profile.model_for(mode).output_si,
        )

    def test_profile_version_must_be_an_integer(self):
        profile = RobotProfile(
            profile_id="helix-01",
            chirality_sign=1,
            magnetization_sign=1,
            models={
                PropulsionMode.ESTIMATED_PROPULSIVE_COMPONENT: make_lut()
            },
        ).to_dict()
        for invalid_version in (1.9, "1", True):
            with self.subTest(version=invalid_version):
                invalid = dict(profile)
                invalid["profile_version"] = invalid_version
                with self.assertRaisesRegex(PropulsionModelError, "integer 1"):
                    RobotProfile.from_dict(invalid)


if __name__ == "__main__":
    unittest.main()
