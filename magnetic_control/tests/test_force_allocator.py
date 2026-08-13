from __future__ import annotations

import unittest

import numpy as np

from magnetic_control.calculator import RotatingFieldCommand
from magnetic_control.force_allocator import AllocationReason, ForceAllocator
from magnetic_control.force_command import PropulsionCommand, PropulsionMode
from magnetic_control.propulsion_model import MonotonicPropulsionLUT, RobotProfile


def make_model(
    mode: PropulsionMode = PropulsionMode.ESTIMATED_PROPULSIVE_COMPONENT,
    *,
    validated: bool = True,
) -> MonotonicPropulsionLUT:
    scale = 1.0 if mode is not PropulsionMode.BLOCKED_THRUST else 2.0
    return MonotonicPropulsionLUT(
        mode=mode,
        field_magnitudes_T=[1e-3, 2e-3],
        frequencies_hz=[0.0, 10.0, 20.0],
        output_si=np.array(
            [
                [0.0, 1e-9, 2e-9],
                [0.0, 2e-9, 4e-9],
            ]
        )
        * scale,
        stepout_frequency_lower_hz=[20.0, 30.0],
        validated=validated,
        stepout_safety_factor=0.75,
    )


def make_profile(*models, chirality=-1, magnetization=1) -> RobotProfile:
    return RobotProfile(
        profile_id="robot-1",
        chirality_sign=chirality,
        magnetization_sign=magnetization,
        models={model.mode: model for model in models},
    )


class ForceAllocatorTests(unittest.TestCase):
    def test_force_is_allocated_with_profile_direction_sign(self):
        profile = make_profile(make_model())
        command = PropulsionCommand(
            [1.2e-9, 1.6e-9, 0.0],
            PropulsionMode.ESTIMATED_PROPULSIVE_COMPONENT,
            "robot-1",
        )
        result = ForceAllocator().allocate(command, profile)

        self.assertTrue(result.accepted)
        self.assertIsInstance(result.rotating_field_command, RotatingFieldCommand)
        np.testing.assert_allclose(result.predicted_vector_si, command.target_xyz_si)
        np.testing.assert_allclose(
            result.rotating_field_command.axis_xyz,
            -command.target_xyz_si / command.target_magnitude_si,
        )
        self.assertGreater(result.stepout_margin_hz, 0.0)

    def test_blocked_thrust_requires_contact_and_uses_its_own_model(self):
        profile = make_profile(make_model(PropulsionMode.BLOCKED_THRUST))
        missing_contact = PropulsionCommand(
            [1e-9, 0.0, 0.0],
            PropulsionMode.BLOCKED_THRUST,
            "robot-1",
        )
        rejected = ForceAllocator().allocate(missing_contact, profile)
        self.assertFalse(rejected.accepted)
        self.assertEqual(rejected.reasons, (AllocationReason.CONTACT_NOT_CONFIRMED,))

        confirmed = PropulsionCommand(
            [1e-9, 0.0, 0.0],
            PropulsionMode.BLOCKED_THRUST,
            "robot-1",
            contact_confirmed=True,
            contact_id="fixture-1",
        )
        self.assertTrue(ForceAllocator().allocate(confirmed, profile).accepted)

    def test_velocity_mode_is_explicitly_rejected(self):
        velocity_model = make_model(PropulsionMode.FREE_SWIM_VELOCITY)
        command = PropulsionCommand(
            [1e-5, 0.0, 0.0],
            PropulsionMode.FREE_SWIM_VELOCITY,
            "robot-1",
        )
        result = ForceAllocator().allocate(command, make_profile(velocity_model))
        self.assertFalse(result.accepted)
        self.assertEqual(result.reasons, (AllocationReason.UNSUPPORTED_MODE,))
        self.assertIsNone(result.rotating_field_command)

    def test_out_of_range_rejects_unless_projection_is_requested(self):
        profile = make_profile(make_model())
        rejected = ForceAllocator().allocate(
            PropulsionCommand(
                [20e-9, 0.0, 0.0],
                PropulsionMode.ESTIMATED_PROPULSIVE_COMPONENT,
                "robot-1",
            ),
            profile,
        )
        self.assertFalse(rejected.accepted)
        self.assertEqual(
            rejected.reasons,
            (AllocationReason.TARGET_OUTSIDE_CALIBRATED_RANGE,),
        )

        projected = ForceAllocator().allocate(
            PropulsionCommand(
                [20e-9, 0.0, 0.0],
                PropulsionMode.ESTIMATED_PROPULSIVE_COMPONENT,
                "robot-1",
                project_to_feasible=True,
            ),
            profile,
        )
        self.assertTrue(projected.accepted)
        self.assertTrue(projected.projected)
        self.assertEqual(projected.reasons, (AllocationReason.PROJECTED_TO_FEASIBLE,))
        self.assertLess(np.linalg.norm(projected.predicted_vector_si), 20e-9)

    def test_zero_profile_mismatch_missing_and_unverified_models_reject(self):
        allocator = ForceAllocator()
        model = make_model()
        profile = make_profile(model)
        zero = PropulsionCommand(
            [0.0, 0.0, 0.0],
            PropulsionMode.ESTIMATED_PROPULSIVE_COMPONENT,
            "robot-1",
        )
        self.assertEqual(
            allocator.allocate(zero, profile).reasons,
            (AllocationReason.ZERO_TARGET_REQUIRES_STOP,),
        )

        mismatch = PropulsionCommand(
            [1e-9, 0.0, 0.0],
            PropulsionMode.ESTIMATED_PROPULSIVE_COMPONENT,
            "robot-2",
        )
        self.assertEqual(
            allocator.allocate(mismatch, profile).reasons,
            (AllocationReason.PROFILE_MISMATCH,),
        )

        wrong_frame = PropulsionCommand(
            [1e-9, 0.0, 0.0],
            PropulsionMode.ESTIMATED_PROPULSIVE_COMPONENT,
            "robot-1",
            frame="camera",
        )
        self.assertEqual(
            allocator.allocate(wrong_frame, profile).reasons,
            (AllocationReason.UNSUPPORTED_FRAME,),
        )

        blocked = PropulsionCommand(
            [1e-9, 0.0, 0.0],
            PropulsionMode.BLOCKED_THRUST,
            "robot-1",
            contact_confirmed=True,
            contact_id="fixture-1",
        )
        self.assertEqual(
            allocator.allocate(blocked, profile).reasons,
            (AllocationReason.MODEL_NOT_AVAILABLE,),
        )

        unverified_profile = make_profile(make_model(validated=False))
        request = PropulsionCommand(
            [1e-9, 0.0, 0.0],
            PropulsionMode.ESTIMATED_PROPULSIVE_COMPONENT,
            "robot-1",
        )
        self.assertEqual(
            allocator.allocate(request, unverified_profile).reasons,
            (AllocationReason.MODEL_NOT_VERIFIED,),
        )


if __name__ == "__main__":
    unittest.main()
