from __future__ import annotations

import unittest

import numpy as np

from magnetic_control.stage import (
    DeadbandRecenterController,
    RecenterAction,
    SimulatedXYStage,
    StageError,
    StageLimits,
    StageState,
)


def make_limits() -> StageLimits:
    return StageLimits(
        minimum_position_m=[-0.02, -0.03],
        maximum_position_m=[0.02, 0.03],
        maximum_speed_m_s=[0.01, 0.015],
        maximum_acceleration_m_s2=[0.1, 0.12],
    )


class StageModelTests(unittest.TestCase):
    def test_limits_validate_and_clip_positions(self):
        limits = make_limits()
        self.assertTrue(limits.contains([0.0, 0.0]))
        self.assertFalse(limits.contains([0.021, 0.0]))
        np.testing.assert_allclose(
            limits.clip([0.1, -0.1]),
            [0.02, -0.03],
        )
        self.assertFalse(limits.minimum_position_m.flags.writeable)
        exported_minimum = limits.minimum_position_m
        exported_minimum.setflags(write=True)
        exported_minimum[0] = -100.0
        np.testing.assert_allclose(limits.minimum_position_m, [-0.02, -0.03])

        with self.assertRaisesRegex(StageError, "exceed"):
            StageLimits(
                minimum_position_m=[0.0, 0.0],
                maximum_position_m=[0.0, 1.0],
                maximum_speed_m_s=1.0,
                maximum_acceleration_m_s2=1.0,
            )

    def test_state_has_timestamp_home_and_fault_fields(self):
        state = StageState(
            position_m=[0.001, -0.002],
            velocity_m_s=[0.0, 0.0],
            timestamp_s=1.25,
            homed=True,
            fault=None,
        )
        self.assertEqual(state.timestamp_s, 1.25)
        self.assertTrue(state.homed)
        self.assertIsNone(state.fault)
        self.assertFalse(state.moving)
        self.assertFalse(state.position_m.flags.writeable)

        with self.assertRaisesRegex(StageError, "non-negative"):
            StageState([0.0, 0.0], [0.0, 0.0], -1.0, True)


class SimulatedStageTests(unittest.TestCase):
    def test_stage_requires_homing_then_reaches_commanded_position(self):
        stage = SimulatedXYStage(make_limits())
        with self.assertRaisesRegex(StageError, "homed"):
            stage.command_position([0.01, 0.0])

        stage.home()
        stage.command_position([0.01, -0.005], maximum_speed_m_s=0.008)
        first = stage.advance(0.01)
        self.assertLessEqual(np.linalg.norm(first.velocity_m_s), 0.002)

        for _ in range(1000):
            state = stage.advance(0.01)
            if not state.moving:
                break
        else:
            self.fail("simulated stage did not reach its target")

        np.testing.assert_allclose(state.position_m, [0.01, -0.005], atol=1e-12)
        self.assertGreater(state.timestamp_s, 0.0)
        self.assertTrue(state.homed)

    def test_commands_enforce_travel_and_speed_limits(self):
        stage = SimulatedXYStage(make_limits(), initially_homed=True)
        with self.assertRaisesRegex(StageError, "travel"):
            stage.command_position([0.03, 0.0])
        with self.assertRaisesRegex(StageError, "speed"):
            stage.command_position([0.01, 0.0], maximum_speed_m_s=0.02)

    def test_stop_and_fault_are_observable(self):
        stage = SimulatedXYStage(make_limits(), initially_homed=True)
        stage.command_position([0.01, 0.0])
        stage.advance(0.1)
        stage.stop()
        stopped = stage.read_state()
        self.assertFalse(stopped.moving)
        np.testing.assert_allclose(stage.target_position_m, stopped.position_m)

        stage.set_fault("limit switch")
        faulted = stage.advance(0.1)
        self.assertEqual(faulted.fault, "limit switch")
        with self.assertRaisesRegex(StageError, "faulted"):
            stage.command_position([0.0, 0.0])
        stage.clear_fault()
        self.assertFalse(stage.read_state().homed)


class RecenterControllerTests(unittest.TestCase):
    def test_hysteresis_holds_moves_and_stops(self):
        stage = SimulatedXYStage(make_limits(), initially_homed=True)
        controller = DeadbandRecenterController(
            trigger_radius_m=0.0055,
            stop_radius_m=0.002,
            maximum_speed_m_s=0.005,
        )

        hold = controller.update(stage, [0.004, 0.0])
        self.assertEqual(hold.action, RecenterAction.HOLD)
        self.assertFalse(controller.active)

        move = controller.update(stage, [0.006, 0.0])
        self.assertEqual(move.action, RecenterAction.MOVE)
        self.assertTrue(controller.active)
        np.testing.assert_allclose(move.target_stage_position_m, [0.006, 0.0])

        still_move = controller.update(stage, [0.004, 0.0])
        self.assertEqual(still_move.action, RecenterAction.MOVE)
        self.assertTrue(controller.active)

        stop = controller.update(stage, [0.001, 0.0])
        self.assertEqual(stop.action, RecenterAction.STOP)
        self.assertFalse(controller.active)

    def test_configurable_coordinate_mapping_and_limit_clipping(self):
        stage = SimulatedXYStage(
            make_limits(),
            initial_position_m=[0.018, 0.0],
            initially_homed=True,
        )
        controller = DeadbandRecenterController(
            trigger_radius_m=0.005,
            stop_radius_m=0.001,
            local_to_stage=[[-1.0, 0.0], [0.0, 1.0]],
        )
        decision = controller.update(stage, [-0.01, 0.006])
        self.assertEqual(decision.action, RecenterAction.MOVE)
        self.assertTrue(decision.saturated)
        np.testing.assert_allclose(
            decision.target_stage_position_m,
            [0.02, 0.006],
        )

    def test_unhomed_or_faulted_stage_blocks_motion(self):
        stage = SimulatedXYStage(make_limits())
        controller = DeadbandRecenterController(
            trigger_radius_m=0.005,
            stop_radius_m=0.001,
        )
        unhomed = controller.update(stage, [0.01, 0.0])
        self.assertEqual(unhomed.action, RecenterAction.BLOCKED)
        self.assertIn("not homed", unhomed.reason)

        stage.home()
        stage.set_fault("emergency stop")
        faulted = controller.update(stage, [0.01, 0.0])
        self.assertEqual(faulted.action, RecenterAction.BLOCKED)
        self.assertIn("emergency stop", faulted.reason)

    def test_invalid_hysteresis_thresholds_are_rejected(self):
        with self.assertRaisesRegex(StageError, "stop"):
            DeadbandRecenterController(
                trigger_radius_m=0.005,
                stop_radius_m=0.005,
            )


if __name__ == "__main__":
    unittest.main()
