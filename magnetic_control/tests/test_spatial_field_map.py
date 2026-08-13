from __future__ import annotations

import unittest

import numpy as np

from magnetic_control.spatial_field_map import (
    CenterFieldMap,
    FieldMapResult,
    RegularGridFieldMap,
    SpatialFieldMapError,
)


def linear_grid_data(axes: tuple[np.ndarray, ...]):
    shape = tuple(axis.size for axis in axes)
    matrices = np.empty(shape + (3, 3), dtype=np.float64)
    ambient = np.empty(shape + (3,), dtype=np.float64)
    for index in np.ndindex(shape):
        position = np.array(
            [axes[dimension][value] for dimension, value in enumerate(index)]
        )
        x = position[0]
        y = position[1]
        z = position[2] if len(axes) == 3 else 0.0
        matrices[index] = np.array(
            [
                [0.010 + 0.2 * x, 0.1 * y, 0.05 * z],
                [-0.04 * y, 0.012 + 0.1 * z, 0.03 * x],
                [0.02 * z, -0.05 * x, 0.009 + 0.2 * y],
            ]
        )
        ambient[index] = np.array(
            [20e-6 + 1e-3 * x, -10e-6 + 2e-3 * y, 5e-6 - 3e-3 * z]
        )
    return matrices, ambient


class CenterFieldMapTests(unittest.TestCase):
    def test_default_map_is_valid_at_center_only(self):
        matrix = np.diag([0.010, 0.012, 0.009])
        ambient = np.array([20e-6, -10e-6, 5e-6])
        field_map = CenterFieldMap(matrix, ambient)

        result = field_map.query([0.0, 0.0])
        self.assertTrue(result.valid)
        np.testing.assert_allclose(result.field_matrix_t_per_a, matrix)
        np.testing.assert_allclose(result.ambient_field_t, ambient)
        self.assertFalse(result.position_m.flags.writeable)
        self.assertFalse(result.field_matrix_t_per_a.flags.writeable)
        self.assertFalse(result.ambient_field_t.flags.writeable)

        outside = field_map.query([1e-12, 0.0])
        self.assertFalse(outside.valid)
        self.assertIsNone(outside.field_matrix_t_per_a)
        self.assertIn("outside", outside.reason)

    def test_explicit_radius_and_three_dimensional_center(self):
        field_map = CenterFieldMap(
            np.eye(3),
            np.zeros(3),
            center_position_m=[0.01, -0.02, 0.03],
            valid_radius_m=0.004,
        )
        self.assertEqual(field_map.dimensions, 3)
        self.assertTrue(field_map.query([0.012, -0.02, 0.03]).valid)
        self.assertFalse(field_map.query([0.015, -0.02, 0.03]).valid)

        exported_center = field_map.center_position_m
        exported_center.setflags(write=True)
        exported_center[0] = 1.0
        np.testing.assert_allclose(
            field_map.center_position_m,
            [0.01, -0.02, 0.03],
        )

    def test_query_rejects_wrong_shape_and_nonfinite_values(self):
        field_map = CenterFieldMap(np.eye(3), np.zeros(3))
        with self.assertRaisesRegex(SpatialFieldMapError, "shape"):
            field_map.query([0.0, 0.0, 0.0])
        with self.assertRaisesRegex(SpatialFieldMapError, "finite"):
            field_map.query([np.nan, 0.0])


class RegularGridFieldMapTests(unittest.TestCase):
    def test_bilinear_interpolation_of_matrix_and_background(self):
        axes = (
            np.array([-0.01, 0.02]),
            np.array([-0.02, 0.00, 0.03]),
        )
        matrices, ambient = linear_grid_data(axes)
        field_map = RegularGridFieldMap(axes, matrices, ambient)

        position = np.array([0.005, 0.012])
        result = field_map.query(position)
        expected_matrices, expected_ambient = linear_grid_data(
            (np.array([position[0], position[0] + 1.0]),
             np.array([position[1], position[1] + 1.0]))
        )

        self.assertTrue(result.valid)
        np.testing.assert_allclose(
            result.field_matrix_t_per_a,
            expected_matrices[0, 0],
            atol=1e-15,
        )
        np.testing.assert_allclose(
            result.ambient_field_t,
            expected_ambient[0, 0],
            atol=1e-15,
        )
        self.assertFalse(result.field_matrix_t_per_a.flags.writeable)
        self.assertFalse(result.ambient_field_t.flags.writeable)

    def test_trilinear_interpolation(self):
        axes = (
            np.array([-0.01, 0.01]),
            np.array([-0.02, 0.02]),
            np.array([0.0, 0.01, 0.04]),
        )
        matrices, ambient = linear_grid_data(axes)
        field_map = RegularGridFieldMap(axes, matrices, ambient)
        position = np.array([0.004, -0.006, 0.025])

        result = field_map.query(position)
        expected_matrices, expected_ambient = linear_grid_data(
            tuple(np.array([coordinate, coordinate + 1.0]) for coordinate in position)
        )
        self.assertTrue(result.valid)
        np.testing.assert_allclose(
            result.field_matrix_t_per_a,
            expected_matrices[(0, 0, 0)],
            atol=1e-15,
        )
        np.testing.assert_allclose(
            result.ambient_field_t,
            expected_ambient[(0, 0, 0)],
            atol=1e-15,
        )

    def test_grid_boundary_is_valid_but_extrapolation_is_rejected(self):
        axes = (np.array([0.0, 0.01]), np.array([-0.02, 0.02]))
        matrices, ambient = linear_grid_data(axes)
        field_map = RegularGridFieldMap(axes, matrices, ambient)

        self.assertTrue(field_map.query([0.01, 0.02]).valid)
        outside = field_map.query([0.0101, 0.0])
        self.assertFalse(outside.valid)
        self.assertIn("outside", outside.reason)

    def test_invalid_measurement_node_invalidates_interpolation_cell(self):
        axes = (np.array([0.0, 0.01]), np.array([0.0, 0.01]))
        matrices, ambient = linear_grid_data(axes)
        mask = np.ones((2, 2), dtype=bool)
        mask[1, 1] = False
        field_map = RegularGridFieldMap(
            axes,
            matrices,
            ambient,
            valid_mask=mask,
        )

        self.assertTrue(field_map.query([0.0, 0.0]).valid)
        result = field_map.query([0.005, 0.005])
        self.assertFalse(result.valid)
        self.assertIn("rejected calibration node", result.reason)

    def test_constructor_validates_axes_and_array_shapes(self):
        axes = (np.array([0.0, 0.01]), np.array([0.0, 0.01]))
        matrices, ambient = linear_grid_data(axes)
        with self.assertRaisesRegex(SpatialFieldMapError, "strictly increasing"):
            RegularGridFieldMap(
                (np.array([0.0, 0.0]), axes[1]),
                matrices,
                ambient,
            )
        with self.assertRaisesRegex(SpatialFieldMapError, "shape"):
            RegularGridFieldMap(axes, np.zeros((2, 2, 3)), ambient)
        with self.assertRaisesRegex(SpatialFieldMapError, "finite"):
            invalid = matrices.copy()
            invalid[0, 0, 0, 0] = np.inf
            RegularGridFieldMap(axes, invalid, ambient)
        with self.assertRaisesRegex(SpatialFieldMapError, "boolean"):
            RegularGridFieldMap(
                axes,
                matrices,
                ambient,
                valid_mask=np.array([[1.0, np.nan], [1.0, 1.0]]),
            )

    def test_exported_axes_and_mask_cannot_mutate_map(self):
        axes = (np.array([0.0, 0.01]), np.array([0.0, 0.01]))
        matrices, ambient = linear_grid_data(axes)
        field_map = RegularGridFieldMap(axes, matrices, ambient)

        exported_axis = field_map.axes_m[0]
        exported_axis.setflags(write=True)
        exported_axis[0] = -10.0
        exported_mask = field_map.valid_mask
        exported_mask.setflags(write=True)
        exported_mask[:] = False

        self.assertTrue(field_map.query([0.005, 0.005]).valid)
        np.testing.assert_allclose(field_map.axes_m[0], [0.0, 0.01])


class FieldMapResultTests(unittest.TestCase):
    def test_invalid_result_cannot_claim_calibration_values(self):
        with self.assertRaisesRegex(SpatialFieldMapError, "must not provide"):
            FieldMapResult(
                position_m=[0.0, 0.0],
                valid=False,
                field_matrix_t_per_a=np.eye(3),
            )


if __name__ == "__main__":
    unittest.main()
