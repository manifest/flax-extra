import unittest
from jax import numpy as jnp
import numpy as np

from flax_extra import operator as op


class ShiftRightTest(unittest.TestCase):
    def test_default(self):
        value = jnp.array([1, 2, 3])
        expect = jnp.array([0, 1, 2])
        shift = op.ShiftRight()
        np.testing.assert_array_equal(shift(value), expect)

    def test_1d_0axis(self):
        value = jnp.array([1, 2, 3])
        expect = jnp.array([0, 1, 2])
        shift = op.ShiftRight(axis=0)
        np.testing.assert_array_equal(shift(value), expect)

    def test_1d_0axis_2pos(self):
        value = jnp.array([1, 2, 3])
        expect = jnp.array([0, 0, 1])
        shift = op.ShiftRight(axis=0, n_positions=2)
        np.testing.assert_array_equal(shift(value), expect)

    def test_3d_0axis(self):
        value = jnp.array(
            [
                [
                    [1, 2, 3],
                    [4, 5, 6],
                ],
                [
                    [1, 2, 3],
                    [4, 5, 6],
                ],
            ]
        )
        expect = jnp.array(
            [
                [
                    [0, 0, 0],
                    [0, 0, 0],
                ],
                [
                    [1, 2, 3],
                    [4, 5, 6],
                ],
            ]
        )
        shift = op.ShiftRight(axis=0)
        np.testing.assert_array_equal(shift(value), expect)

    def test_3d_1axis(self):
        value = jnp.array(
            [
                [
                    [1, 2, 3],
                    [4, 5, 6],
                ],
                [
                    [1, 2, 3],
                    [4, 5, 6],
                ],
            ]
        )
        expect = jnp.array(
            [
                [
                    [0, 0, 0],
                    [1, 2, 3],
                ],
                [
                    [0, 0, 0],
                    [1, 2, 3],
                ],
            ]
        )
        shift = op.ShiftRight(axis=1)
        np.testing.assert_array_equal(shift(value), expect)

    def test_3d_2axis(self):
        value = jnp.array(
            [
                [
                    [1, 2, 3],
                    [4, 5, 6],
                ],
                [
                    [1, 2, 3],
                    [4, 5, 6],
                ],
            ]
        )
        expect = jnp.array(
            [
                [
                    [0, 1, 2],
                    [0, 4, 5],
                ],
                [
                    [0, 1, 2],
                    [0, 4, 5],
                ],
            ]
        )
        shift = op.ShiftRight(axis=2)
        np.testing.assert_array_equal(shift(value), expect)
