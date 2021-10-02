import unittest
from jax import numpy as jnp
import numpy as np

from flax_extra import combinator as cb


class ConcatenateTest(unittest.TestCase):
    def test_signature(self):
        concatenate = cb.concatenate()
        self.assertEqual(concatenate.signature.n_in, 2)
        self.assertEqual(concatenate.signature.n_out, 1)
        self.assertEqual(
            concatenate.signature.in_shape,
            ((), ()),
        )

    def test_default(self):
        concatenate = cb.concatenate()
        np.testing.assert_array_equal(
            concatenate(jnp.array([1, 2]), jnp.array([3])),
            jnp.array([1, 2, 3]),
        )

    def test_single_item(self):
        concatenate = cb.concatenate(n_in=1)
        np.testing.assert_array_equal(
            concatenate(jnp.array([1, 2])),
            jnp.array([1, 2]),
        )

    def test_many_items(self):
        concatenate = cb.concatenate(n_in=3)
        np.testing.assert_array_equal(
            concatenate(jnp.array([1, 2]), jnp.array([3]), jnp.array([4, 5])),
            jnp.array([1, 2, 3, 4, 5]),
        )

    def test_without_any_item(self):
        concatenate = cb.concatenate(n_in=0)
        with self.assertRaises(ValueError):
            concatenate()

    def test_extra_input(self):
        concatenate = cb.concatenate(n_in=1)
        result, extra = concatenate(jnp.array([1, 2]), jnp.array([3]))
        np.testing.assert_array_equal(result, jnp.array([1, 2]))
        np.testing.assert_array_equal(extra, jnp.array([3]))

    def test_less_input(self):
        concatenate = cb.concatenate(n_in=2)
        with self.assertRaises(ValueError):
            concatenate(jnp.array([1, 2]))
