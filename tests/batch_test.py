import unittest
from jax import numpy as jnp
import numpy as np
from flax_extra.batch import normalize_batch, normalize_batch_per_device


class NormalizeBatchTest(unittest.TestCase):
    def test(self):
        batch_size = 4
        y = x = jnp.ones((batch_size, 3))
        test = [
            ((x, y), ((x,), (y,))),
            (((x,), (y,)), ((x,), (y,))),
            (((x, x), y), ((x, x), (y,))),
            ((x, (y, y)), ((x,), (y, y))),
            (((x, x), (y, y)), ((x, x), (y, y))),
            ((x, ()), ((x,), ())),
        ]
        for (value, expect) in test:
            result_inputs, result_targets = normalize_batch(value)
            expect_inputs, expect_targets = expect
            for i in range(len(expect_inputs)):
                np.testing.assert_array_equal(result_inputs[i], expect_inputs[i])
            for i in range(len(expect_targets)):
                np.testing.assert_array_equal(result_targets[i], expect_targets[i])


class NormalizeBatchPerDeviceTest(unittest.TestCase):
    def test(self):
        n_devices = 2
        batch_size = 4
        y = x = jnp.ones((batch_size, 3))
        dy = dx = jnp.ones((n_devices, batch_size // n_devices, 3))
        test = [
            ((x, y), ((dx,), (dy,))),
            (((x,), (y,)), ((dx,), (dy,))),
            (((x, x), y), ((dx, dx), (dy,))),
            ((x, (y, y)), ((dx,), (dy, dy))),
            (((x, x), (y, y)), ((dx, dx), (dy, dy))),
            ((x, ()), ((dx,), ())),
        ]
        for (value, expect) in test:
            result_inputs, result_targets = normalize_batch_per_device(
                value,
                n_devices=n_devices,
            )
            expect_inputs, expect_targets = expect
            for i in range(len(expect_inputs)):
                np.testing.assert_array_equal(result_inputs[i], expect_inputs[i])
            for i in range(len(expect_targets)):
                np.testing.assert_array_equal(result_targets[i], expect_targets[i])
