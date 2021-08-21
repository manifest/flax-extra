import unittest
import jax
from jax import numpy as jnp
import numpy as np
from flax_extra.util import batch_per_device, originate


class OriginateTest(unittest.TestCase):
    def test(self):
        n_devices = 2
        original_x = jnp.ones((4, 3))
        replicated_x = jnp.ones((n_devices, 4, 3))
        value = dict(leaf=replicated_x)
        expect = dict(leaf=original_x)
        np.testing.assert_array_equal(
            originate(value).get("leaf"),
            expect.get("leaf"),
        )


class BatchPerDeviceTest(unittest.TestCase):
    def test_inputs(self):
        n_devices = 2
        batch_size = 4
        value = jnp.ones((batch_size, 3))
        expect = jnp.ones((n_devices, batch_size // n_devices, 3))
        np.testing.assert_array_equal(
            batch_per_device(value, n_devices=n_devices),
            expect,
        )

    def test_default(self):
        n_devices = jax.local_device_count()
        batch_size = 4
        value = jnp.ones((batch_size, 3))
        expect = jnp.ones((n_devices, batch_size // n_devices, 3))
        np.testing.assert_array_equal(
            batch_per_device(value),
            expect,
        )
