#!/usr/bin/env python
import unittest
from appoppy.petalometer import Petalometer
import numpy as np
import astropy.units as u


class PetalometerTest(unittest.TestCase):

    def test_construction(self):
        efp = Petalometer()
        self.assertIsInstance(efp, Petalometer)

    def test_default_is_null_jump(self):
        pet = Petalometer()
        wanted = np.zeros(12)
        actual = pet.sense_wavefront_jumps().to_value(u.nm)
        np.testing.assert_allclose(actual, wanted, atol=1e-10)


if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()
