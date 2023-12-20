#!/usr/bin/env python
import unittest
from appoppy.petalometer import Petalometer
import numpy as np
import astropy.units as u


class PetalometerTest(unittest.TestCase):

    def test_construction(self):
        efp = Petalometer()
        self.assertIsInstance(efp, Petalometer)

    def test_default_is_null_petals(self):
        pet = Petalometer()
        wanted = np.zeros(6)
        pet.sense_wavefront_jumps()
        actual = pet.estimated_petals
        np.testing.assert_allclose(actual, wanted, atol=1e-10)

    def test_call_everything(self):
        pet = Petalometer()
        pet.sense_wavefront_jumps()
        np.testing.assert_allclose(np.zeros(6),
                                   pet.estimated_petals.to_value(u.nm), atol=1e-10)
        np.testing.assert_allclose(np.zeros(6),
                                   pet.error_petals.to_value(u.nm), atol=1e-10)
        self.assertAlmostEqual(0, pet.pupil_opd.mean())
        self.assertAlmostEqual(0, pet.pupil_opd.std())

    def test_set_step_idx(self):
        pet = Petalometer(tracking_number='none')
        pet.set_step_idx(10)
        self.assertEqual(pet.step_idx, 10)
        pet.advance_step_idx()
        self.assertEqual(pet.step_idx, 11)


if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()
