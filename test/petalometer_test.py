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
                                   pet.difference_between_estimated_petals_and_m4_petals.to_value(u.nm), atol=1e-10)
        self.assertAlmostEqual(0, pet.pupil_opd.mean())
        self.assertAlmostEqual(0, pet.pupil_opd.std())

    def test_set_step_idx(self):
        pet = Petalometer(tracking_number='none')
        pet.set_step_idx(10)
        self.assertEqual(pet.step_idx, 10)
        pet.advance_step_idx()
        self.assertEqual(pet.step_idx, 11)

    def test_estimated_petals_have_zero_mean(self):
        m4_petals = np.array([0, 100, 200, 0, 300, 0])*u.nm
        pet = Petalometer(tracking_number='none',
                          petals=m4_petals)
        wanted = m4_petals - m4_petals.mean()
        pet.sense_wavefront_jumps()
        actual = pet.estimated_petals
        np.testing.assert_allclose(actual, wanted, atol=1e-10)


if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()
