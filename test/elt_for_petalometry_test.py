#!/usr/bin/env python
import unittest
from appoppy.elt_for_petalometry import EltForPetalometry
import numpy as np
from astropy import units as u

class EltForPetalometryTest(unittest.TestCase):

    def test_construction(self):
        efp = EltForPetalometry()
        self.assertIsInstance(efp, EltForPetalometry)

    def test_none_arguments_makes_minimal_system(self):
        efp = EltForPetalometry()
        self.assertEqual(len(efp.optical_system.planes), 5)


    def test_default_is_null_opd(self):
        efp = EltForPetalometry()
        self.assertAlmostEqual(0, efp.pupil_opd().std())
        self.assertAlmostEqual(0, efp.pupil_opd().mean())

    def test_call_all(self):
        efp = EltForPetalometry()
        res = efp.pupil_opd()
        res = efp.psf()
        res = efp.pupil_phase()
        res = efp.pupil_mask()
        aa = efp.get_snapshot()

    def test_set_m4_petals(self):
        efp = EltForPetalometry(wavelength=100*u.nm)
        efp.set_m4_petals(np.array([0, 200, 0, 0, 0, 0])*u.nm)
        opd = efp.pupil_opd()
        masked_phase = np.ma.array(
            data=efp.pupil_phase(),
            mask=efp.pupil_mask())
        self.assertAlmostEqual(74.5, opd.std(), delta=0.2)
        self.assertAlmostEqual(0, masked_phase.std(), delta=0.01)

    def test_create_with_zern_and_set(self):
        efp = EltForPetalometry(zern_coeff=np.array([0, 100])*u.nm)
        new_zern_coeff = np.array([0, 200])*u.nm
        efp.set_input_wavefront_zernike(new_zern_coeff)
        np.testing.assert_allclose(new_zern_coeff.to_value(u.m),
                                   efp.zernike_coefficients)

    def test_create_without_zern_and_set_raises_exception(self):
        efp = EltForPetalometry()
        new_zern_coeff = np.array([0, 100])*u.nm
        self.assertRaises(Exception,
                          efp.set_input_wavefront_zernike,
                          new_zern_coeff)

    def test_create_with_everything(self):
        efp = EltForPetalometry(r0=0.1,
                                tracking_number='none',
                                zern_coeff=np.array([0, 0, 0, 100e-9]),
                                lwe_speed=0.5,
                                rotation_angle=43,
                                kolm_seed=2134)
        res = efp.pupil_opd()
        res = efp.pupil_phase()
        res = efp.pupil_wavefront()
        aa = efp.get_snapshot()
        psf = efp.psf()

if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()
