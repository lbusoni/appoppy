#!/usr/bin/env python
import unittest
import numpy as np
from appoppy.two_wavelength_interferometry import fractional_wavelength,\
    twi1, synthetic_lambda


class TwoWavelengthInterferometryTest(unittest.TestCase):

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_twi1_no_noise(self):
        opd = np.arange(-30000, 30000)
        wv1 = 1500
        wv2 = 1600

        # the sensor measure the opd in fraction of wavelength
        meas1 = fractional_wavelength(opd, wv1)
        meas2 = fractional_wavelength(opd, wv2)

        meas_opd, _ = twi1(meas1, meas2, wv1, wv2)

        valid_opd_range = self.index_opd_within_synth_wv(opd, wv1, wv2)
        np.testing.assert_allclose(meas_opd[valid_opd_range],
                                   opd[valid_opd_range])

    def test_synthetic_lambda(self):
        wv1 = 1500
        wv2 = 1600
        self.assertEqual(24000, synthetic_lambda(wv1, wv2))

    def test_twi1_raises_if_wv1_greater_than_wv2(self):
        opd = np.arange(-30000, 30000)
        wv1 = 1600
        wv2 = 1500

        # the sensor measure the opd in fraction of wavelength
        meas1 = fractional_wavelength(opd, wv1)
        meas2 = fractional_wavelength(opd, wv2)

        self.assertRaises(ValueError,
                          twi1,
                          meas1, meas2, wv1, wv2)

    @staticmethod
    def index_opd_within_synth_wv(opd, wv1, wv2):
        synth_wv = synthetic_lambda(wv1, wv2)
        half_synth_wv = synth_wv / 2
        idx = np.intersect1d(np.nonzero(opd > -half_synth_wv)[0],
                             np.nonzero(opd < half_synth_wv)[0])
        return idx

    def test_twi1_is_periodic_with_syntetic_lambda(self):
        opd = np.arange(-30000, 30000)
        # 1500,1600 gives synth_lambda=24000
        wv1 = 1500
        wv2 = 1600
        meas1 = fractional_wavelength(opd, wv1)
        meas2 = fractional_wavelength(opd, wv2)
        meas_opd = twi1(meas1, meas2, wv1, wv2)
        np.testing.assert_allclose(meas_opd[20000:30000],
                                   meas_opd[44000:54000])


if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()
