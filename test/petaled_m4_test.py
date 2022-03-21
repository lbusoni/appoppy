#!/usr/bin/env python
import unittest
import numpy as np
from astropy import units as u
from appoppy.petaled_m4 import PetaledM4
import poppy


class PetaledM4Test(unittest.TestCase):

    def testOrder(self):
        wl = 1e-6
        ss = poppy.OpticalSystem(pupil_diameter=1, npix=256)
        wv = ss.input_wavefront(wl)

        # petal 0
        pm4 = PetaledM4(piston=np.array([100, 0, 0, 0, 0, 0]) * u.nm)
        opd = pm4.get_opd(wv)
        self.assertAlmostEqual(opd[200, 200], 100e-9)

        # petal 1
        pm4 = PetaledM4(piston=np.array([0, 100, 0, 0, 0, 0]) * u.nm)
        opd = pm4.get_opd(wv)
        self.assertAlmostEqual(opd[128, 200], 100e-9)

        # petal 2
        pm4 = PetaledM4(piston=np.array([0, 0, 100, 0, 0, 0]) * u.nm)
        opd = pm4.get_opd(wv)
        self.assertAlmostEqual(opd[50, 200], 100e-9)


if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()
