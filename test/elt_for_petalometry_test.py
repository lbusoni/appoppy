#!/usr/bin/env python
import unittest
from appoppy.elt_for_petalometry import EltForPetalometry


class EltForPetalometryTest(unittest.TestCase):

    def test_construction(self):
        efp = EltForPetalometry()
        self.assertIsInstance(efp, EltForPetalometry)

    def test_default_is_null_opd(self):
        efp = EltForPetalometry()
        self.assertAlmostEqual(0, efp.pupil_opd().std())
        self.assertAlmostEqual(0, efp.pupil_opd().mean())


if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()
