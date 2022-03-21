#!/usr/bin/env python
import unittest
from appoppy.mask import sector_mask


class SectorMaskTest(unittest.TestCase):

    def test_around_pi(self):

        # Fron -150 to 150 includes E. not W
        mask = sector_mask((100, 100), (-150, 150))
        self.assertFalse(mask[50, 10])
        self.assertTrue(mask[50, 90])

        # Fron -210 to 150 includes W. not E
        mask = sector_mask((100, 100), (-210, -150))
        self.assertTrue(mask[50, 10])
        self.assertFalse(mask[50, 60])

        # Fron 150 to 210 includes W. not E
        mask = sector_mask((100, 100), (-210, -150))
        self.assertTrue(mask[50, 10])
        self.assertFalse(mask[50, 60])

        # Fron 150 to -150 raise ValueError
        with self.assertRaises(ValueError):
            sector_mask((100, 100), (150, -150))

    def test_first_quadrant(self):
        mask = sector_mask((100, 100), (0, 90))
        self.assertFalse(mask[60, 49])
        self.assertFalse(mask[49, 60])
        self.assertTrue(mask[60, 60])


if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()
