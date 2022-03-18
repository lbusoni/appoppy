#!/usr/bin/env python
import unittest
from appoppy.circular_math import difference
from astropy import units as u


class CircularMathTest(unittest.TestCase):

    def test_difference(self):
        wl = 2200 * u.nm
        a = 1000 * u.nm
        b = -1000 * u.nm
        c = 980 * u.nm
        d = 5000 * u.nm
        e = 1100 * u.nm
        f = -1100 * u.nm
        g = 100 * u.nm
        self.assertAlmostEqual(-200, difference(a, b, wl).to_value(u.nm))
        self.assertAlmostEqual(-20, difference(c, a, wl).to_value(u.nm))
        self.assertAlmostEqual(-400, difference(d, a, wl).to_value(u.nm))
        self.assertAlmostEqual(0, difference(e, f, wl).to_value(u.nm))
        self.assertAlmostEqual(0, difference(c, c, wl).to_value(u.nm))
        self.assertAlmostEqual(1000, difference(f, g, wl).to_value(u.nm))

    def test_wrap(self):
        pass


if __name__ == "__main__":
    unittest.main()
