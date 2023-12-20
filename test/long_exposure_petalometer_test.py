#!/usr/bin/env python
import os
import tempfile
import unittest
import numpy as np
from astropy import units as u
from appoppy.long_exposure import LongExposurePetalometer


class LongExposurePetalometerTest(unittest.TestCase):

    def test_construction(self):
        lep = LongExposurePetalometer('foo', 'none')
        self.assertIsInstance(lep, LongExposurePetalometer)

    def test_call_all(self):
        lep = LongExposurePetalometer('foo',
                                      'none',
                                      start_from_step=0,
                                      n_iter=5)
        lep.run()
        fname = self._temp_fname()
        lep.save(fname)
        lep2 = LongExposurePetalometer.load(fname)
        np.testing.assert_allclose(
            lep.reconstructed_phase(),
            lep2.reconstructed_phase())

    def _temp_fname(self):
        return os.path.join(tempfile.gettempdir(), 'lep.fits')


if __name__ == "__main__":
    # import sys;sys.argv = ['', 'Test.testName']
    unittest.main()
