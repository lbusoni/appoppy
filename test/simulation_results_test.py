#!/usr/bin/env python
import unittest
from appoppy.long_exposure_simulation import LepSnapshotEntry
from appoppy.petalometer import Petalometer
import numpy as np
import astropy.units as u

from appoppy.simulation_results import SimulationResults


class SimulationResultsTest(unittest.TestCase):

    def _make_hdr(self):
        self._niter = 10
        p = SimulationResults.SNAPSHOT_PREFIX
        l = LepSnapshotEntry
        hdr = {}
        hdr[p+'.'+l.NITER] = self._niter
        hdr[p+'.'+l.ROT_ANGLE] = 13
        hdr[p+'.'+l.SIMUL_TRACKNUM] = 'goo_1234'
        hdr[p+'.'+l.TRACKNUM] = 'blah'
        hdr[p+'.PET.PATH1.PASSATA.TIME_STEP'] = 0.001
        hdr[p+'.'+l.WAVELENGTH] = 3.14e-6
        hdr[p+'.'+l.PIXELSZ] = 42.123
        hdr[p+'.'+l.M4_PETALS] = '[12. 11 10 9 8 7] nm'
        hdr[p+'.'+l.STARTSTEP] = 123
        hdr[p+'.'+l.SIMUL_MODE] = 'CLWFS'
        hdr[p+'.'+l.INTEGRAL_GAIN] = 0.66

        return hdr

    def setUp(self):
        hdr = self._make_hdr()
        sh = (self._niter, 5, 5)
        self._rec_phase = np.ones(sh)
        self._meas_petals = np.arange(self._niter*6).reshape((self._niter, 6))
        self._input_opd = np.ones(sh) * 100
        self._input_opd[0] = np.arange(25).reshape((5, 5))
        self._corrected_opd = np.ones(sh) * 42
        self._res = SimulationResults(
            hdr, self._rec_phase, self._meas_petals, self._input_opd, self._corrected_opd)

    def test_petals_have_zero_global_piston(self):
        np.testing.assert_allclose(
            self._res.petals().mean(axis=1), np.zeros(self._niter))

    def test_input_opd_has_zero_global_piston(self):
        np.testing.assert_allclose(
            self._res.input_opd().mean(axis=(1, 2)),
            np.zeros(self._niter))

    def test_corrected_opd_has_zero_global_piston(self):
        np.testing.assert_allclose(
            self._res.corrected_opd().mean(axis=(1, 2)),
            np.zeros(self._niter))

    def test_input_opd_std_is_time_averaged(self):
        # input_opd.std() is 0 for every iter != 0
        wanted = np.arange(25).std()/self._niter
        self.assertAlmostEqual(self._res.input_opd_std(), wanted)


if __name__ == "__main__":
    # import sys;sys.argv = ['', 'Test.testName']
    unittest.main()
