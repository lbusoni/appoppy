#!/usr/bin/env python
import logging
import os
import tempfile
import unittest
import numpy as np
from astropy import units as u
from appoppy.simulation_results import SimulationResults
from appoppy.long_exposure_simulation import LongExposureSimulation, animation_folder
from appoppy.maory_residual_wfe import PASSATASimulationConverter
from appoppy.package_data import ROOT_DIR_KEY
import shutil


class SimulationIntegrationTest(unittest.TestCase):

    def setUp(self):
        self._setUpBasicLogging()
        self._modify_root_dir()

    def tearDown(self):
        try:
            self._delete_created_dir()
        except Exception as e:
            self._logger.error(
                'Could not delete folders created by the integration test. (%s)' % e)
        self._reset_root_dir()

    def _setUpBasicLogging(self):
        logging.basicConfig(level=logging.INFO)
        self._logger = logging.getLogger('Integration Test')
        self._logger.info("Integration Test started")

    def _modify_root_dir(self):
        os.environ[ROOT_DIR_KEY] = os.path.join(
            'test', 'data')
        self._root_dir = os.environ[ROOT_DIR_KEY]

    def _reset_root_dir(self):
        try:
            del os.environ[ROOT_DIR_KEY]
        except Exception:
            pass

    def _delete_created_dir(self):
        shutil.rmtree(os.path.join(self._root_dir,
                      'passata_simulations_converted'))
        shutil.rmtree(animation_folder(self._tn))

    def test_call_all(self):

        self._tn = 'foo_0000'
        aores_tn = 'none'

        print(self._root_dir)
        psc = PASSATASimulationConverter()
        psc.create_none_tracknum(niter=5)
        lep = LongExposureSimulation(self._tn,
                                     aores_tn,
                                     petals=np.array(
                                         [0, 123, 0, 0, 0, 0]) * u.nm,
                                     start_from_step=0,
                                     n_iter=4)
        lep.run()
        lep.save()

        sr = SimulationResults.load(self._tn)
        self._logger.info('Input opd %g' % sr.input_opd_std())
        self._logger.info('Corr opd %g' % sr.corrected_opd_std())
        np.testing.assert_allclose(
            lep._reconstructed_phase,
            sr.reconstructed_phase())

        sr.animate_corrected_opd()
        sr.animate_reconstructed_phase()

        self.assertAlmostEqual(45.7, sr.input_opd_std(), delta=1)
        self.assertAlmostEqual(0, sr.corrected_opd_std(), delta=1)
        self.assertAlmostEqual(
            0,
            sr.corrected_opd_from_reconstructed_phase_ave()[0].std(),
            delta=1)

        petals = sr.petals()[0]
        self._logger.info('Petals %s' % petals)
        self.assertAlmostEqual(123, (petals[1]-petals[0]))
        self.assertEqual(1, sr.time_step)

    def _temp_fname(self):
        return os.path.join(tempfile.gettempdir(), 'lep.fits')


if __name__ == "__main__":
    # import sys;sys.argv = ['', 'Test.testName']
    unittest.main()
