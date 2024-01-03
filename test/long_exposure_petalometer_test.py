#!/usr/bin/env python
import logging
import os
import tempfile
import unittest
import numpy as np
from astropy import units as u
from appoppy.long_exposure import LongExposurePetalometer, animation_folder
from appoppy.maory_residual_wfe import PASSATASimulationConverter
from appoppy.package_data import ROOT_DIR_KEY
import shutil


class AppoppyLongExposureIntegrationTest(unittest.TestCase):

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
        lep = LongExposurePetalometer(self._tn,
                                      aores_tn,
                                      petals=np.array(
                                          [0, 123, 0, 0, 0, 0]) * u.nm,
                                      start_from_step=0,
                                      n_iter=4)
        lep.run()
        self._logger.info('Input opd %g' % lep.input_opd_std())
        self._logger.info('Corr opd %g' % lep.corrected_opd_std())
        lep.save()
        lep.animate_corrected_opd()
        lep.animate_reconstructed_phase()
        self.assertAlmostEqual(45.7, lep.input_opd_std(), delta=1)
        self.assertAlmostEqual(0, lep.corrected_opd_std(), delta=1)
        self.assertAlmostEqual(
            0,
            lep.corrected_opd_from_reconstructed_phase_ave()[0].std(),
            delta=1)

        lep2 = LongExposurePetalometer.load(self._tn)
        np.testing.assert_allclose(
            lep.reconstructed_phase(),
            lep2.reconstructed_phase())

        petals, jumps = lep2.petals_from_reconstructed_phase_map(
            lep2.reconstructed_phase()[0])
        self._logger.info('Petals %s' % petals)
        self.assertAlmostEqual(123, (petals[1]-petals[0]).to_value(u.nm))
        self.assertEqual(1, lep2.time_step)


    def _temp_fname(self):
        return os.path.join(tempfile.gettempdir(), 'lep.fits')


if __name__ == "__main__":
    # import sys;sys.argv = ['', 'Test.testName']
    unittest.main()
