import logging
import os
import numpy as np
from astropy import units as u
from appoppy.simulation_results import SimulationResults
from appoppy.long_exposure_simulation import ClosedLoopSimulation, LongExposureSimulation, long_exposure_filename, long_exposure_tracknum
import matplotlib.pyplot as plt
from arte.utils.marechal import wavefront_rms_2_strehl_ratio
from arte.utils.quadratic_sum import quadraticSum

TN_NONE = 'none'
TN_MCAO_1 = '20231209_202232.0_coo55.0_0.0'
TN_MCAO_2 = '20231209_202232.0_coo55.0_120.0'
TN_DAO_1 = '20231209_202232.0_coo55.0_0.0DAO'
TN_SCAO_1000 = '20231212_212912.0_coo0.0_0.0'
TN_SCAO_2000 = '20231213_101833.0_coo0.0_0.0'
TN_REF_500 = '20231213_123051.0_coo0.0_0.0'
TN_REF_100 = '20231213_123200.0_coo0.0_0.0'
TN_REF_10 = '20231213_123403.0_coo0.0_0.0'


def _create_closed_loop_generic(tn,
                                code,
                                rot_angle=60,
                                petals=np.array([0, 0, 0, 0, 0, 0]) * u.nm,
                                wavelength=1650 * u.nm,
                                n_iter=1000):
    _setUpBasicLogging()
    le = ClosedLoopSimulation(
        long_exposure_tracknum(tn, code),
        tn,
        rot_angle=rot_angle,
        petals=petals,
        wavelength=wavelength,
        n_iter=n_iter,
        gain=0.5)
    le.run()
    le.save()
    return le


def _setUpBasicLogging():
    logging.basicConfig(level=logging.WARNING)
    logging.getLogger('poppy').setLevel(logging.ERROR)


def create_all():
    create_mcao_1_1002()
    create_none_1002()


def create_none_1002():
    return _create_closed_loop_generic(TN_NONE, '1002', n_iter=100,
                                       petals=np.array([0, 0, 0, 0, 400, 0]) * u.nm)


def create_mcao_1_1002():
    return _create_closed_loop_generic(TN_MCAO_1, '1002', petals=np.array([0, 0, 0, 0, 400, 0]) * u.nm)


def _analyze_long_exposure(tracknum, code):
    '''
    Measure and compensate for petals on MORFEO residuals
    '''
    le = SimulationResults.load(long_exposure_tracknum(tracknum, code))
    std_input = le.input_opd()[20:].std(axis=(1, 2))
    std_corr_inst = le.corrected_opd()[20:].std(axis=(1, 2))
    std_corr_long = le.corrected_opd_from_reconstructed_phase_ave()[
        20:].std(axis=(1, 2))
    _plot_stdev_residual(le, title="%s %s" % (tracknum, code))
    print('\nMean of MORFEO residuals stds: %s' % std_input.mean())
    print(
        'Mean of MORFEO residuals (with long exposure petal correction) stds: %s'
        % std_corr_long.mean())
    print(
        'Mean of MORFEO residuals (with short exposure petal correction) stds: %s'
        % std_corr_inst.mean())

    petals, jumps = le.petals_from_reconstructed_phase_map(
        le.reconstructed_phase_ave())
    print('\nMeasured petals: %s' % petals)
    print('Measured jumps: %s' % jumps[::2])

    return le


def _stdev_after_transient(what):
    return what[20:].std(axis=(1, 2))


def _plot_stdev_residual(le, title=''):
    std_input = _stdev_after_transient(le.input_opd())
    std_corr_inst = _stdev_after_transient(le.corrected_opd())
    std_corr_long = _stdev_after_transient(
        le.corrected_opd_from_reconstructed_phase_ave())
    timev = np.arange(len(std_input)) * le.time_step
    plt.figure()
    plt.plot(timev, std_input, label=r'Petalometer Off $\sqrt{\sigma_{off}}$')
    plt.plot(timev, std_corr_inst,
             label='Petalometer On short exposure $\sqrt{\sigma_{short}}$')
    plt.plot(timev, std_corr_long,
             label='Petalometer On long exposure $\sqrt{\sigma_{long}}$')
    quadr_diff_inst = quadraticSum([std_input, -std_corr_inst])
    quadr_diff_long = quadraticSum([std_input, -std_corr_long])
    plt.plot(timev, quadr_diff_inst,
             label=r'$\sqrt{\sigma_{off}^2-\sigma_{short}^2}$')
    plt.plot(timev, quadr_diff_long,
             label=r'$\sqrt{\sigma_{off}^2-\sigma_{long}^2}$')
    plt.legend()
    plt.grid()
    plt.ylabel('Std [nm]')
    plt.xlabel('Time [s]')
    plt.title(title)
