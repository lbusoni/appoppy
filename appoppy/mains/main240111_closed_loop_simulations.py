import logging
import numpy as np
from astropy import units as u
from appoppy.simulation_results import SimulationResults
from appoppy.long_exposure_simulation import ClosedLoopSimulation, long_exposure_filename, long_exposure_tracknum
import matplotlib.pyplot as plt
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

TN_MCAO_1_PS = '20231209_202232.0_coo55.0_0.0_ps'
TN_MCAO_2_PS = '20231209_202232.0_coo55.0_120.0_ps'
TN_DAO_1_PS = '20231209_202232.0_coo55.0_0.0DAO_ps'
TN_SCAO_1000_PS = '20231212_212912.0_coo0.0_0.0_ps'
TN_SCAO_2000_PS = '20231213_101833.0_coo0.0_0.0_ps'
TN_REF_500_PS = '20231213_123051.0_coo0.0_0.0_ps'
TN_REF_100_PS = '20231213_123200.0_coo0.0_0.0_ps'
TN_REF_10_PS = '20231213_123403.0_coo0.0_0.0_ps'

# TN_MCAO_1_LWE_1 = '20240109_171236.0_coo55.0_0.0'  # oDAO?
# TN_REF_500_LWE_1 = '20240110_092253.0_coo0.0_0.0'
# TN_SCA0_2000_LWE_1 = '20240109_165105.0_coo0.0_0.0'


def _create_closed_loop_generic(tn,
                                code,
                                rot_angle=60,
                                petals=np.array([0, 0, 0, 0, 0, 0]) * u.nm,
                                wavelength=1650 * u.nm,
                                n_iter=1000):
    _setUpBasicLogging()
    if _result_exists(long_exposure_tracknum(tn, code)):
        logging.getLogger('create_generic').warning(
            'Skipping existing file %s' % long_exposure_tracknum(tn, code))
        return
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


def _result_exists(simul_tn):
    filename = long_exposure_filename(simul_tn)
    from pathlib import Path
    my_file = Path(filename)
    if my_file.is_file():
        return True
    return False


def _setUpBasicLogging():
    import importlib
    import logging
    importlib.reload(logging)
    logging.basicConfig(level=logging.INFO)
    logging.getLogger('poppy').setLevel(logging.ERROR)


def _petal_scrambles():
    '''
    10 scramble occurrence (ref. email 08/01/2024).
    '''
    return np.array([
        [-10, -130, 90, -70, 150, -350],
        [130, 40, -230, 160, -40, -150],
        [0, 50, 40, 140, -350, 0],
        [10, 120, -150, 10, 30, 70],
        [-80, 120, -80, -140, -210, -90],
        [10, 90, 80, -40, 40, -170],
        [30, 130, -230, -40, 20, 120],
        [-30, -20, -180, 180, 300, -140],
        [-30, -90, 80, 30, 70, -50],
        [-10, -80, 0, -100, 0, -100],
    ]) * u.nm


def _code_petals_dictionary():
    seqs = {'1000': np.array([0, 0, 0, 0, 0, 0]) * u.nm,
            '1002': np.array([0, 0, 0, 0, 400, 0]) * u.nm,
            '1003': np.array([0, 0, 0, 0, 200, 0]) * u.nm
            }
    for idx, pp in enumerate(_petal_scrambles()):
        code = str(1004 + idx)
        seqs[code] = pp
    return seqs


def create_all_petals_sequences_for(tn):
    code_petals_dict = _code_petals_dictionary()
    for code, pp in code_petals_dict.items():
        logging.getLogger('main').info(
            'Creating %s %s - petals %s' % (tn, code, pp))
        _create_closed_loop_generic(tn, code, petals=pp)


def create_all():
    create_none_1002()
    create_all_petals_sequences_for(TN_MCAO_1_PS)
    create_all_petals_sequences_for(TN_SCAO_2000_PS)
    create_all_petals_sequences_for(TN_REF_10_PS)


def create_none_1002():
    _create_closed_loop_generic(TN_NONE, '1002', n_iter=100,
                                       petals=np.array([0, 0, 0, 0, 400, 0]) * u.nm)

# def create_mcao_1_1000():
#     _create_closed_loop_generic(
#         TN_MCAO_1, '1000', petals=np.array([0, 0, 0, 0, 0, 0]) * u.nm)

# def create_mcao_1_1002():
#     _create_closed_loop_generic(
#         TN_MCAO_1, '1002', petals=np.array([0, 0, 0, 0, 400, 0]) * u.nm)

# def create_mcao_1_1003():
#     _create_closed_loop_generic(
#         TN_MCAO_1, '1003', petals=np.array([0, 0, 0, 0, 200, 0]) * u.nm)

# def create_mcao_1_1004():
#     '''
#     0th scramble occurrence (ref. email 08/01/2024).
#     '''
#     return _create_closed_loop_generic(
#         TN_MCAO_1, '1004',
#         petals=np.array([-10, -130, 90, -70, 150, -350]) * u.nm)

# def create_mcao_1_1005():
#     '''
#     1st scramble occurrence (ref. email 08/01/2024).
#     '''
#     return _create_closed_loop_generic(
#         TN_MCAO_1, '1005',
#         petals=np.array([130, 40, -230, 160, -40, -150]) * u.nm)

# def create_mcao_1_1006():
#     '''
#     2nd scramble occurrence (ref. email 08/01/2024).
#     '''
#     return _create_closed_loop_generic(
#         TN_MCAO_1, '1006',
#         petals=np.array([0, 50, 40, 140, -350, 0]) * u.nm)

# def create_mcao_1_1007():
#     '''
#     3rd scramble occurrence (ref. email 08/01/2024).
#     '''
#     return _create_closed_loop_generic(
#         TN_MCAO_1, '1007',
#         petals=np.array([10, 120, -150, 10, 30, 70]) * u.nm)

# def create_mcao_1_1008():
#     '''
#     4th scramble occurrence (ref. email 08/01/2024).
#     '''
#     return _create_closed_loop_generic(
#         TN_MCAO_1, '1008',
#         petals=np.array([-80, 120, -80, -140, -210, -90]) * u.nm)

# def create_mcao_1_1009():
#     '''
#     5th scramble occurrence (ref. email 08/01/2024).
#     '''
#     return _create_closed_loop_generic(
#         TN_MCAO_1, '1009',
#         petals=np.array([10, 90, 80, -40, 40, -170]) * u.nm)

# def create_mcao_1_1010():
#     '''
#     6th scramble occurrence (ref. email 08/01/2024).
#     '''
#     return _create_closed_loop_generic(
#         TN_MCAO_1, '1010',
#         petals=np.array([30, 130, -230, -40, 20, 120]) * u.nm)

# def create_mcao_1_1011():
#     '''
#     7th scramble occurrence (ref. email 08/01/2024).
#     '''
#     return _create_closed_loop_generic(
#         TN_MCAO_1, '1011',
#         petals=np.array([-30, -20, -180, 180, 300, -140]) * u.nm)

# def create_mcao_1_1012():
#     '''
#     8th scramble occurrence (ref. email 08/01/2024).
#     '''
#     return _create_closed_loop_generic(
#         TN_MCAO_1, '1012',
#         petals=np.array([-30, -90, 80, 30, 70, -50]) * u.nm)

# def create_mcao_1_1013():
#     '''
#     9th scramble occurrence (ref. email 08/01/2024).
#     '''
#     return _create_closed_loop_generic(
#         TN_MCAO_1, '1013',
#         petals=np.array([-10, -80, 0, -100, 0, -100]) * u.nm)


def analyze_mcao_1_1002():
    return _analyze_simul_results(TN_MCAO_1, '1002')


def analyze_mcao_1_1003():
    return _analyze_simul_results(TN_MCAO_1, '1003')


def analyze_mcao_1_1004():
    return _analyze_simul_results(TN_MCAO_1_PS, '1004')


def analyze_mcao_1_1005():
    return _analyze_simul_results(TN_MCAO_1, '1005')


def analyze_mcao_1_1006():
    return _analyze_simul_results(TN_MCAO_1, '1006')


def analyze_mcao_1_1007():
    return _analyze_simul_results(TN_MCAO_1, '1007')


def analyze_mcao_1_1008():
    return _analyze_simul_results(TN_MCAO_1, '1008')


def analyze_mcao_1_1009():
    return _analyze_simul_results(TN_MCAO_1, '1009')


def analyze_mcao_1_1010():
    return _analyze_simul_results(TN_MCAO_1, '1010')


def analyze_mcao_1_1011():
    return _analyze_simul_results(TN_MCAO_1, '1011')


def analyze_mcao_1_1012():
    return _analyze_simul_results(TN_MCAO_1, '1012')


def analyze_mcao_1_1013():
    return _analyze_simul_results(TN_MCAO_1, '1013')


def analyze_mcao_1_ps_1002():
    return _analyze_simul_results(TN_MCAO_1_PS, '1002')


def analyze_mcao_1_ps_1003():
    return _analyze_simul_results(TN_MCAO_1_PS, '1003')


def analyze_mcao_1_ps_1004():
    return _analyze_simul_results(TN_MCAO_1_PS, '1004')


def analyze_mcao_1_ps_1005():
    return _analyze_simul_results(TN_MCAO_1_PS, '1005')


def analyze_mcao_1_ps_1006():
    return _analyze_simul_results(TN_MCAO_1_PS, '1006')


def analyze_mcao_1_ps_1007():
    return _analyze_simul_results(TN_MCAO_1_PS, '1007')


def analyze_mcao_1_ps_1008():
    return _analyze_simul_results(TN_MCAO_1_PS, '1008')


def analyze_mcao_1_ps_1009():
    return _analyze_simul_results(TN_MCAO_1_PS, '1009')


def analyze_mcao_1_ps_1010():
    return _analyze_simul_results(TN_MCAO_1_PS, '1010')


def analyze_mcao_1_ps_1011():
    return _analyze_simul_results(TN_MCAO_1_PS, '1011')


def analyze_mcao_1_ps_1012():
    return _analyze_simul_results(TN_MCAO_1_PS, '1012')


def analyze_mcao_1_ps_1013():
    return _analyze_simul_results(TN_MCAO_1_PS, '1013')

#
# def _analyze_long_exposure(tracknum, code):
#     '''
#     Measure and compensate for petals on MORFEO residuals
#     '''
#     le = SimulationResults.load(long_exposure_tracknum(tracknum, code))
#     std_input = le.input_opd()[20:].std(axis=(1, 2))
#     std_corr_inst = le.corrected_opd()[20:].std(axis=(1, 2))
#     std_corr_long = le.corrected_opd_from_reconstructed_phase_ave()[
#         20:].std(axis=(1, 2))
#     _plot_stdev_residual(le, title="%s %s" % (tracknum, code))
#     print('\nMean of MORFEO residuals stds: %s' % std_input.mean())
#     print(
#         'Mean of MORFEO residuals (with long exposure petal correction) stds: %s'
#         % std_corr_long.mean())
#     print(
#         'Mean of MORFEO residuals (with short exposure petal correction) stds: %s'
#         % std_corr_inst.mean())
#
#     petals, jumps = le.petals_from_reconstructed_phase_map(
#         le.reconstructed_phase_ave())
#     print('\nMeasured petals: %s' % petals)
#     print('Measured jumps: %s' % jumps[::2])
#
#     return le


def _analyze_simul_results(tracknum, code_scramble):
    '''
    Measure and compensate for petals on MORFEO residuals
    '''
    code_noscramble = '0000'
    le_scramble = SimulationResults.load(
        long_exposure_tracknum(tracknum, code_scramble))
    le_noscramble = SimulationResults.load(
        long_exposure_tracknum(tracknum, code_noscramble))
    std_input = le_noscramble.input_opd()[20:].std(axis=(1, 2))
    std_corr_inst = le_scramble.corrected_opd()[20:].std(axis=(1, 2))
#    std_corr_long = le_scramble.corrected_opd_from_reconstructed_phase_ave()[20:].std(axis=(1, 2))
    _plot_stdev_residual(le_scramble, le_noscramble,
                           title="%s %s" % (tracknum, code_scramble))
    print('\nMean of MORFEO residuals stds: %s' % std_input.mean())
#    print('Mean of MORFEO residuals (with long exposure petal correction) stds: %s'
#        % std_corr_long.mean())
    print(
        'Mean of MORFEO residuals (with short exposure petal correction) stds: %s'
        % std_corr_inst.mean())

    petals, jumps = le_scramble.petals_from_reconstructed_phase_map(
        le_scramble.reconstructed_phase_ave())
    print('\nMeasured petals: %s' % petals)
    print('Measured jumps: %s' % jumps[::2])

    return le_scramble


def _stdev_after_transient(what):
    return what[20:].std(axis=(1, 2))

# def _plot_stdev_residual(le, title=''):
#     std_input = _stdev_after_transient(le.input_opd())
#     std_corr_inst = _stdev_after_transient(le.corrected_opd())
#     std_corr_long = _stdev_after_transient(
#         le.corrected_opd_from_reconstructed_phase_ave())
#     timev = np.arange(len(std_input)) * le.time_step
#     plt.figure()
#     plt.plot(timev, std_input, label=r'Petalometer Off $\sqrt{\sigma_{off}}$')
#     plt.plot(timev, std_corr_inst,
#              label='Petalometer On short exposure $\sqrt{\sigma_{short}}$')
#     plt.plot(timev, std_corr_long,
#              label='Petalometer On long exposure $\sqrt{\sigma_{long}}$')
#     quadr_diff_inst = quadraticSum([std_input, -std_corr_inst])
#     quadr_diff_long = quadraticSum([std_input, -std_corr_long])
#     plt.plot(timev, quadr_diff_inst,
#              label=r'$\sqrt{\sigma_{off}^2-\sigma_{short}^2}$')
#     plt.plot(timev, quadr_diff_long,
#              label=r'$\sqrt{\sigma_{off}^2-\sigma_{long}^2}$')
#     plt.legend()
#     plt.grid()
#     plt.ylabel('Std [nm]')
#     plt.xlabel('Time [s]')
#     plt.title(title)


def _plot_stdev_residual(le_scramble, le_noscramble, title=''):
    std_input = _stdev_after_transient(le_noscramble.input_opd())
    std_corr_inst = _stdev_after_transient(le_scramble.corrected_opd())
    # std_corr_long = _stdev_after_transient(
    #     le_scramble.corrected_opd_from_reconstructed_phase_ave())
    timev = np.arange(len(std_input)) * le_noscramble.time_step
    plt.figure()
    plt.plot(timev, std_input,
             label=r'Petalometer Off - No scramble $\sqrt{\sigma_{off}}$')
    plt.plot(timev, std_corr_inst,
             label='Petalometer On short exposure $\sqrt{\sigma_{short}}$')
    # plt.plot(timev, std_corr_long,
    #          label='Petalometer On long exposure $\sqrt{\sigma_{long}}$')
    quadr_diff_inst = quadraticSum([std_corr_inst, -std_input])
    # quadr_diff_long = quadraticSum([std_corr_long, -std_input])
    # quadr_diff_inst = np.sqrt(
    #     std_corr_inst ** 2 - std_input ** 2)
    # quadr_diff_long = np.sqrt(
    #     std_corr_long ** 2 - std_input ** 2)
    print('\nResidual short [nm]: %s' % (quadr_diff_inst.mean()))
    # print('\nResidual long [nm]: %s' % (quadr_diff_long.mean()))
    plt.plot(timev, quadr_diff_inst,
             label=r'$\sqrt{\sigma_{off}^2-\sigma_{short}^2}$')
    # plt.plot(timev, quadr_diff_long,
    #          label=r'$\sqrt{\sigma_{off}^2-\sigma_{long}^2}$')
    plt.legend()
    plt.grid()
    plt.ylabel('Std [nm]')
    plt.xlabel('Time [s]')
    plt.title(title)
