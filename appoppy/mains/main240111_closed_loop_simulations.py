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
TN_MCAO_LWE_1 = '20240109_171236.0_coo55.0_0.0'  # MCAO or DAO?
TN_SCAO_2000_LWE = '20240109_165105.0_coo0.0_0.0'

TN_MCAO_1_PS = '20231209_202232.0_coo55.0_0.0_ps'
TN_MCAO_2_PS = '20231209_202232.0_coo55.0_120.0_ps'
TN_DAO_1_PS = '20231209_202232.0_coo55.0_0.0DAO_ps'
TN_SCAO_1000_PS = '20231212_212912.0_coo0.0_0.0_ps'
TN_SCAO_2000_PS = '20231213_101833.0_coo0.0_0.0_ps'
TN_REF_500_PS = '20231213_123051.0_coo0.0_0.0_ps'
TN_REF_100_PS = '20231213_123200.0_coo0.0_0.0_ps'
TN_REF_10_PS = '20231213_123403.0_coo0.0_0.0_ps'


# TN_REF_500_LWE_1 = '20240110_092253.0_coo0.0_0.0'


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
    ])*u.nm


def _code_petals_dictionary():
    seqs = {'1000': np.array([0, 0, 0, 0, 0, 0]) * u.nm,
            '1002': np.array([0, 0, 0, 0, 400, 0]) * u.nm,
            '1003': np.array([0, 0, 0, 0, 200, 0]) * u.nm
            }
    for idx, pp in enumerate(_petal_scrambles()):
        code = str(1004+idx)
        seqs[code] = pp
    return seqs


def create_all_petals_sequences_for(tn):
    code_petals_dict = _code_petals_dictionary()
    for code, pp in code_petals_dict.items():
        logging.getLogger('main').info(
            'Creating %s %s - petals %s' % (tn, code, pp))
        _create_closed_loop_generic(tn, code, petals=pp)


def create_some_petals_sequences_for(tn):
    ss = _code_petals_dictionary()
    code_petals_dict = {key: value for (
        key, value) in ss.items() if int(key) < 1007}
    for code, pp in code_petals_dict.items():
        logging.getLogger('main').info(
            'Creating %s %s - petals %s' % (tn, code, pp))
        _create_closed_loop_generic(tn, code, petals=pp)


def create_all():
    create_none_1002()
    create_all_petals_sequences_for(TN_MCAO_1_PS)
    create_all_petals_sequences_for(TN_SCAO_2000_PS)
#    create_all_petals_sequences_for(TN_REF_10_PS)
    create_some_petals_sequences_for(TN_MCAO_1)
    create_some_petals_sequences_for(TN_SCAO_2000)
    create_some_petals_sequences_for(TN_MCAO_LWE_1)
    create_some_petals_sequences_for(TN_SCAO_2000_LWE)


def create_none_1002():
    _create_closed_loop_generic(TN_NONE, '1002', n_iter=100,
                                       petals=np.array([0, 0, 0, 0, 400, 0]) * u.nm)


# analyze no ps

def analyze_mcao_1_1002():
    return _analyze_simul_results(TN_MCAO_1, '1002', '1000')


def analyze_mcao_1_1003():
    return _analyze_simul_results(TN_MCAO_1, '1003', '1000')


def analyze_mcao_1_1004():
    return _analyze_simul_results(TN_MCAO_1, '1004', '1000')


# analyze ps

def analyze_mcao_1_ps_1002():
    return _analyze_simul_results(TN_MCAO_1_PS, '1002', '1000')


def analyze_mcao_1_ps_1003():
    return _analyze_simul_results(TN_MCAO_1_PS, '1003', '1000')


def analyze_mcao_1_ps_1004():
    return _analyze_simul_results(TN_MCAO_1_PS, '1004', '1000')


def analyze_mcao_1_ps_1005():
    return _analyze_simul_results(TN_MCAO_1_PS, '1005', '0000')


def analyze_mcao_1_ps_1006():
    return _analyze_simul_results(TN_MCAO_1_PS, '1006', '0000')


def analyze_mcao_1_ps_1007():
    return _analyze_simul_results(TN_MCAO_1_PS, '1007', '0000')


def analyze_scao_2000_ps_1002():
    return _analyze_simul_results(TN_SCAO_2000_PS, '1002', '1000')


def analyze_scao_2000_ps_1003():
    return _analyze_simul_results(TN_SCAO_2000_PS, '1003', '1000')


def analyze_scao_2000_ps_1004():
    return _analyze_simul_results(TN_SCAO_2000_PS, '1004', '1000')


def _analyze_simul_results(passata_converted_tracknum, code_with_petals, code_without_petals):
    '''
    
    '''
    le_scramble = SimulationResults.load(
        long_exposure_tracknum(passata_converted_tracknum, code_with_petals))
    le_noscramble = SimulationResults.load(
        long_exposure_tracknum(passata_converted_tracknum, code_without_petals))
    skip_steps = 20
    std_input = le_noscramble.input_opd()[skip_steps:].std(axis=(1, 2))
    std_corr_inst = le_scramble.corrected_opd()[skip_steps:].std(axis=(1, 2))
#    std_corr_long = le_scramble.corrected_opd_from_reconstructed_phase_ave()[20:].std(axis=(1, 2))
    _plot_stdev_residual(le_scramble, le_noscramble,
                         title="%s %s" % (passata_converted_tracknum, code_with_petals))
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


def _plot_stdev_residual(le_scramble, le_no_petals, title=''):
    std_input = _stdev_after_transient(le_no_petals.input_opd())
    std_corr_inst = _stdev_after_transient(le_scramble.corrected_opd())
    std_corr_long = _stdev_after_transient(
        le_scramble.corrected_opd_from_reconstructed_phase_ave())
    timev = np.arange(len(std_input)) * le_no_petals.time_step
    plt.figure()
    plt.plot(timev, std_input,
             label=r'Petalometer Off - No scramble $\sqrt{\sigma_{off}}$')
    plt.plot(timev, std_corr_inst,
             label='Petalometer On short exposure $\sqrt{\sigma_{short}}$')
    plt.plot(timev, std_corr_long,
             label='Petalometer On long exposure $\sqrt{\sigma_{long}}$')
    quadr_diff_inst = quadraticSum([std_input, -std_corr_inst])
    quadr_diff_long = quadraticSum([std_input, -std_corr_long])
    # quadr_diff_inst = np.sqrt(np.mean(std_corr_inst ** 2 - std_input ** 2))
    # quadr_diff_long = np.sqrt(np.mean(std_corr_inst ** 2 - std_input ** 2))
    print('\nResidual short [nm]: %s' % (quadr_diff_inst.mean()))
    print('\nResidual long [nm]: %s' % (quadr_diff_long.mean()))
    plt.plot(timev, quadr_diff_inst,
             label=r'$\sqrt{\sigma_{off}^2-\sigma_{short}^2}$')
    plt.plot(timev, quadr_diff_long,
             label=r'$\sqrt{\sigma_{off}^2-\sigma_{long}^2}$')
    plt.legend()
    plt.grid()
    plt.ylabel('Std [nm]')
    plt.xlabel('Time [s]')
    plt.title(title)
