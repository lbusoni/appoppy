import logging
import numpy as np
from astropy import units as u
from appoppy.mains.main240112_convert_passata_residuals_with_petal_subtraction import KnownTracknums
from appoppy.simulation_results import SimulationResults
from appoppy.long_exposure_simulation import ClosedLoopSimulation, long_exposure_filename, long_exposure_tracknum
import matplotlib.pyplot as plt
from arte.utils.quadratic_sum import quadraticSum




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
    create_all_petals_sequences_for(KnownTracknums.TN_MCAO_1_PS)
    create_all_petals_sequences_for(KnownTracknums.TN_SCAO_2000_PS)
#    create_all_petals_sequences_for(KnownTracknums.TN_REF_10_PS)
    create_some_petals_sequences_for(KnownTracknums.TN_MCAO_1)
    create_some_petals_sequences_for(KnownTracknums.TN_SCAO_2000)
    create_some_petals_sequences_for(KnownTracknums.TN_MCAO_LWE_1)
    create_some_petals_sequences_for(KnownTracknums.TN_SCAO_2000_LWE)


def create_none_1002():
    _create_closed_loop_generic(KnownTracknums.TN_NONE, '1002', n_iter=100,
                                       petals=np.array([0, 0, 0, 0, 400, 0]) * u.nm)


# analyze no ps

def analyze_mcao_1(code):
    'res = analyze_mcao_1(1002)'
    return _analyze_simul_results(KnownTracknums.TN_MCAO_1, str(code))


# analyze ps
def analyze_mcao_1_ps(code):
    'res = analyze_mcao_1_ps(1002)'
    return _analyze_simul_results(KnownTracknums.TN_MCAO_1_PS, str(code))


def analyze_scao_2000_ps(code):
    return _analyze_simul_results(KnownTracknums.TN_SCAO_2000_PS, str(code))

# analyze lwe

def analyze_mcao_lwe(code):
   return _analyze_simul_results_cedric_style(KnownTracknums.TN_MCAO_LWE_1, str(code)) 

# analyze mcao on axis
def analyze_mcao_0():
    return _analyze_simul_results_onaxis(KnownTracknums.TN_MCAO_1, '1000',
                                         KnownTracknums.TN_MCAO_0, '0000')

def _analyze_simul_results(passata_tracknum, code_with_petals):
    code_no_petals = '0000'
    if passata_tracknum[-3:] != '_ps':
        passata_tracknum_no_petals = passata_tracknum + '_ps'
    else:
        passata_tracknum_no_petals = passata_tracknum
    le_with_petals = SimulationResults.load(
        long_exposure_tracknum(passata_tracknum, code_with_petals))
    le_no_petals = SimulationResults.load(
        long_exposure_tracknum(passata_tracknum_no_petals, code_no_petals))
    skip_steps = 20
    std_input = le_no_petals.input_opd()[skip_steps:].std(axis=(1, 2))
    std_corr_inst = le_with_petals.corrected_opd()[skip_steps:].std(axis=(1, 2))
    _plot_stdev_residual(le_with_petals, le_no_petals,
                         title="%s %s" % (passata_tracknum, code_with_petals))
    print('\nMean of MORFEO residuals stds: %s' % std_input.mean())
    print(
        'Mean of MORFEO residuals (with short exposure petal correction) stds: %s'
        % std_corr_inst.mean())

    petals, jumps = le_with_petals.petals_from_reconstructed_phase_map(
        le_with_petals.reconstructed_phase_ave())
    print('\nMeasured petals: %s' % petals)
    print('Measured jumps: %s' % jumps[::2])

    return le_with_petals

def _analyze_simul_results_cedric_style(passata_tracknum, code_with_petals):
    code_no_petals = '0000'
    if passata_tracknum[-3:] != '_ps':
        passata_tracknum_no_petals = passata_tracknum + '_ps'
    else:
        passata_tracknum_no_petals = passata_tracknum
    le_with_petals = SimulationResults.load(
        long_exposure_tracknum(passata_tracknum, code_with_petals))
    le_no_petals = SimulationResults.load(
        long_exposure_tracknum(passata_tracknum_no_petals, code_no_petals))
    skip_steps = 20
    std_input_raw_corrected = le_no_petals.input_opd()[skip_steps:].std(axis=(1, 2))
    std_corr_inst = le_with_petals.corrected_opd()[skip_steps:].std(axis=(1, 2))
    _plot_stdev_residual_cedric_style(le_with_petals, le_no_petals,
                         title="%s %s" % (passata_tracknum, code_with_petals))
    print('\nMean of MORFEO residuals stds: %s' % std_input_raw_corrected.mean())
    print(
        'Mean of MORFEO residuals (with short exposure petal correction) stds: %s'
        % std_corr_inst.mean())

    petals, jumps = le_with_petals.petals_from_reconstructed_phase_map(
        le_with_petals.reconstructed_phase_ave())
    print('\nMeasured petals: %s' % petals)
    print('Measured jumps: %s' % jumps[::2])

    return le_with_petals


def _analyze_simul_results_onaxis(passata_tracknum_offaxis, code_cl_no_petals, passata_tracknum_onaxis, code_ol_no_petals):
    '''
    Compute the petals correction from off-axis AO residuals and apply the correction on axis.
    '''
    skip_steps = 20
    le_onaxis = SimulationResults.load(long_exposure_tracknum(passata_tracknum_onaxis, code_ol_no_petals))
    input_opd_onaxis = le_onaxis.input_opd()[skip_steps:]
    std_input_onaxis = input_opd_onaxis.std(axis=(1, 2))
    le_offaxis = SimulationResults.load(
        long_exposure_tracknum(passata_tracknum_offaxis, code_cl_no_petals))
    input_opd_offaxis = le_offaxis.input_opd()[skip_steps:]
    std_input_offaxis = input_opd_offaxis.std(axis=(1, 2))
    corrected_opd_offaxis = le_offaxis.corrected_opd()[skip_steps:] 
    correction_opd_from_offaxis = input_opd_offaxis - corrected_opd_offaxis
    std_corrected_offaxis = corrected_opd_offaxis.std(axis=(1, 2))
    corrected_opd_onaxis = input_opd_onaxis - correction_opd_from_offaxis
    std_corrected_onaxis = corrected_opd_onaxis.std(axis=(1, 2))
    timev = np.arange(len(std_input_onaxis)) * le_onaxis.time_step
    plt.figure()
    plt.plot(timev, std_input_onaxis, label='Input on axis')
    plt.plot(timev, std_corrected_onaxis, label='Residual on axis')
    plt.plot(timev, std_input_offaxis, label='Input off axis')
    plt.plot(timev, std_corrected_offaxis, label='Residual off axis')
    plt.legend()
    plt.grid()
    plt.ylabel('Std [nm]')
    plt.xlabel('Time [s]')
    plt.title("%s %s" % (passata_tracknum_onaxis, code_ol_no_petals))


def _stdev_after_transient(what):
    return what[20:].std(axis=(1, 2))


def _plot_stdev_residual(le_with_petals, le_no_petals, title=''):
    std_input = _stdev_after_transient(le_no_petals.input_opd())
    std_corr_inst = _stdev_after_transient(le_with_petals.corrected_opd())
    timev = np.arange(len(std_input)) * le_no_petals.time_step
    plt.figure()
    plt.plot(timev, std_input,
             label=r'Petalometer Off - No petals $\sqrt{\sigma_{off}}$=%d nm'% std_input.mean())
    plt.plot(timev, std_corr_inst,
             label='Petalometer On short exposure $\sqrt{\sigma_{short}}$=%d nm' % std_corr_inst.mean())
    quadr_diff_inst = quadraticSum([std_corr_inst, -std_input])
    print('\nResidual short [nm]: %s' % (quadr_diff_inst.mean()))
    plt.plot(timev, quadr_diff_inst,
             label=r'$\sqrt{\sigma_{short}^2 - \sigma_{off}^2}$=%d nm' % quadr_diff_inst.mean())
    plt.legend()
    plt.grid()
    plt.ylabel('Std [nm]')
    plt.xlabel('Time [s]')
    plt.title(title)


def _plot_stdev_residual_cedric_style(le_with_petals, le_no_petals, title=''):
    std_input_raw_corrected = _stdev_after_transient(le_no_petals.input_opd())
    std_input_raw_pist = _stdev_after_transient(le_with_petals.input_opd())
    std_corr_inst = _stdev_after_transient(le_with_petals.corrected_opd())
    timev = np.arange(len(std_input_raw_corrected)) * le_no_petals.time_step
    plt.figure()
    plt.plot(timev, std_input_raw_pist, label='Raw + pist (LWE)', color='red')
    plt.plot(timev, std_input_raw_corrected, label='Raw corrected', color='green')
    plt.plot(timev, std_corr_inst, label='Raw + pist (LWE) corrected', color='blue')
    quadr_diff_inst = quadraticSum([std_corr_inst, -std_input_raw_corrected])
    print('\nResidual short [nm]: %s' % (quadr_diff_inst.mean()))
    plt.legend()
    plt.grid()
    plt.ylabel('WFE [nm]')
    plt.xlabel('Time [s]')
    plt.title(title)
