
import os
import numpy as np
from astropy import units as u
from appoppy.long_exposure import LongExposurePetalometer, long_exposure_filename, long_exposure_tracknum
import matplotlib.pyplot as plt
from arte.utils.marechal import wavefront_rms_2_strehl_ratio
from arte.utils.quadratic_sum import quadraticSum

from appoppy.package_data import data_root_dir

# ROOT_DIR = '/Users/lbusoni/Library/CloudStorage/GoogleDrive-lorenzo.busoni@inaf.it/Il mio Drive/adopt/Varie/CiaoCiaoWFS/analysis/data/from_appoppy/'

TN_NONE = 'none'
TN_MCAO_1 = '20231209_202232.0_coo55.0_0.0'
TN_MCAO_2 = '20231209_202232.0_coo55.0_120.0'
TN_DAO_1 = '20231209_202232.0_coo55.0_0.0DAO'
TN_SCAO_1000 = '20231212_212912.0_coo0.0_0.0'
TN_SCAO_2000 = '20231213_101833.0_coo0.0_0.0'
TN_REF_500 = '20231213_123051.0_coo0.0_0.0'
TN_REF_100 = '20231213_123200.0_coo0.0_0.0'
TN_REF_10 = '20231213_123403.0_coo0.0_0.0'





def _create_long_exposure_generic(tn,
                                  code,
                                  rot_angle=60,
                                  petals=np.array([0, 0, 0, 0, 0, 0]) * u.nm,
                                  wavelength=1650*u.nm,
                                  n_iter=1000):
    le = LongExposurePetalometer(
        long_exposure_tracknum(tn, code),
        tn,
        rot_angle=rot_angle,
        petals=petals,
        wavelength=wavelength,
        n_iter=n_iter)
    le.run()
    le.save(long_exposure_tracknum(tn, code))
    return le


def create_all():
    create_long_exposure_mcao_1_0000()
    create_long_exposure_mcao_1_0002()
    create_long_exposure_dao_1_0000()
    create_long_exposure_dao_1_0002()
    create_long_exposure_none_0000()
    create_long_exposure_none_0002()
    create_long_exposure_scao_1000_0000()
    create_long_exposure_scao_1000_0002()
    create_long_exposure_scao_2000_0000()
    create_long_exposure_scao_2000_0002()
    create_long_exposure_ref_500_0000()
    create_long_exposure_ref_500_0002()
    create_long_exposure_ref_100_0000()
    create_long_exposure_ref_100_0002()
    create_long_exposure_ref_10_0000()
    create_long_exposure_ref_10_0002()


def create_long_exposure_none_0000():
    return _create_long_exposure_generic(TN_NONE, '0000', n_iter=100)


def create_long_exposure_none_0002():
    return _create_long_exposure_generic(TN_NONE, '0002', n_iter=100,
                                         petals=np.array([0, 0, 0, 0, 400, 0]) * u.nm)


def create_long_exposure_mcao_1_0000():
    return _create_long_exposure_generic(TN_MCAO_1, '0000')


def create_long_exposure_mcao_1_0001():
    return _create_long_exposure_generic(TN_MCAO_1, '0001', petals=np.array([0, 0, 0, 0, 100, 0]) * u.nm)


def create_long_exposure_mcao_1_0002():
    return _create_long_exposure_generic(TN_MCAO_1, '0002', petals=np.array([0, 0, 0, 0, 400, 0]) * u.nm)


def create_long_exposure_dao_1_0000():
    return _create_long_exposure_generic(TN_DAO_1, '0000')


def create_long_exposure_dao_1_0002():
    return _create_long_exposure_generic(TN_DAO_1, '0002', petals=np.array([0, 0, 0, 0, 400, 0]) * u.nm)


def create_long_exposure_scao_1000_0000():
    return _create_long_exposure_generic(TN_SCAO_1000, '0000', petals=np.array([0, 0, 0, 0, 0, 0]) * u.nm)


def create_long_exposure_scao_1000_0002():
    return _create_long_exposure_generic(TN_SCAO_1000, '0002', petals=np.array([0, 0, 0, 0, 400, 0]) * u.nm)


def create_long_exposure_scao_2000_0000():
    return _create_long_exposure_generic(TN_SCAO_2000, '0000', petals=np.array([0, 0, 0, 0, 0, 0]) * u.nm)


def create_long_exposure_scao_2000_0002():
    return _create_long_exposure_generic(TN_SCAO_2000, '0002', petals=np.array([0, 0, 0, 0, 400, 0]) * u.nm)


def create_long_exposure_ref_500_0000():
    return _create_long_exposure_generic(TN_REF_500, '0000', petals=np.array([0, 0, 0, 0, 0, 0]) * u.nm)


def create_long_exposure_ref_500_0002():
    return _create_long_exposure_generic(TN_REF_500, '0002', petals=np.array([0, 0, 0, 0, 400, 0]) * u.nm)


def create_long_exposure_ref_100_0000():
    return _create_long_exposure_generic(TN_REF_100, '0000', petals=np.array([0, 0, 0, 0, 0, 0]) * u.nm)


def create_long_exposure_ref_100_0002():
    return _create_long_exposure_generic(TN_REF_100, '0002', petals=np.array([0, 0, 0, 0, 400, 0]) * u.nm)


def create_long_exposure_ref_10_0000():
    return _create_long_exposure_generic(TN_REF_10, '0000', petals=np.array([0, 0, 0, 0, 0, 0]) * u.nm)


def create_long_exposure_ref_10_0002():
    return _create_long_exposure_generic(TN_REF_10, '0002', petals=np.array([0, 0, 0, 0, 400, 0]) * u.nm)


def analyze_none_0000():
    return _analyze_long_exposure(TN_NONE, '0000')


def analyze_none_0002():
    return _analyze_long_exposure(TN_NONE, '0002')


def analyze_mcao_1_0000():
    return _analyze_long_exposure(TN_MCAO_1, '0000')


def analyze_mcao_1_0001():
    return _analyze_long_exposure(TN_MCAO_1, '0001')


def analyze_dao_1_0000():
    return _analyze_long_exposure(TN_DAO_1, '0000')


def analyze_dao_1_0002():
    return _analyze_long_exposure(TN_DAO_1, '0002')


def analyze_mcao_0002():
    return _analyze_long_exposure(TN_MCAO_1, '0002')


def analyze_scao_1000_0000():
    return _analyze_long_exposure(TN_SCAO_1000, '0000')


def analyze_scao_1000_0001():
    return _analyze_long_exposure(TN_SCAO_1000, '0001')


def analyze_scao_1000_0002():
    return _analyze_long_exposure(TN_SCAO_1000, '0002')


def analyze_scao_2000_0000():
    return _analyze_long_exposure(TN_SCAO_2000, '0000')


def analyze_scao_2000_0001():
    return _analyze_long_exposure(TN_SCAO_2000, '0001')


def analyze_scao_2000_0002():
    return _analyze_long_exposure(TN_SCAO_2000, '0002')


def analyze_ref_500_0000():
    return _analyze_long_exposure(TN_REF_500, '0000')


def analyze_ref_500_0002():
    return _analyze_long_exposure(TN_REF_500, '0002')


def analyze_ref_100_0000():
    return _analyze_long_exposure(TN_REF_100, '0000')


def analyze_ref_100_0002():
    return _analyze_long_exposure(TN_REF_100, '0002')


def analyze_ref_10_0000():
    return _analyze_long_exposure(TN_REF_10, '0000')


def analyze_ref_10_0002():
    return _analyze_long_exposure(TN_REF_10, '0002')


def animate_all(lep):
    lep.animate_input_opd(vminmax=(-1000, 1000))
    lep.animate_input_opd_cumulative_average(vminmax=(-1000, 1000))
    lep.animate_reconstructed_phase(vminmax=(-900, 900))
    lep.animate_reconstructed_phase_cumulative_average(vminmax=(-900, 900))
    lep.animate_corrected_opd(vminmax=(-900, 900))
    lep.animate_corrected_opd_cumulative_average(vminmax=(-900, 900))


def _analyze_long_exposure(tracknum, code):
    '''
    Measure and compensate for petals on MORFEO residuals
    '''
    le = LongExposurePetalometer.load(long_exposure_tracknum(tracknum, code))
    std_input = le.input_opd()[20:].std(axis=(1, 2))
    std_corr_inst = le.corrected_opd()[20:].std(axis=(1, 2))
    std_corr_long = le.corrected_opd_from_reconstructed_phase_ave()[
        20:].std(axis=(1, 2))
    petals, jumps = le.petals_from_reconstructed_phase_map(
        le.reconstructed_phase_ave())
    _plot_stdev_residual(le, title="%s %s" % (tracknum, code))
    print('\nMean of MORFEO residuals stds: %s' % std_input.mean())
    print(
        'Mean of MORFEO residuals (with long exposure petal correction) stds: %s'
        % std_corr_long.mean())
    print(
        'Mean of MORFEO residuals (with short exposure petal correction) stds: %s'
        % std_corr_inst.mean())
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
    timev = np.arange(len(std_input)) * le._aores.time_step
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


def plot_cedric(le_no_pet, le_pet, title=''):
    std_input_no_pet = _stdev_after_transient(le_no_pet.input_opd())
    std_input_pet = _stdev_after_transient(le_pet.input_opd())
    std_corr_inst_no_pet = _stdev_after_transient(le_no_pet.corrected_opd())
    std_corr_inst = _stdev_after_transient(le_pet.corrected_opd())
    # std_corr_long = _stdev_after_transient(
    #    le_pet.corrected_opd_from_reconstructed_phase_ave())
    timev = np.arange(len(std_input_no_pet)) * le_no_pet._aores.time_step
    sr_input_no_pet = wavefront_rms_2_strehl_ratio(std_input_no_pet, 2200)
    sr_input_pet = wavefront_rms_2_strehl_ratio(std_input_pet, 2200)
    sr_corr_inst = wavefront_rms_2_strehl_ratio(std_corr_inst, 2200)
    # sr_corr_long = wavefront_rms_2_strehl_ratio(std_corr_long, 2200)

    fig, ax1 = plt.subplots()

    ax1.plot(timev, sr_input_no_pet, color='tab:gray',
             label=r'No added petal $SR_{off}^{no pet} = %.3f$' % sr_input_no_pet.mean())
    ax1.plot(timev, sr_input_pet, color='tab:red',
             label=r'Uncorrected $SR_{off}^{pet} = %.3f$' % sr_input_pet.mean())
    ax1.plot(timev, sr_corr_inst, color='tab:green',
             label=r'Corrected short exposure $SR_{short}^{pet} = %.3f$' % sr_corr_inst.mean())
    # ax1.plot(timev, sr_corr_long, color='tab:blue',
    #         label='Corrected long exposure %.3g' % sr_corr_long.mean())
    ax1.tick_params(axis='y')
    ax1.set_xlabel('Time [s]')
    ax1.set_ylabel('SR(2200nm)')
    ax1.set_ylim(0, 1)
    ax1.grid(True)
    ax1.legend(loc='lower right')

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    quadr_diff_inst = quadraticSum([std_input_pet, -std_corr_inst])
    # quadr_diff_long = quadraticSum([std_input_pet, -std_corr_long])
    quadr_diff_nopet_pet = quadraticSum([-std_corr_inst_no_pet, std_corr_inst])
    quadr_diff_input_nopet_pet = quadraticSum(
        [-std_input_no_pet, std_input_pet])
    ax2.plot(timev, quadr_diff_inst, linestyle='dotted', color='tab:green',
             label=r'$\sqrt{{\sigma_{off}^{pet}}^2-{\sigma_{short}^{pet}}^2} = %d nm$' % quadr_diff_inst.mean())
    # ax2.plot(timev, quadr_diff_long, linestyle='dotted', color='tab:blue',
    #         label=r'$\sqrt{\sigma_{off}^2-\sigma_{long}^2} = %d nm$' % quadr_diff_long.mean())
    ax2.plot(timev, quadr_diff_nopet_pet, linestyle='dotted', color='tab:olive',
             label=r'$\sqrt{{\sigma_{short}^{pet}}^2-{\sigma_{short}^{no pet}}^2} = %d nm$' % quadr_diff_nopet_pet.mean())
    ax2.tick_params(axis='y')
    ax2.set_ylabel('correction stdev [nm]')
    ax2.set_ylim(bottom=-500, top=500)
    ax2.legend(loc='lower left')
    ax2.text(0.05, 0.95,
             'off: correction off\nshort: loop frame 2ms\npet: 400nm petal added\nno pet: no petal added',
             transform=ax2.transAxes, fontsize=12,
             verticalalignment='top')

    ax1.set_title(title)
    fig.tight_layout()  # otherwise the right y-label is slightly clipped

    print('stdev non-corrected opd: added petal %.1f nm / no added petal %.1f nm / diff %.1f' %
          (std_input_pet.mean(), std_input_no_pet.mean(), quadr_diff_input_nopet_pet.mean()))

    print('stdev corrected opd: added petal %.1f nm / no added petal %.1f nm / diff %.1f' %
          (std_corr_inst.mean(), std_corr_inst_no_pet.mean(), quadr_diff_nopet_pet.mean()))


def _analyze_two_leps(passata_tracknum, lep_code_1, lep_code_2):
    le1 = _analyze_long_exposure(passata_tracknum, lep_code_1)
    le2 = _analyze_long_exposure(passata_tracknum, lep_code_2)
    plot_cedric(le1, le2, title=le1._passata_tracking_number)
    return le1, le2


def plot_none():
    return _analyze_two_leps(TN_NONE, '0000', '0002')


def plot_mcao_1():
    return _analyze_two_leps(TN_MCAO_1, '0000', '0002')


def plot_dao_1():
    return _analyze_two_leps(TN_DAO_1, '0000', '0002')


def plot_ref_500():
    return _analyze_two_leps(TN_REF_500, '0000', '0002')


def plot_ref_100():
    return _analyze_two_leps(TN_REF_100, '0000', '0002')


def plot_ref_10():
    return _analyze_two_leps(TN_REF_10, '0000', '0002')


def plot_scao_1000():
    return _analyze_two_leps(TN_SCAO_1000, '0000', '0002')


def plot_scao_2000():
    return _analyze_two_leps(TN_SCAO_2000, '0000', '0002')



def update_header(tn, code):
    from astropy.io import fits
    # tn = TN_MCAO_1
    # code = '0002'
    fn = long_exposure_filename(tn, code)
    tc = long_exposure_tracknum(tn, code)
    fits.setval(fn, 'HIERARCH LPE.LPE_TRACKNUM', value=tc)
    fits.setval(fn, 'HIERARCH LPE.TRACKNUM', value=tn)
