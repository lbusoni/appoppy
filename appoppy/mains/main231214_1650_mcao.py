
import os
import numpy as np
from astropy import units as u
from appoppy.long_exposure import LongExposurePetalometer
import matplotlib.pyplot as plt


ROOT_DIR = '/Users/lbusoni/Library/CloudStorage/GoogleDrive-lorenzo.busoni@inaf.it/Il mio Drive/adopt/Varie/CiaoCiaoWFS/analysis/data/from_appoppy/'

TN_MCAO_1 = '20231209_202232.0_coo55.0_0.0'
TN_SCAO_1000 = '20231212_212912.0_coo0.0_0.0'
TN_SCAO_2000 = '20231213_101833.0_coo0.0_0.0'
TN_REF_1000 = '20231213_123051.0_coo0.0_0.0'
TN_REF_100 = '20231213_123200.0_coo0.0_0.0'
TN_REF_10 = '20231213_123403.0_coo0.0_0.0'


def _tracknum_code(tn, code):
    return "%s_%s" % (tn, code)


def _filename(tn, code):
    return os.path.join(ROOT_DIR, _tracknum_code(tn, code), 'long_exp.fits')


def _create_long_exposure_generic(tn,
                                  code,
                                  rot_angle=60,
                                  petals=np.array([0, 0, 0, 0, 0, 0]) * u.nm,
                                  wavelength=1650*u.nm,
                                  n_iter=1000):
    le = LongExposurePetalometer(
        tracking_number=_tracknum_code(tn, code), rot_angle=rot_angle,
        petals=petals,
        wavelength=wavelength,
        n_iter=n_iter)
    le.run()
    le.save(_filename(tn, code))
    return le


def create_long_exposure_mcao_0000():
    return _create_long_exposure_generic(TN_MCAO_1, '0000')


def create_long_exposure_mcao_0001():
    return _create_long_exposure_generic(TN_MCAO_1, '0001', petals=np.array([100, 0, 0, 0, 0, 0]) * u.nm)


def create_long_exposure_mcao_0002():
    return _create_long_exposure_generic(TN_MCAO_1, '0002', petals=np.array([400, 0, 0, 0, 0, 0]) * u.nm)


def create_long_exposure_scao_1000_0000():
    return _create_long_exposure_generic(TN_SCAO_1000, '0000', petals=np.array([0, 0, 0, 0, 0, 0]) * u.nm)


def create_long_exposure_scao_1000_0001():
    return _create_long_exposure_generic(TN_SCAO_1000, '0001', petals=np.array([100, 0, 0, 0, 0, 0]) * u.nm)


def create_long_exposure_scao_1000_0002():
    return _create_long_exposure_generic(TN_SCAO_1000, '0002', petals=np.array([400, 0, 0, 0, 0, 0]) * u.nm)


def create_long_exposure_scao_2000_0000():
    return _create_long_exposure_generic(TN_SCAO_2000, '0000', petals=np.array([0, 0, 0, 0, 0, 0]) * u.nm)


def create_long_exposure_scao_2000_0001():
    return _create_long_exposure_generic(TN_SCAO_2000, '0001', petals=np.array([100, 0, 0, 0, 0, 0]) * u.nm)


def create_long_exposure_scao_2000_0002():
    return _create_long_exposure_generic(TN_SCAO_2000, '0002', petals=np.array([400, 0, 0, 0, 0, 0]) * u.nm)


def analyze_mcao_0000():
    return _analyze_long_exposure(TN_MCAO_1, '0000')


def analyze_mcao_0001():
    return _analyze_long_exposure(TN_MCAO_1, '0001')


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


def animate_all(lep):
    lep.animate_phase_screens_cumulative_average(vminmax=(-1000, 1000))
    lep.animate_phase_screens(vminmax=(-1000, 1000))
    lep.animate_phase_difference(vminmax=(-900, 900))
    lep.animate_phase_difference_cumulative_average(vminmax=(-900, 900))

def _analyze_long_exposure(tracknum, code):
    '''
    Measure and compensate for petals on MORFEO residuals
    '''
    le = LongExposurePetalometer.load(_filename(tracknum, code))
    phase_screens_petal_corrected = le.phase_screen_petal_corrected()
    std_input = le.phase_screen()[20:].std(axis=(1, 2))
    std_corr = phase_screens_petal_corrected[20:].std(axis=(1, 2))
    petals, jumps = le.petals_from_phase_difference_ave()
    _plot_stdev_residual(std_input, std_corr,
                         title="%s %s" % (tracknum, code))
    print('\n\nMean of MORFEO residuals stds: %s' % std_input.mean())
    print(
        '\nMean of MORFEO residuals (with petalometer correction) stds: %s'
        % std_corr.mean())
    print('\nMeasured petals: %s' % petals)
    print('\nMeasured jumps: %s' % jumps[::2])
    return le, std_input, std_corr


def _plot_stdev_residual(std_input, std_res_pet_corr, title=''):
    plt.figure()
    plt.plot(std_input, label='Petalometer OFF')
    plt.plot(std_res_pet_corr, label='Petalometer ON')
    plt.legend()
    plt.grid()
    plt.ylabel('Std [nm]')
    plt.xlabel('Step number')
    plt.title(title)


def update_header(tn, code):
    from astropy.io import fits
    # tn = TN_MCAO_1
    # code = '0002'
    fn = _filename(tn, code)
    tc = _tracknum_code(tn, code)
    fits.setval(fn, 'HIERARCH LPE.LPE_TRACKNUM', value=tc)
    fits.setval(fn, 'HIERARCH LPE.TRACKNUM', value=tn)
