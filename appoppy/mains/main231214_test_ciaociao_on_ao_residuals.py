import numpy as np
import os
import matplotlib.pyplot as plt
from appoppy.long_exposure_simulation import LongExposureSimulation

ROOT_DIR = '/Users/giuliacarla/Documents/INAF/Lavoro/Progetti/MORFEO/Analysis/Petalometer/CiaoCiaoWFS/analysis/data/from_appoppy/'
MORFEO_55 = '20231209_202232.0_coo55.0_0.0'


def main231214_ciaociao_on_MORFEO_residuals_with_one_100nm_piston_in_H():
    '''
    Measure and compensate for petals on MORFEO residuals that include one
    injected piston of 100 nm. Sensing wavelength is 1.65 um.
    '''
    le100 = LongExposureSimulation.load(
        os.path.join(
            ROOT_DIR, MORFEO_55,
            'long_exp_1.65um_60deg_pet_0_0_0_0_100_0.fits'))
    le = LongExposureSimulation.load(
        os.path.join(
            ROOT_DIR, MORFEO_55,
            'long_exp_1.65um_60deg.fits'))
    petals_res, jumps_res = le.petals_from_phase_difference_ave()
    petals_100, jumps_100 = le100.petals_from_phase_difference_ave()
    le.plot_petals()
    plt.title('w/o piston')
    plt.ylim(-300, 350)
    le100.plot_petals()
    plt.title('100 nm piston')
    plt.ylim(-300, 350)
    print('Pistons measured on MORFEO residuals:\n %s' % petals_res)
    print('Pistons measured on MORFEO residuals + 100 nm piston:\n %s'
          % petals_100)
    print('Difference between pistons:\n %s' % (
        np.array(petals_100) - np.array(petals_res)))
    
    le_phase_screen = le.input_opd()[1:]
    le100_phase_screen = le100.input_opd()[1:]
    le_stds = le_phase_screen.std(axis=(1, 2))
    le100_stds = le100_phase_screen.std(axis=(1, 2))
    plt.figure()
    plt.plot(le_stds, label='w/o piston, mean std=%s' % le_stds.mean())
    plt.plot(le100_stds, label='100 nm piston, mean std=%s' % le100_stds.mean())
    plt.ylabel('Std [nm]')
    correction = np.tile(
        le100.opd_correction_from_reconstructed_phase_ave(),
        (le_phase_screen.shape[0], 1, 1))
    corrected_screens = le_phase_screen - correction
    corr_screen_stds = corrected_screens.std(axis=(1, 2))
    plt.plot(corr_screen_stds, label='Corrected, mean std=%s'
             % corr_screen_stds.mean())
    plt.grid()
    plt.legend()
    
    return le, le100
