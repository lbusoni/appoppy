import numpy as np
from appoppy.long_exposure import LongExposurePetalometer
import os

ROOT_DIR = '/Users/giuliacarla/Documents/INAF/Lavoro/Progetti/MORFEO/Petalometer/CiaoCiaoWFS/analysis/data/from_appoppy/20230517_160708.0_coo0.0_0.0'


def main230519_petalometer_on_MORFEO_residuals_with_LWE(rot_angle=60):
    le = LongExposurePetalometer.load(
        os.path.join(
            ROOT_DIR, 'long_exp_%sdeg.fits' % rot_angle))
    phase_screen_ave = le.phase_screen_ave()
    phase_screens = le.phase_screen()
    phase_correction = np.tile(
        le.phase_correction_from_petalometer(), (900, 1, 1))
    phase_screens_res = phase_screens - phase_correction
    phase_screen_res_ave = phase_screens_res.mean(axis=0)
    print('\nStd of long exposure MORFEO residuals: %s' % phase_screen_ave.std())
    print(
        'Std of long exposure MORFEO residuals with petalometer correction: %s' 
        % phase_screen_res_ave.std())
    return le
