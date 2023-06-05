import numpy as np
import astropy.units as u
import matplotlib.pyplot as plt
from appoppy.long_exposure import LongExposurePetalometer
import os
from appoppy.petalometer import Petalometer
from appoppy.ao_residuals import AOResidual
from appoppy.low_wind_effect import LowWindEffectWavefront
from appoppy.mask import mask_phase_screen

ROOT_DIR = '/Users/giuliacarla/Documents/INAF/Lavoro/Progetti/MORFEO/Petalometer/CiaoCiaoWFS/analysis/data/from_appoppy/'

TN_LWE = '20230517_160708.0_coo0.0_0.0'
TN_P10 = '20210512_081313.0_coo'
TN_P50 = '20210511_144618.0_coo'
    

def main230519_petalometer_on_MORFEO_residuals_with_LWE(rot_angle=60,
                                                        wv_in_um=24):
    '''
    Measure and compensate for petals on MORFEO residuals that include LWE and
    its correction.
    '''
    le = LongExposurePetalometer.load(
        os.path.join(
            ROOT_DIR, TN_LWE,
            'long_exp_%sum_%sdeg.fits' % (wv_in_um, rot_angle)))
    phase_screens = le.phase_screen()
    phase_correction = np.tile(
        le.phase_correction_from_petalometer(), (900, 1, 1))
    phase_screens_res = phase_screens - phase_correction
    std_res = phase_screens.std(axis=(1, 2))
    std_res_lwe = phase_screens_res.std(axis=(1, 2))
    petals, jumps = le.petals_from_phase_difference_ave()
    plt.figure()
    plt.plot(std_res, label='Petalometer OFF')
    plt.plot(std_res_lwe, label='Petalometer ON')
    plt.legend()
    plt.grid()
    plt.ylabel('Std [nm]')
    plt.xlabel('Step number')
    plt.title('Rotation angle: %s deg' % rot_angle)
    print('\n\nMean of MORFEO residuals stds: %s' % std_res.mean())
    print(
        '\nMean of MORFEO residuals (with petalometer correction) stds: %s' 
        % std_res_lwe.mean())
    print('\nMeasured petals: %s' % petals)
    print('\nMeasured jumps: %s' % jumps[::2])
    return le, phase_correction, std_res, std_res_lwe


def main230525_estimate_noise_due_to_MORFEO_residuals_P50(
        rho='0.0', theta='0.0', rot_angle=60, wv_in_um=24):
    '''
    Estimate the contribution of MORFEO residuals to petals signal.
    '''
    le = LongExposurePetalometer.load(os.path.join(
        ROOT_DIR, TN_P50 + rho + '_' + theta,
        'long_exp_%sum_%sdeg.fits' % (wv_in_um, rot_angle)))
    phase_screens = le.phase_screen()
    std_ao_ps = phase_screens.std(axis=(1, 2))
    plt.figure()
    plt.plot(std_ao_ps)
    plt.grid()
    plt.ylabel('Std [nm]')
    plt.xlabel('Step number')
    print('\n\nMean of MORFEO residuals stds: %s' % std_ao_ps.mean())
    pp, _ = le.petals_from_phase_difference_ave()
    print('\nPetals from AO residuals: %s' % pp)
    return le

    
def main230525_petalometer_on_MORFEO_residuals_P50_one_frame(
        petals_small=[200, 30, -100, 370, 500, 10] * u.nm,
        petals_large=[1200, -1000, 3000, -370, 1500, 20] * u.nm,
        rho='0.0', theta='0.0',
        rot_angle=60):
    '''
    Compare petals measurement obtained with small or large petals added to
    MORFEO residuals.
    '''
    tracking_number = TN_P50 + '%s_%s' % (rho, theta)
    wv = 24e-6 * u.m
    plt.figure()
    pet1 = Petalometer(
        tracking_number=tracking_number,
        petals=petals_small,
        rotation_angle=rot_angle,
        wavelength=wv)
    plt.figure()
    pet2 = Petalometer(
        tracking_number=tracking_number,
        petals=petals_large,
        rotation_angle=rot_angle,
        wavelength=wv)
    print(
        '\n\nError with small petals: %s'
        % pet1.error_petals)
    print('\nError with large petals: %s'
          % pet2.error_petals)
    print(
        '\n\nRelative error with small petals: %s'
        % (pet1.error_petals / pet1.petals))
    print('\nRelative error with large petals: %s'
          % (pet2.error_petals / pet2.petals))
    return pet1, pet2        

    
def main230529_plot_structure_function_of_AO_LWE_maps():
    '''
    Plot (full pupil) structure function of MORFEO residuals that include LWE.
    '''
    aores = AOResidual(TN_LWE)
    stf_0s, stf_2s, seps = aores.plot_structure_function(899)
    return aores, stf_0s, stf_2s, seps


def main230531_plot_structure_function_of_AO_LWE_maps(step_map=500,
                                                      angs=(30, 90)):
    '''
    Plot (one sector) structure function of MORFEO residuals that include LWE.
    '''
    aores = AOResidual(TN_LWE)
    masks = aores.mask_phase_screen(angs)
    seps_px = np.array([1, 2, 5, 10, 20, 50, 100])
    stf = []
    for s in seps_px:
        scr2 = np.roll(masks, s, axis=2)
        dd = (masks[step_map] - scr2[step_map])[:, s:]
        stf.append(np.mean(dd ** 2))
    plt.plot(seps_px, stf)
    plt.xlabel(r'separation $\rho$  [m]')
    plt.ylabel(r'$D_{\phi}$ of residual phase $[nm^2]$')
    plt.grid()


def main230531_petalometer_correction_on_P50_MORFEO_residuals():
    '''
    '''
    le = LongExposurePetalometer.load(
        os.path.join(
            ROOT_DIR, TN_P50 + '0.0_0.0',
            'long_exp_24um_60deg_pet_200_30_m100_370_500_10.fits'))
    phase_screens = le.phase_screen()
    phase_correction = np.tile(
        le.phase_correction_from_petalometer(), (900, 1, 1))
    phase_screens_res = phase_screens - phase_correction
    std_res = phase_screens.std(axis=(1, 2))
    std_res_pet = phase_screens_res.std(axis=(1, 2))
    petals, jumps = le.petals_from_phase_difference_ave()
    plt.figure()
    plt.plot(std_res, label='Petalometer OFF')
    plt.plot(std_res_pet, label='Petalometer ON')
    plt.legend()
    plt.grid()
    plt.ylabel('Std [nm]')
    plt.xlabel('Step number')
    print('\n\nMean of MORFEO residuals stds: %s' % std_res.mean())
    print(
        '\nMean of MORFEO residuals (with petalometer correction) stds: %s' 
        % std_res_pet.mean())
    print('\nMeasured petals: %s' % petals)
    print('\nMeasured jumps: %s' % jumps[::2])
    return le, phase_correction, std_res, std_res_pet


def main230531_petalometer_correction_on_P50_MORFEO_residuals_large_petals():
    '''
    '''
    le = LongExposurePetalometer.load(
        os.path.join(
            ROOT_DIR, TN_P50 + '0.0_0.0',
            'long_exp_24um_60deg_pet_1200_m1000_3000_370_1500_20.fits'))
    phase_screens = le.phase_screen()
    phase_correction = np.tile(
        le.phase_correction_from_petalometer(),
        (phase_screens.shape[0], 1, 1))
    phase_screens_res = phase_screens - phase_correction
    std_res = phase_screens.std(axis=(1, 2))
    std_res_pet = phase_screens_res.std(axis=(1, 2))
    petals, jumps = le.petals_from_phase_difference_ave()
    plt.figure()
    plt.plot(std_res, label='Petalometer OFF')
    plt.plot(std_res_pet, label='Petalometer ON')
    plt.legend()
    plt.grid()
    plt.ylabel('Std [nm]')
    plt.xlabel('Step number')
    print('\n\nMean of MORFEO residuals stds: %s' % std_res.mean())
    print(
        '\nMean of MORFEO residuals (with petalometer correction) stds: %s' 
        % std_res_pet.mean())
    print('\nMeasured petals: %s' % petals)
    print('\nMeasured jumps: %s' % jumps[::2])
    return le, phase_correction, std_res, std_res_pet


def main230605_mean_on_sector(angs=(30, 90)):
    lwe = LowWindEffectWavefront(wind_speed=0.5)
    cube_maps = lwe._cube
    masks = mask_phase_screen(cube_maps, angs)
    return cube_maps, masks
