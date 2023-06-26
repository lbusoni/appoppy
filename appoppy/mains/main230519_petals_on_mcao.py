import numpy as np
import astropy.units as u
import matplotlib.pyplot as plt
from appoppy.long_exposure import LongExposurePetalometer
import os
from appoppy.petalometer import Petalometer
from appoppy.ao_residuals import AOResidual
from appoppy.gif_animator import Gif2DMapsAnimator
from astropy.io import fits

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
    phase_screens = le.phase_screen()[100:]
    phase_correction = np.tile(
        le.phase_correction_from_petalometer(), (phase_screens.shape[0], 1, 1))
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
    phase_screens = le.phase_screen()[100:]
    phase_correction = np.tile(
        le.phase_correction_from_petalometer(), (phase_screens.shape[0], 1, 1))
    phase_screens_res = phase_screens - phase_correction
    std_res = phase_screens.std(axis=(1, 2))
    std_res_pet = phase_screens_res.std(axis=(1, 2))
    _, jumps = le.petals_from_phase_difference_ave()
    petals = -1 * np.cumsum(jumps[::2])
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
    phase_screens = le.phase_screen()[100:]
    phase_correction = np.tile(
        le.phase_correction_from_petalometer(),
        (phase_screens.shape[0], 1, 1))
    phase_screens_res = phase_screens - phase_correction
    std_res = phase_screens.std(axis=(1, 2))
    std_res_pet = phase_screens_res.std(axis=(1, 2))
    _, jumps = le.petals_from_phase_difference_ave()
    petals = -1 * np.cumsum(jumps[::2])
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


def main230623_P50_MORFEO_residuals_characterization():
    aores = AOResidual('20210511_144618.0_coo0.0_0.0')
    ps = aores.phase_screen
    ps_cumave = aores.phase_screen_cumave
    print('RMS [nm]: %s' % np.mean(ps.std(axis=(1, 2))))
    dir_path = '/Users/giuliacarla/Documents/INAF/Lavoro/Progetti/MORFEO/Petalometer/CiaoCiaoWFS/analysis/animations/20210511_144618.0_coo0.0_0.0_24um_60deg'
    gf = Gif2DMapsAnimator(
        os.path.join(dir_path, 'phase_screen'), ps, deltat=aores.time_step,
        pixelsize=aores._pxscale, vminmax=(-1100, 1100))
    gf.make_gif()
    gf = Gif2DMapsAnimator(
        os.path.join(dir_path, 'phase_screen_cum'), ps_cumave,
        deltat=aores.time_step, vminmax=(-300, 300))
    gf.make_gif()


def main230623_MORFEO_residuals_with_LWE_characterization():
    aores = AOResidual('20230517_160708.0_coo0.0_0.0')
    ps = aores.phase_screen
    ps_cumave = aores.phase_screen_cumave
    print('RMS [nm]: %s' % np.mean(ps.std(axis=(1, 2))))
    dir_path = '/Users/giuliacarla/Documents/INAF/Lavoro/Progetti/MORFEO/Petalometer/CiaoCiaoWFS/analysis/animations/20230517_160708.0_coo0.0_0.0_24um_60deg'
    gf = Gif2DMapsAnimator(
        os.path.join(dir_path, 'phase_screen_2'), ps, deltat=aores.time_step,
        pixelsize=aores._pxscale,
        vminmax=(-4500, 4500))
    gf.make_gif()
    gf = Gif2DMapsAnimator(
        os.path.join(dir_path, 'phase_screen_cum_2'), ps_cumave,
        deltat=aores.time_step, vminmax=(-3500, 3500))
    gf.make_gif()
    plt.imshow(ps_cumave[-1], origin='lower', vmin=-3500, vmax=3500,
               cmap='twilight')
    

def main230625_100_petals_realization_on_MORFEO_residuals_P50(
        rho='0.0', theta='0.0',
        rot_angle=60):
    '''
    Compare petals measurement obtained with random realization of petals
    between 100 nm and 100 nm added to MORFEO residuals.
    '''
    tracking_number = TN_P50 + '%s_%s' % (rho, theta)
    wv = 24e-6 * u.m
    
    petals_input = []
    petals_output = []
    
    for i in range(100):
        print(i)
        petals = np.random.randint(100, 1000, 6)
        pet = Petalometer(
            tracking_number=tracking_number,
            petals=petals * u.nm,
            rotation_angle=rot_angle,
            wavelength=wv,
            should_display=False)
        est_petals = pet.estimated_petals
        petals_input.append(petals)
        petals_output.append(est_petals)

    return petals_input, petals_output  


def main230625_100_petals_realization_on_MORFEO_residuals_P50_analysis():
    tracking_number = TN_P50 + '0.0_0.0'
    aores = AOResidual(tracking_number)
    ps = aores.phase_screen
    std_ao_100frame = ps[100].std()
    dirpath = '/Users/giuliacarla/Documents/INAF/Lavoro/Progetti/MORFEO/Petalometer/CiaoCiaoWFS/analysis/data/from_appoppy/20230517_160708.0_coo0.0_0.0/100_realizations_of_petals'
    pet_in = fits.getdata(os.path.join(dirpath, 'input_petals.fits'))
    pet_out = fits.getdata(os.path.join(dirpath, 'output_petals.fits'))
    errors = np.array([
        abs(pet_out[i] - pet_in[i]) / pet_in[i] for i in range(
            pet_in.shape[0])])
    pet_in_flat = pet_in.flatten()
    idcs = np.argsort(pet_in_flat)
    plt.figure()
    plt.plot(pet_in_flat[idcs], errors.flatten()[idcs])
    plt.axvline(std_ao_100frame, linestyle=':', label='AO residual RMS')
    plt.ylabel('Relative error [%]')
    plt.xlabel('Input piston [nm]')
    plt.grid()
    plt.legend()
    return pet_in, pet_out, errors


def main230625_100_petals_realization_on_MORFEO_residuals_P50_long_exp(
        rho='0.0', theta='0.0'):
    '''
    Compare petals measurement obtained with random realization of petals
    between 100 nm and 100 nm added to MORFEO residuals.
    '''
    tracking_number = TN_P50 + '%s_%s' % (rho, theta)
    wv = 24e-6 * u.m
    
    petals_input = []
    petals_output = []
    jumps_output = []
    
    for i in range(100):
        print(i)
        petals = np.random.randint(100, 1000, 6)
        le = LongExposurePetalometer(
            tracking_number=tracking_number, rot_angle=60,
            start_from_step=100,
            petals=petals * u.nm,
            wavelength=wv)
        le.run()
        est_petals, est_jumps = le.petals_from_phase_difference_ave()
        petals_input.append(petals)
        petals_output.append(est_petals)
        jumps_output.append(est_jumps)

    return petals_input, petals_output, jumps_output
    
