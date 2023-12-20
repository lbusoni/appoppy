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
ROOT_DIR = '/Users/lbusoni/Library/CloudStorage/GoogleDrive-lorenzo.busoni@inaf.it/Drive condivisi/MORFEO-OAA/Petalometro Ciao Ciao/analysis/data/from_appoppy'

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
    phase_screens = le.input_opd()[100:]
    phase_correction = np.tile(
        le.opd_correction_from_reconstructed_phase_ave(), (phase_screens.shape[0], 1, 1))
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
    phase_screens = le.input_opd()
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
    phase_screens = le.input_opd()[100:]
    phase_correction = np.tile(
        le.opd_correction_from_reconstructed_phase_ave(), (phase_screens.shape[0], 1, 1))
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
    phase_screens = le.input_opd()[100:]
    phase_correction = np.tile(
        le.opd_correction_from_reconstructed_phase_ave(),
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
        os.path.join(dir_path, 'phase_screen'), ps, deltat=aores.time_step,
        pixelsize=aores._pxscale,
        vminmax=(-4500, 4500))
    gf.make_gif()
    gf = Gif2DMapsAnimator(
        os.path.join(dir_path, 'phase_screen_cum'), ps_cumave,
        deltat=aores.time_step,
        pixelsize=aores._pxscale, vminmax=(-3500, 3500))
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
        abs(pet_out[i] - pet_in[i]) for i in range(
            pet_in.shape[0])])
    pet_in_flat = pet_in.flatten()
    idcs = np.argsort(pet_in_flat)
    plt.figure()
    plt.plot(pet_in_flat[idcs], errors.flatten()[idcs])
    plt.axvline(std_ao_100frame, linestyle=':', label='AO residual RMS')
    plt.ylabel('Error [nm]')
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


def main230627_petalometer_on_MORFEO_residuals_with_petals_in_Ks():
    '''
    Measure and compensate for petals on MORFEO residuals that include injected
    pistons. Sensing wavelength is 2.2 um.
    '''
    le_pet = LongExposurePetalometer.load(
        os.path.join(
            ROOT_DIR, TN_P50 + '0.0_0.0',
            'long_exp_2.2um_60deg_pet_200_30_m100_370_500_0.fits'))
    true_petals = [200, 30, -100, 370, 500, 0] * u.nm
    le = LongExposurePetalometer.load(
        os.path.join(
            ROOT_DIR, TN_P50 + '0.0_0.0',
            'long_exp_2.2um_60deg.fits'))
    _, jumps_res = le.petals_from_phase_difference_ave()
    _, jumps_pet = le_pet.petals_from_phase_difference_ave()
    petals_res = -1 * np.cumsum(jumps_res[::2])
    petals_pet = -1 * np.cumsum(jumps_pet[::2])
    meas_pet = np.array(petals_pet) - np.array(petals_res) 
    print(meas_pet)
    print(true_petals)
    print('Error: %s' % (true_petals.value - meas_pet))
    print('Error std: %s' % (true_petals.value - meas_pet).std())
    return true_petals, meas_pet, jumps_res, jumps_pet, petals_res, petals_pet


def main230627_correction_on_MORFEO_residuals_with_petals_in_Ks():
    '''
    '''
    le = LongExposurePetalometer.load(
        os.path.join(
            ROOT_DIR, TN_P50 + '0.0_0.0',
            'long_exp_2.2um_60deg_pet_200_30_m100_370_500_0.fits'))
    phase_screens = le.input_opd()[100:]
    phase_correction = np.tile(
        le.opd_correction_from_reconstructed_phase_ave(), (phase_screens.shape[0], 1, 1))
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
    print('\n\nMean of MORFEO residuals stds: %s' % std_res.mean())
    print(
        '\nMean of MORFEO residuals (with petalometer correction) stds: %s' 
        % std_res_lwe.mean())
    print('\nMeasured petals: %s' % petals)
    print('\nMeasured jumps: %s' % jumps[::2])
    return le, phase_correction, std_res, std_res_lwe


def main230627_petalometer_on_MORFEO_residuals_with_small_petals_in_Ks():
    '''
    Measure and compensate for petals on MORFEO residuals that include injected
    pistons. Sensing wavelength is 2.2 um.
    '''
    le_pet = LongExposurePetalometer.load(
        os.path.join(
            ROOT_DIR, TN_P50 + '0.0_0.0',
            'long_exp_2.2um_60deg_pet_50_30_m15_70_100_0.fits'))
    true_petals = [50, 30, -15, 70, 100, 0] * u.nm
    le = LongExposurePetalometer.load(
        os.path.join(
            ROOT_DIR, TN_P50 + '0.0_0.0',
            'long_exp_2.2um_60deg.fits'))
    _, jumps_res = le.petals_from_phase_difference_ave()
    _, jumps_pet = le_pet.petals_from_phase_difference_ave()
    petals_res = -1 * np.cumsum(jumps_res[::2])
    petals_pet = -1 * np.cumsum(jumps_pet[::2])
    meas_pet = np.array(petals_pet) - np.array(petals_res) 
    print(meas_pet)
    print(true_petals)
    print('Error: %s' % (true_petals.value - meas_pet))
    print('Error std: %s' % (true_petals.value - meas_pet).std())
    return true_petals, meas_pet, jumps_res, jumps_pet, petals_res, petals_pet


def main230627_petalometer_on_MORFEO_residuals_with_LWE_dual_wavelength():
    '''
    Measure and compensate for petals on MORFEO residuals that include LWE and
    its correction.
    '''
    rot_angle = 60
    le_24um = LongExposurePetalometer.load(
        os.path.join(
            ROOT_DIR, TN_LWE,
            'long_exp_%sum_%sdeg.fits' % (24, rot_angle)))
    le_1_5um = LongExposurePetalometer.load(
        os.path.join(
            ROOT_DIR, TN_LWE,
            'long_exp_%sum_%sdeg.fits' % (1.5, rot_angle)))
    le_1_6um = LongExposurePetalometer.load(
        os.path.join(
            ROOT_DIR, TN_LWE,
            'long_exp_%sum_%sdeg.fits' % (1.6, rot_angle)))
    # phase_screens = le.phase_screen()[100:]
    # phase_correction = np.tile(
    #     le.phase_correction_from_petalometer(), (phase_screens.shape[0], 1, 1))
    # phase_screens_res = phase_screens - phase_correction
    # std_res = phase_screens.std(axis=(1, 2))
    # std_res_lwe = phase_screens_res.std(axis=(1, 2))
    petals_24um, jumps_24um = le_24um.petals_from_phase_difference_ave()
    petals_1_5um, jumps_1_5um = le_1_5um.petals_from_phase_difference_ave()
    petals_1_6um, jumps_1_6um = le_1_6um.petals_from_phase_difference_ave()
    
    return le_24um, petals_24um, petals_1_5um, petals_1_6um


def main230627_test_capture_range():
    wv = 2.2e-6 * u.m
    true_petals = [-1500, -1100, -500, 500, 1100, 1500]
    est_petals = []
    for p in true_petals:
        pet = Petalometer(petals=[p, 0, 0, 0, 0, 0] * u.nm,
                          rotation_angle=60,
                          wavelength=wv,
                          should_display=False)
        est_pet = pet.estimated_petals
        est_petals.append(est_pet[0].value)
    plt.figure()
    plt.plot(true_petals, est_petals, '.-')
    return true_petals, est_petals


def main230627_test_one_petals_on_one_MORFEO_frame(rot_angle=60):
    wv = 2.2e-6 * u.m
    true_petals = [500, 0, 0, 0, 0, 0]
    pet_res = Petalometer(tracking_number=TN_P50 + '0.0_0.0',
                      rotation_angle=rot_angle,
                      wavelength=wv,
                      should_display=True)
    plt.figure()
    pet_res_pet = Petalometer(tracking_number=TN_P50 + '0.0_0.0',
                          petals=true_petals * u.nm,
                          rotation_angle=rot_angle,
                          wavelength=wv,
                          should_display=True)
    print(pet_res_pet.estimated_petals - pet_res.estimated_petals)
    return pet_res, pet_res_pet


def main230627_petalometer_on_MORFEO_residuals_with_one_500nm_petals_in_Ks():
    '''
    Measure and compensate for petals on MORFEO residuals that include one
    injected pistons. Sensing wavelength is 2.2 um.
    '''
    le500 = LongExposurePetalometer.load(
        os.path.join(
            ROOT_DIR, TN_P50 + '0.0_0.0',
            'long_exp_2.2um_60deg_pet_500_0_0_0_0_0.fits'))
    le = LongExposurePetalometer.load(
        os.path.join(
            ROOT_DIR, TN_P50 + '0.0_0.0',
            'long_exp_2.2um_60deg.fits'))
    petals_res, jumps_res = le.petals_from_phase_difference_ave()
    petals_500, jumps_500 = le500.petals_from_phase_difference_ave()
    print(np.array(petals_500) - np.array(petals_res))
    return jumps_res, jumps_500, petals_res, petals_500


def main230627_petalometer_on_MORFEO_residuals_with_one_800nm_petals_in_Ks():
    '''
    Measure and compensate for petals on MORFEO residuals that include one
    injected pistons. Sensing wavelength is 2.2 um.
    '''
    le800 = LongExposurePetalometer.load(
        os.path.join(
            ROOT_DIR, TN_P50 + '0.0_0.0',
            'long_exp_2.2um_60deg_pet_800_0_0_0_0_0.fits'))
    le = LongExposurePetalometer.load(
        os.path.join(
            ROOT_DIR, TN_P50 + '0.0_0.0',
            'long_exp_2.2um_60deg.fits'))
    petals_res, jumps_res = le.petals_from_phase_difference_ave()
    petals_800, jumps_800 = le800.petals_from_phase_difference_ave()
    print(np.array(petals_800) - np.array(petals_res))
    return jumps_res, jumps_800, petals_res, petals_800
