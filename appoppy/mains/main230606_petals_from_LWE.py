import numpy as np
import matplotlib.pyplot as plt
import astropy.units as u
from appoppy.petalometer import Petalometer
from appoppy.low_wind_effect import LowWindEffectWavefront
from appoppy.mask import mask_phase_screen


def main230605_correction_of_LWE_induced_petals_between_sectors():
    rot_angle = 60
    wv = 24e-6 * u.m
    pet = Petalometer(lwe_speed=0.5, wavelength=wv,
                      rotation_angle=rot_angle,
                      should_display=False)
    phase_screen = pet.pupil_opd
    jj = pet.across_islands_jumps
    pp = pet.estimated_petals
    pp_corr = pp - pp.mean()
    pet_corr = Petalometer(
                petals=pp_corr,
                rotation_angle=rot_angle,
                npix=phase_screen.shape[0],
                wavelength=wv,
                should_display=False)
    phase_correction = pet_corr.pupil_opd
    phase_screen_res = phase_screen - phase_correction
    plt.figure()
    plt.imshow(phase_screen, origin='lower', vmin=0, vmax=5600)
    plt.colorbar()
    plt.figure()
    plt.imshow(phase_screen_res, origin='lower', vmin=0, vmax=5600)
    plt.colorbar()
    std_lwe = phase_screen.std()
    std_res = phase_screen_res.std()
    print('\n\nMeasured petals: %s' % (pp.to(u.um)))
    print('\nMeasured jumps: %s' % (jj.to(u.um)))
    print('\nLWE maps stds: %s' % std_lwe)
    print(
        '\nLWE maps (with petalometer correction) stds: %s' 
        % std_res)
    return pet


def main230606_LWE_maps_temporal_mean_and_std_on_sectors(wind_speed=0.5):
    '''
    Sector 0 is in (90, 30)deg, sector 1 in (30, -30) and so on.
    '''
    sectors = 6
    lwe = LowWindEffectWavefront(wind_speed=wind_speed,
                                 start_from=None,
                                 step=0,
                                 average_on=1)
    lwe_maps = lwe.phase_screens()
    n_step = lwe_maps.shape[0]
    times = np.linspace(lwe.time_step, lwe.time_step * n_step, n_step)
    angs = np.linspace(90, -270, 7)
    means = []
    stds = []
    for i in range(sectors):
        masks = [mask_phase_screen(m, (angs[i + 1], angs[i])) for
                 m in lwe_maps]
        means.append(np.ma.mean(masks, axis=(1, 2)))
        stds.append(np.ma.std(masks, axis=(1, 2)))
    plt.figure()
    for i in range(sectors):
        plt.plot(times, means[i], label='S%s' % (i + 1))
    plt.xlabel('Time [s]')
    plt.ylabel('Average OPD [um]')
    plt.legend()
    plt.grid()
    plt.figure()
    for i in range(sectors - 1):
        plt.plot(means[i] - means[i+1], label='S%s - S%s' %(i, i+1))
    plt.legend()
    plt.grid()
    plt.xlabel('Time [s]')
    plt.ylabel('LWE-induced petals [um]')
    return masks, means, stds


def main230607_LWE_induced_petals_across_spiders(rot_angle=5):
    wv = 24e-6 * u.m
    pet = Petalometer(lwe_speed=0.5, wavelength=wv,
                      rotation_angle=rot_angle,
                      should_display=True)
    pp = pet.estimated_petals
    jj = pet.across_islands_jumps
    print('\n\nMeasured petals: %s' % (pp.to(u.um)))
    print('\n\nMeasured jumps: %s' % (jj.to(u.um)))
    return pet
