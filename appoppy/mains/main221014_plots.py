from appoppy.petalometer import Petalometer
import numpy as np
from astropy import units as u
import matplotlib.pyplot as plt
from appoppy.elt_for_petalometry import EltForPetalometry
from appoppy.long_exposure import LongExposurePetalometer


def main_plot_pupil():
    pet = Petalometer(use_simulated_residual_wfe=False,
                      r0=99999,
                      petals=np.zeros(6) * u.nm,
                      rotation_angle=20)
    pet._i4.display_pupil_intensity()
    pet._model1.display_pupil_intensity()
    pet._model2.display_pupil_intensity()

    pet = Petalometer(use_simulated_residual_wfe=False,
                      r0=99999,
                      petals=np.zeros(6) * u.nm,
                      rotation_angle=20,
                      zernike=np.array([0, 0, 10]) * u.um)
    plt.clf()
    plt.imshow(pet._model2.pupil_phase() * pet._model2.pupil_intensity(),
               origin='lower',
               cmap='twilight')
    plt.clf()
    plt.imshow(pet._model1.pupil_phase() * pet._model2.pupil_intensity(),
               origin='lower',
               cmap='twilight')
    pet._i4.display_pupil_intensity()


def opd_turbolenza_kolmo():
    m2 = EltForPetalometry(
        use_simulated_residual_wfe=False,
        r0=0.26, kolm_seed=np.random.randint(2147483647))
    kopd = m2.optical_system.planes[0].get_opd(
        m2.optical_system.input_wavefront(2.2e-6 * u.m))
    kopdm = np.ma.MaskedArray(kopd, mask=m2.pupil_mask())
    print('std %g' % kopdm.std())


def opd_turbolenza_residui_MCAO(start_from=0):
    m2 = EltForPetalometry(
        use_simulated_residual_wfe=True,
        tracking_number='20210518_223459.0',
        residual_wavefront_start_from=start_from)
    kopd = m2.optical_system.planes[0].get_opd(
        m2.optical_system.input_wavefront(2.2e-6 * u.m))
    kopdm = np.ma.MaskedArray(kopd, mask=m2.pupil_mask())
    print('std %g' % kopdm.std())
    return m2
    # perchè sembra diversa dal cubo di Guido? no è ok da 1.8um a 330nm


def phase_shift():
    pet = Petalometer(use_simulated_residual_wfe=False,
                      r0=99999,
                      petals=np.array([0, 200, -400, 600, -800, 1000]) * u.nm,
                      rotation_angle=20,
                      zernike=np.array([0, 10000, -5000, 3000, 400, 500, 600, -200, 100]) * u.nm)
    pet._i4._wf_0.display()
    pet._i4._wf_1.display()
    pet._i4._wf_2.display()
    pet._i4._wf_3.display()


def no_turbolence():
    pet = Petalometer(use_simulated_residual_wfe=False,
                      r0=99999,
                      petals=np.array([0, 100, -200, 300, -400, 500]) * u.nm,
                      rotation_angle=20)

    pet._model2.display_pupil_opd()
    np.round(pet.all_jumps)
    np.round(pet.estimated_petals - pet.estimated_petals[0])
    np.round(pet.error_jumps)
    np.round(pet.difference_between_estimated_petals_and_m4_petals)


# os.path.join('/Users', 'lbusoni', 'Downloads', 'anim')


def petal_estimate_55():
    soi = LongExposurePetalometer.load(
        '/Users/lbusoni/Downloads/anim/soi55.fits')
    eopd, epet, ejump = soi.petals_from_phase_difference_ave()

    mpet = soi._aores.petals_average * u.nm

    return soi, eopd, epet, ejump, mpet
