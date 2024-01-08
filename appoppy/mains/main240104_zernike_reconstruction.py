import numpy as np
import astropy.units as u
import matplotlib.pyplot as plt
from appoppy.petalometer import Petalometer
from appoppy.elt_for_petalometry import EltForPetalometry
from appoppy.phase_shift_interferometer import PhaseShiftInterferometer
from skimage.restoration.unwrap import unwrap_phase
from arte.utils.modal_decomposer import ModalDecomposer
from arte.types.wavefront import Wavefront
from arte.types.mask import CircularMask, BaseMask


def tip_vs_ciaociao_rotation_angle():
    pet_tip_1lmd_20deg = Petalometer(
        rotation_angle=20, zernike=np.array([0, 2.2]) * u.um,
        should_display=False)
    pet_tip_1lmd_20deg.sense_wavefront_jumps()
    pet_tip_1lmd_40deg = Petalometer(
        rotation_angle=40, zernike=np.array([0, 2.2]) * u.um,
        should_display=False)
    pet_tip_1lmd_40deg.sense_wavefront_jumps()
    pet_tip_1lmd_60deg = Petalometer(
        rotation_angle=60, zernike=np.array([0, 2.2]) * u.um,
        should_display=False)
    pet_tip_1lmd_60deg.sense_wavefront_jumps()
    plt.figure()
    plt.imshow(pet_tip_1lmd_20deg.reconstructed_phase, origin='lower')
    plt.title('20 deg')
    plt.colorbar(label='[nm]')
    plt.figure()
    plt.imshow(pet_tip_1lmd_20deg.pupil_opd, origin='lower')
    plt.colorbar(label='[nm]')
    plt.title('OPD')
    plt.figure()
    plt.imshow(pet_tip_1lmd_40deg.reconstructed_phase, origin='lower')
    plt.title('40 deg')
    plt.colorbar(label='[nm]')
    plt.figure()
    plt.imshow(pet_tip_1lmd_60deg.reconstructed_phase, origin='lower')
    plt.title('60 deg')
    plt.colorbar(label='[nm]')


def dumb_test_of_tip_measurement_from_interferogram_1():
    '''
    Tilt with a2 = 1 wv.
    Find the number of waves from the number of maxima in the interferogram.
    '''
    model1 = EltForPetalometry(r0=np.inf, zern_coeff=[0, 0] * u.um)
    model2 = EltForPetalometry(r0=np.inf, zern_coeff=[0, 2.2] * u.um)
    psi = PhaseShiftInterferometer(model1, model2)
    psi.acquire()
    rec_phase = psi.interferogram()
    one_wv = np.ptp(rec_phase)
    nr_of_waves = np.argwhere(rec_phase == rec_phase.max()).flatten().shape[0]
    rms_to_ptv = 4
    a2 = one_wv * nr_of_waves / rms_to_ptv 
    return rec_phase, psi, a2


def dumb_test_of_tip_measurement_from_interferogram_2():
    '''
    Tilt with a2 = 1 wv.
    Use unwrap_phase() function implemented (but not used) in
    PhaseShiftInterferometer. Fix the code in PhaseShiftInterferometer: the
    unwrapped phase is not included in a masked array.
    Evaluate the OPD from the ptv of the unwrapped phase (converted into nm ptv)
    and convert it into nm rms. 
    '''
    model1 = EltForPetalometry(r0=np.inf, zern_coeff=[0, 0] * u.um)
    model2 = EltForPetalometry(r0=np.inf, zern_coeff=[0, 2.2, 0, 2.2] * u.um)
    psi = PhaseShiftInterferometer(model1, model2)
    psi._should_unwrap = True
    psi.acquire()
    # _ = psi.interferogram()
    opd = psi.interferogram()
    wrapped_phase = psi._wrapped
    plt.figure()
    plt.imshow(wrapped_phase, origin='lower')
    plt.colorbar()
    plt.title('Wrapped phase')
    # unwrapped_phase = np.ma.masked_array(unwrap_phase(wrapped_phase),
    # mask=psi.global_mask())
    plt.figure()
    plt.imshow(opd, origin='lower')
    plt.colorbar()
    plt.title('Unwrapped phase')
    plt.figure()
    plt.plot(opd[int(opd.shape[0] / 2),:])
    plt.title('Cut on x axis of unwrapped phase')
    # opd = (np.ptp(unwrapped_phase) * psi._wf_0.wavelength / (2 * np.pi)
    #        ).to_value(u.nm)
    rms_to_ptv = 4
    a2 = np.ptp(opd) / rms_to_ptv
   
    circular_mask = CircularMask((256, 256), 128, (128, 128))
    md = ModalDecomposer(3)
    zc = md.measureZernikeCoefficientsFromWavefront(
        Wavefront.fromNumpyArray(opd),
        circular_mask,
        BaseMask.from_masked_array(opd.mask))
    return a2, zc 
