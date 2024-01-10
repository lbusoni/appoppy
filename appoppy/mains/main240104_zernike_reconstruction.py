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
from arte.utils.zernike_projection_on_subaperture import ZernikeProjectionOnSubaperture


def tip_vs_ciaociao_rotation_angle():
    '''
    Plot of reconstructed OPDs from CiaoCiao at different rotation angles.
    The wavefront aberration is only tip with a2 = 1 wv.
    '''
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


def test_of_zernike_measurement_from_interferogram(
        zernikes=[0, 2.2]):
    '''
    Test the measurement of Zernike coefficients on a pure interferogram.
    A flat wavefront and an aberrated one are the inputs of
    PhaseShiftInterferometer class and the ModalDecomposer class is used to 
    reconstruct the Zernike coefficients.
    '''
    model1 = EltForPetalometry(r0=np.inf, zern_coeff=[0, 0] * u.um)
    model2 = EltForPetalometry(r0=np.inf, zern_coeff=zernikes * u.um)
    psi = PhaseShiftInterferometer(model1, model2, should_unwrap=True)
    psi.acquire()
    opd = psi.interferogram()
    wrapped_phase = psi._wrapped
    plt.figure()
    plt.imshow(wrapped_phase, origin='lower')
    plt.colorbar(label='[rad]')
    plt.title('Wrapped phase')
    plt.figure()
    plt.imshow(opd, origin='lower')
    plt.colorbar(label='[nm]')
    plt.title('Unwrapped OPD')
    plt.figure()
    plt.plot(opd[int(opd.shape[0] / 2),:])
    plt.title('Cut on x axis of unwrapped phase')
   
    circular_mask = CircularMask(frameShape=opd.shape,
                                 maskRadius=opd.shape[0] / 2,
                                 maskCenter=(opd.shape[0] / 2, opd.shape[0] / 2)
                                 )
    md = ModalDecomposer(len(zernikes))
    zc = md.measureZernikeCoefficientsFromWavefront(
        Wavefront.fromNumpyArray(opd),
        circular_mask,
        BaseMask(opd.mask))
    print('Measured Zernike coefficients: %s' % zc.toNumpyArray())
    return zc


def test_of_zernike_measurement_from_interferogram_subaperture(
        zernikes=[0, 2.2]):
    '''
    Test the measurement of Zernike coefficients on a subaperture defined
    within the interferogram. The ModalDecomposer class measures the
    coefficients on the subaperture and then the Negro84 matrix (the inverse)
    is used to derive the coefficients on the pupil. 
    '''
    model1 = EltForPetalometry(r0=np.inf, zern_coeff=[0, 0] * u.um)
    model2 = EltForPetalometry(r0=np.inf, zern_coeff=zernikes * u.um)
    psi = PhaseShiftInterferometer(model1, model2, should_unwrap=True)
    psi.acquire()
    opd = psi.interferogram()
    wrapped_phase = psi._wrapped
    
    subap_center = [128, 41]
    subap_radius = 20
    subap_circ_mask = CircularMask(
        frameShape=opd.shape, maskCenter=subap_center, maskRadius=subap_radius)
    opd_in_subap = np.ma.masked_array(opd.data, mask=subap_circ_mask.mask())
    
    plt.figure()
    plt.imshow(wrapped_phase, origin='lower')
    plt.colorbar(label='[rad]')
    plt.title('Wrapped phase')
    plt.figure()
    plt.imshow(opd, origin='lower')
    plt.colorbar(label='[nm]')
    plt.title('Unwrapped OPD')
    plt.figure()
    plt.imshow(opd_in_subap, origin='lower')
    plt.colorbar(label='[nm]')
    plt.title('OPD in subaperture')
    
    md = ModalDecomposer(10)
    zc = md.measureZernikeCoefficientsFromWavefront(
        Wavefront.fromNumpyArray(opd_in_subap),
        subap_circ_mask,
        BaseMask(subap_circ_mask.mask()))
    zp = ZernikeProjectionOnSubaperture(
        pupilRadiusInMeter=opd.shape[0] / 2, subapsRadiusInMeter=subap_radius,
        subapOffAxisRadiusInMeter=np.sqrt(
            subap_center[1] ** 2 + subap_center[0] ** 2),
        subapOffAxisAzimuthInDegrees=np.tan(subap_center[0] / subap_center[1]))
    
    zern_coeff_in_pupil = np.dot(
        zc.toNumpyArray(), np.linalg.pinv(zp.get_projection_matrix()))                         
    print('Measured Zernike coefficients in nm: %s' % zern_coeff_in_pupil)
    return zern_coeff_in_pupil


def test_of_tip_measurement_from_ciaociao_acquisition(tip_coeff_in_um=2.2):
    '''
    The reconstruction of tip on the CiaoCiao reconstructed phase fails because
    the unwrapping does not work in this case. This is due to the fact that the
    wrapped phase shows a weird behavior within the points of the mask.
    '''
    pet = Petalometer(rotation_angle=20,
                 zernike=np.array([0, tip_coeff_in_um]) * u.um,
                 should_unwrap=True)
    pet.sense_wavefront_jumps()
    #
    wrapped_phase = pet._i4._wrapped
    # wrapped_phase_masked = np.ma.masked_array(
    #     wrapped_phase, mask=pet._i4.global_mask())
    # unwrapped_phase = unwrap_phase(wrapped_phase_masked)
    # opd = unwrapped_phase * (
    #     pet._i4._wf_0.wavelength / (2 * np.pi)).to_value(u.nm)
    
    opd = pet.reconstructed_phase
    plt.figure()
    plt.imshow(opd, origin='lower')
    plt.colorbar(label='[nm]')
    plt.title('Unwrapped OPD')
    plt.figure()
    plt.imshow(wrapped_phase, origin='lower')
    plt.colorbar(label='[rad]')
    plt.title('Wrapped phase')
    # plt.figure()
    # plt.imshow(wrapped_phase_masked, origin='lower')
    # plt.colorbar(label='[rad]')
    # plt.title('Wrapped phase masked')
       
    circular_mask = CircularMask(frameShape=opd.shape,
                                 maskRadius=opd.shape[0] / 2,
                                 maskCenter=(opd.shape[0] / 2, opd.shape[0] / 2)
                                 )
    md = ModalDecomposer(10)
    zc = md.measureZernikeCoefficientsFromWavefront(
        Wavefront.fromNumpyArray(opd),
        circular_mask,
        BaseMask(opd.mask))
    return zc
# , opd, wrapped_phase_masked, circular_mask


def test_of_tip_measurement_from_ciao_ciao_acquisition_with_subaperture(
        tip_coeff_in_um=2.2):
    '''
    First we perform the phase unwrapping in a subaperture defined in a way to
    avoid masked points. Then we estimate the tip coefficient with the
    ModalDecomposer class acting on the subaperture and we convert it with
    Negro84 matrix to get Zernikes on the pupil. 
    The results are consistent with the math, showing that a tipped wavefront
    is seen on the CiaoCiao as a combination of tip and tilt. To retrieve
    the input tip coefficient a2, the measured a2' must be scaled by 1-cosR
    and the measured a3' must be scaled by sinR, where R is the CiaoCiao
    rotation angle.
    '''
    rot_angle = 20 * u.deg
    pet = Petalometer(rotation_angle=rot_angle.value,
                      zernike=np.array([0, tip_coeff_in_um]) * u.um,
                      should_unwrap=False)
    pet.sense_wavefront_jumps()
    wrapped_opd = pet.reconstructed_phase
    subap_center = [140, 40]
    subap_radius = 20
    subap_circ_mask = CircularMask(
        frameShape=wrapped_opd.shape, maskCenter=subap_center, maskRadius=subap_radius)
    opd_in_subap = np.ma.masked_array(
        wrapped_opd.data, mask=subap_circ_mask.mask())
    phase_in_subap = opd_in_subap * 2 * np.pi / pet.wavelength.to(u.nm) 
    unwrapped_opd_in_subap = unwrap_phase(
        phase_in_subap) * pet.wavelength.to(u.nm) / (2 * np.pi)
    
    plt.figure()
    plt.imshow(wrapped_opd, origin='lower')
    plt.colorbar(label='[nm]')
    plt.title('Wrapped OPD')
    plt.figure()
    plt.imshow(opd_in_subap, origin='lower')
    plt.colorbar(label='[nm]')
    plt.title('Wrapped OPD in subaperture')
    plt.figure()
    plt.imshow(unwrapped_opd_in_subap, origin='lower')
    plt.colorbar(label='[nm]')
    plt.title('Unwrapped OPD in subaperture')
   
    md = ModalDecomposer(10)
    zc = md.measureZernikeCoefficientsFromWavefront(
        Wavefront.fromNumpyArray(unwrapped_opd_in_subap),
        subap_circ_mask,
        BaseMask(subap_circ_mask.mask()))
    zp = ZernikeProjectionOnSubaperture(
        pupilRadiusInMeter=wrapped_opd.shape[0] / 2,
        subapsRadiusInMeter=subap_radius,
        subapOffAxisRadiusInMeter=np.sqrt(
            subap_center[1] ** 2 + subap_center[0] ** 2),
        subapOffAxisAzimuthInDegrees=np.tan(subap_center[0] / subap_center[1]))

    zern_coeff_in_pupil = np.dot(
        zc.toNumpyArray(), np.linalg.pinv(zp.get_projection_matrix()))                         
    print('\nMeasured Zernike coefficients in nm: %s' % zern_coeff_in_pupil)
    
    tip_factor = 1 - np.cos(rot_angle)
    tilt_factor = np.sin(rot_angle)
    print('\nDerived tip in nm: %s' % (zern_coeff_in_pupil[0] / tip_factor))
    print('Derived tilt in nm: %s' % (zern_coeff_in_pupil[1] / tilt_factor))

    return zern_coeff_in_pupil
