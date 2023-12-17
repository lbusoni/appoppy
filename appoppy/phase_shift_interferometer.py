import numpy as np
import poppy
import skimage
from astropy import units as u
import matplotlib
import matplotlib.pyplot as plt
from appoppy.mask import mask_from_median
import logging
from appoppy.snapshotable import Snapshotable


class PsiSnapshotEntry(object):
    SHOULD_UNWRAP = "SHOULD_UNWRAP"


class PhaseShiftInterferometer(Snapshotable):
    '''
    Returns an Optical System mimicking a phase shift interferometer

    The system is fed with an input wavefront equal to the sum of
    the wavefront in the exit pupils of the two given optical systems.

    The system has the capability of measuring an interferogram using
    phase shift technique


    Parameters
    ----------
        optical_system_1: poppy.OpticalSystem
           optical system for reference beam. It must implement set_phase_shift

        optical_system_2: poppy.OpticalSystem


    '''

    def __init__(self, optical_system_1, optical_system_2):
        self._os1 = optical_system_1
        self._os2 = optical_system_2
        self._should_unwrap = False
        self._log = logging.getLogger('appoppy')
        self._create_system()
        self._ios_wf = None
        # assert self._os1.wavelength == self._os2.wavelength
        # TODO probably some check?

    def get_snapshot(self, prefix='PSI'):
        snapshot = {}
        snapshot[PsiSnapshotEntry.SHOULD_UNWRAP] = self._should_unwrap
        return Snapshotable.prepend(prefix, snapshot)

    def _create_system(self):
        self._ios = poppy.OpticalSystem(
            oversample=self._os1.optical_system.oversample,
            npix=self._os1.optical_system.npix,
            pupil_diameter=2 * self._os1.telescope_radius)

        self._ios.add_pupil(
            poppy.CircularAperture(radius=self._os1.telescope_radius,
                                   name='Entrance Pupil'))

    @property
    def combined_wavefront(self):
        return self._propagate()

    def _propagate(self):
        self._wv1 = self._os1.pupil_wavefront()
        self._wv2 = self._os2.pupil_wavefront()
        ret = self._ios.propagate(
            self._ios.input_wavefront(self._os1.wavelength,
                                      self._wv1 + self._wv2))
        return ret

    def _phase_shift_step(self, step):
        self._log.info('phase shift step %g' % step)
        self._os1.set_phase_shift(step)
        return self._propagate()

    def acquire(self):
        '''
        Acquire an interferogram with a 4 step phase shift

        Let's define the interferogram intensity dependence on the phase x as
        I = A + B cos(x)

        The routine performs 4 exposures acting on the phase shift of os1
        to estimate
        I0 = A + B cos(x)
        I1 = A + B cos(x+pi/2) = A - B sin(x)
        I2 = A + B cos(x+pi) = A - B cos(x)
        I3 = A + B cos(x+3pi/2) = A - B sin(x)

        The wrapped phase map is then computed as arctan(I3-I1/I0-I2)
        The maps of amplitude B, offset A and visibility (B/A) are also
        estimated

        The resulting phase map is bounded in the domain (-pi,pi)
        '''
        self._log.info('phase shift acquisition')
        self._wf_0 = self._phase_shift_step(0)
        self._wf_1 = self._phase_shift_step(0.25)
        self._wf_2 = self._phase_shift_step(0.5)
        self._wf_3 = self._phase_shift_step(0.75)
        bsin = self._wf_3.intensity - self._wf_1.intensity
        bcos = self._wf_0.intensity - self._wf_2.intensity
        self._ps = np.arctan2(bsin, bcos)
        self._b_map = np.sqrt((bsin**2 + bcos**2) / 4)
        self._a_map = 0.25 * (self._wf_0.intensity + self._wf_1.intensity +
                              self._wf_2.intensity + self._wf_3.intensity)
        self._visibility = self._b_map / self._a_map
        self._os1.set_phase_shift(0)

    def acquire_without_phase_shift(self, A, B):
        '''
        Acquire a phase map without phase shift.

        The amplitude must be estimated/calibrated before

        Let's define the interferogram intensity dependence on the phase x as
        I = A + B cos(x)

        The routine performs a single exposures to estimate
        x = arccos( (I-A)/B )

        The maps of amplitude B and offset A must have been estimated/
        calibrated before

        The resulting phase map is bounded in the domain (0,pi)
        '''
        self._log.info('acquisition without phase shift')
        self._wf_0 = self._phase_shift_step(0)
        self._ps = np.arccos((self._wf_0.intensity - A) / B)
        self._visibility = B / A

    def visibility(self):
        return self._visibility

    def visibility_mask(self):
        return mask_from_median(self._visibility, 2)

    def interferogram(self):
        '''

        Returns
        -------
        opd: numpy array
            optical path difference between the 2 beams in nm units,
            bounded in the range (-wavelength/2, wavelength/2)
        '''

        self._wrapped = np.arctan2(np.sin(self._ps), np.cos(self._ps))
        if self._should_unwrap:
            self._unwrapped = skimage.restoration.unwrap_phase(self._wrapped)
        else:
            self._unwrapped = self._wrapped.copy()
        self._ifgram = np.ma.masked_array(
            (self._unwrapped * self._wf_0.wavelength / (2 * np.pi)).to_value(u.nm),
            mask=self.global_mask()
        )
        return self._ifgram

    def global_mask(self):
        mask1 = mask_from_median(self._os1.pupil_intensity(), 10)
        mask2 = mask_from_median(self._os2.pupil_intensity(), 10)
        return np.ma.mask_or(mask1, mask2)

    def interferogram_mask(self, cut=1000):
        return mask_from_median(self.interferogram(), cut)

    def pupil_intensity(self):
        return self.combined_wavefront.intensity

    def pupil_phase(self):
        return self.combined_wavefront.phase

    def display_pupil_intensity(self, **kwargs):
        self.combined_wavefront.display(what='intensity', **kwargs)

    def display_pupil_phase(self, **kwargs):
        self.combined_wavefront.display(what='phase', **kwargs)

    def _pump_up_zero_for_log_display(self, image):
        ret = image * 1.0
        ret[np.where(image == 0)] = \
            np.min(image[np.where(image != 0)]) / 10
        return ret

    def display_interferogram(self, scale='linear'):
        image = self.interferogram()
        title = 'Phase shifted interferogram'

        if scale == 'linear':
            # norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
            norm = matplotlib.colors.Normalize()
        elif scale == 'log':
            image = self._pump_up_zero_for_log_display(image)
            vmax = np.max(image)
            vmin = np.maximum(np.min(image), np.max(image) / 1e4)
            norm = matplotlib.colors.LogNorm(vmin=vmin, vmax=vmax)
        else:
            raise Exception('Unknown scale %s' % scale)

        pc = self._wf_0.pupil_coordinates(image.shape, self._wf_0.pixelscale)
        extent = [pc[0].min(), pc[0].max(), pc[1].min(), pc[1].max()]

        plt.clf()
        plt.imshow(image,
                   norm=norm,
                   extent=extent,
                   origin='lower',
                   cmap='twilight')
        plt.title(title)
        plt.colorbar()
