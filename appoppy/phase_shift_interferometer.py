import numpy as np
import poppy
import skimage
from astropy import units as u
import matplotlib
import matplotlib.pyplot as plt
from appoppy.mask import mask_from_median


class PhaseShiftInterferometer():
    '''
    Returns an Optical System whose input wavefront is equal to the sum of
    the wavefront in the exit pupils of the two given optical systems.


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
        # assert self._os1.wavelength == self._os2.wavelength
        # TODO probably some check?

    def combine(self):
        self._ios = poppy.OpticalSystem(
            oversample=self._os1._osys.oversample,
            npix=self._os1._osys.npix,
            pupil_diameter=2 * self._os1.telescope_radius)

        self._ios.add_pupil(
            poppy.CircularAperture(radius=self._os1.telescope_radius,
                                   name='Entrance Pupil'))
        self._ios_wf = self._propagate()

    def _propagate(self):
        self._wv1 = self._os1.pupil_wavefront()
        self._wv2 = self._os2.pupil_wavefront()
        return self._ios.propagate(
            self._ios.input_wavefront(self._os1.wavelength,
                                      self._wv1 + self._wv2))

    def _phase_shift_step(self, step):
        self._os1.set_phase_shift(step)
        return self._propagate()

    def acquire(self):
        self._wf_0 = self._phase_shift_step(0)
        self._wf_1 = self._phase_shift_step(0.25)
        self._wf_2 = self._phase_shift_step(0.5)
        self._wf_3 = self._phase_shift_step(0.75)
        bsin = self._wf_3.intensity - self._wf_1.intensity
        bcos = self._wf_0.intensity - self._wf_2.intensity
        self._ps = np.arctan2(bsin, bcos)
        b = np.sqrt((bsin**2 + bcos**2) / 4)
        a = 0.25 * (self._wf_0.intensity + self._wf_1.intensity +
                    self._wf_2.intensity + self._wf_3.intensity)
        self._visibility = b / a
        self._os1.set_phase_shift(0)

    def visibility(self):
        return self._visibility

    def visibility_mask(self):
        return mask_from_median(self._visibility, 2)

    def interferogram(self):
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
        return self._ios_wf.intensity

    def pupil_phase(self):
        return self._ios_wf.phase

    def display_pupil_intensity(self):
        self._ios_wf.display(what='intensity')

    def display_pupil_phase(self):
        self._ios_wf.display(what='phase')

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
