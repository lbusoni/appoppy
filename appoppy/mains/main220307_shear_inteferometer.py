import matplotlib.pyplot as plt
import astropy.units as u
import poppy
import numpy as np
from poppy.wfe import WavefrontError
import matplotlib
from appoppy.pyramid_wfs import PyramidWFS
from poppy.poppy_core import PlaneType
from astropy.io import fits
from poppy.optics import ScalarOpticalPathDifference
import skimage


class SegmentedM4(WavefrontError):

    N_PETALS = 6

    def __init__(self, piston=None, name="M4", **kwargs):
        if piston is None:
            piston = np.zeros(self.N_PETALS) * u.nm
        self._piston = piston
        kwargs.update({'name': name})
        super(SegmentedM4, self).__init__(**kwargs)

    def _mask_for_petal(self, x, y, petal_idx):
        return np.logical_and(
            np.arctan2(y, x) < (petal_idx - 2) * np.pi / 3,
            np.arctan2(y, x) > (petal_idx - 3) * np.pi / 3)

    def get_opd(self, wave):
        y, x = self.get_coordinates(wave)  # in meters
        opd = np.zeros(wave.shape, dtype=np.float64)
        for petal_idx in range(self.N_PETALS):
            mask = self._mask_for_petal(x, y, petal_idx)
            opd[mask] = self._piston[petal_idx].to(u.m).value
        return opd


def sector_mask(shape, angle_range, centre=None, radius=None):
    """
    Return a boolean mask for a circular sector. The start/stop angles in  
    `angle_range` should be given in clockwise order.
    """

    x, y = np.ogrid[:shape[0], :shape[1]]
    if centre is None:
        centre = (shape[0] / 2, shape[1] / 2)
    if radius is None:
        radius = np.min(centre)
    cx, cy = centre
    tmin, tmax = np.deg2rad(angle_range)

    # ensure stop angle > start angle
    if tmax < tmin:
        tmax += 2 * np.pi

    # convert cartesian --> polar coordinates
    r2 = (x - cx) * (x - cx) + (y - cy) * (y - cy)
    theta = np.arctan2(x - cx, y - cy) - tmin

    # wrap angles between 0 and 2*pi
    theta %= (2 * np.pi)

    # circular mask
    circmask = r2 <= radius * radius

    # angular mask
    anglemask = theta <= (tmax - tmin)

    return circmask * anglemask


class ShearInterferometerExample(object):

    def __init__(self,
                 zern_coeff=[0.0],
                 r0=0.2,
                 oversample=2,
                 npix=256,
                 rotation_angle=0,
                 seed=0):
        self.wavelength = 2.2e-6 * u.m
        self.telescope_radius = 19.5 * u.m
        self.lambda_over_d = (
            self.wavelength / (2 * self.telescope_radius)).to(
                u.arcsec, equivalencies=u.dimensionless_angles())
        self._kolm_seed = seed
        self.r0 = r0 * u.m * (self.wavelength / (0.5e-6 * u.m))**(6 / 5)
        self.pupil_rotation_angle = rotation_angle

        self._osys = poppy.OpticalSystem(oversample=oversample,
                                         npix=npix,
                                         pupil_diameter=2 * self.telescope_radius)
        self._osys.add_pupil(poppy.KolmogorovWFE(
            name='Turbulence',
            r0=self.r0,
            dz=1,
            seed=self._kolm_seed))
        self._osys.add_pupil(poppy.ZernikeWFE(name='Zernike WFE',
                                              coefficients=zern_coeff,
                                              radius=self.telescope_radius))
        self._osys.add_pupil(SegmentedM4())
        self._osys.add_pupil(poppy.SecondaryObscuration(
            secondary_radius=3.0, n_supports=6, support_width=50 * u.cm))
        self._osys.add_pupil(ScalarOpticalPathDifference(
            opd=0 * u.nm, planetype=PlaneType.pupil))
        self._osys.add_rotation(self.pupil_rotation_angle)
        self._osys.add_pupil(poppy.CircularAperture(radius=self.telescope_radius,
                                                    name='Entrance Pupil'))
        self._osys.add_detector(
            pixelscale=0.5 * self.lambda_over_d / (1 * u.pixel),
            fov_arcsec=2)

        self._zernike_wavefront_plane = 1
        self._m4_wavefront_plane = 2
        self._phase_shift_plane = -4
        self._exit_pupil_plane = -2
        self.display_intermediates = False
        self._reset_intermediate_wfs()

    def _reset_intermediate_wfs(self):
        self._intermediates_wfs = None

    def set_input_wavefront_zernike(self, zern_coeff):
        self._reset_intermediate_wfs()
        in_wfe = poppy.ZernikeWFE(name='Zernike WFE',
                                  coefficients=zern_coeff,
                                  radius=self.telescope_radius)
        self._osys.planes[self._zernike_wavefront_plane] = in_wfe

    def set_m4_petals(self, piston):
        self._reset_intermediate_wfs()
        in_wfe = SegmentedM4(piston, name='Piston WFE')
        self._osys.planes[self._m4_wavefront_plane] = in_wfe

    def set_phase_shift(self, shift_in_lambda):
        self._reset_intermediate_wfs()
        in_wfe = ScalarOpticalPathDifference(
            opd=shift_in_lambda * self.wavelength,
            planetype=PlaneType.pupil,
            name='phase_shift')
        self._osys.planes[self._phase_shift_plane] = in_wfe

    def propagate(self):
        _, self._intermediates_wfs = self._osys.propagate(
            self._osys.input_wavefront(self.wavelength),
            normalize='first',
            display_intermediates=self.display_intermediates,
            return_intermediates=True)

    def run(self):
        self._psf, self._intermediates_wfs = self._osys.calc_psf(
            self.wavelength,
            return_intermediates=True,
            display_intermediates=self.display_intermediates,
            normalize='first')

    def display_psf(self):
        poppy.display_psf(self._psf)

    def _pump_up_zero_for_log_display(self, image):
        ret = image * 1.0
        ret[np.where(image == 0)] = \
            np.min(image[np.where(image != 0)]) / 10
        return ret

    def _wave(self, plane_no):
        if not self._intermediates_wfs:
            self.propagate()
        return self._intermediates_wfs[plane_no]

    def pupil_wavefront(self):
        return self._wave(self._exit_pupil_plane)

    def pupil_phase(self):
        return self.pupil_wavefront().phase

    def pupil_amplitude(self):
        return self.pupil_wavefront().amplitude

    def pupil_intensity(self):
        return self.pupil_wavefront().intensity

    def _display_on_plane(self, what, plane_number, scale='linear'):
        wave = self._wave(plane_number)
        if what == 'intensity':
            image = wave.intensity
            cmap = 'cividis'
        elif what == 'phase':
            image = wave.phase
            cmap = 'twilight'
        else:
            raise Exception('Unknown property to display: %s')
        title = wave.location

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

        if wave.planetype == PlaneType.pupil:
            pc = wave.pupil_coordinates(image.shape, wave.pixelscale)
            extent = [pc[0].min(), pc[0].max(), pc[1].min(), pc[1].max()]
        elif wave.planetype == PlaneType.image:
            extent = [-1, 1, -1, 1]

        plt.clf()
        plt.imshow(image, norm=norm, extent=extent, origin='lower', cmap=cmap)
        plt.title(title)
        plt.colorbar()

    def display_intensity_on_plane(self, plane_number, scale='linear'):
        self._display_on_plane('intensity', plane_number, scale='linear')

    def display_phase_on_plane(self, plane_number, scale='linear'):
        self._display_on_plane('phase', plane_number, scale='linear')

    def display_pupil_intensity(self, **kw):
        self.display_intensity_on_plane(self._exit_pupil_plane, **kw)

    def display_pupil_phase(self, **kw):
        self.display_phase_on_plane(self._exit_pupil_plane, **kw)

    def plot_pupil_intensity_row(self, row=None, scale='linear'):
        aa = self._wave(self._exit_pupil_plane)
        image = aa.intensity
        title = aa.location
        if row is None:
            row = np.int(image.shape[0] / 2)

        if scale == 'linear':
            plt.plot(image[row, :])
        elif scale == 'log':
            plt.semilogy(image[row, :])
        else:
            raise Exception('Unknown scale %s' % scale)
        plt.title(title)
        plt.grid()


class Interferometer():
    '''
    Returns an Optical System whose input wavefront is equal to the sum of
    the wavefront in the exit pupils of the two given optical systems.
    '''

    def __init__(self, optical_system_1, optical_system_2):
        self._os1 = optical_system_1
        self._os2 = optical_system_2
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
        self._os1.set_phase_shift(0)

    def contrast_map(self):
        pass

    def contrast_mask(self):
        pass

    def interferogram(self):
        self._ps = np.arctan2(self._wf_3.intensity - self._wf_1.intensity,
                              self._wf_0.intensity - self._wf_2.intensity)
        self._wrapped = np.arctan2(np.sin(self._ps), np.cos(self._ps))
        self._unwrapped = skimage.restoration.unwrap_phase(self._wrapped)
#        self._ifgram = np.ma.masked_where(
#            self.pupil_intensity() < np.median(self.pupil_intensity()) / 1e9,
#            (self._unwrapped * self._wf_0.wavelength / (2 * np.pi)).to(u.nm).value)
        self._ifgram = np.ma.masked_array(
            (self._unwrapped * self._wf_0.wavelength / (2 * np.pi)).to(u.nm).value,
            mask=self.global_mask()
        )
        return self._ifgram

    def global_mask(self):
        mask1 = self._mask_from_median(self._os1.pupil_intensity(), 10)
        mask2 = self._mask_from_median(self._os2.pupil_intensity(), 10)
        return np.ma.mask_or(mask1, mask2)

    def interferogram_mask(self, cut=1000):
        return self._mask_from_median(self.interferogram(), cut)

    def _mask_from_median(self, image, cut):
        imask = image < np.median(image) / cut
        mask = np.zeros(image.shape)
        mask[imask] = 1
        return mask

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


def main2(r0=np.inf,
          petals=np.array([800, 0, 0, 0, 0, 0]) * u.nm,
          rotation_angle=15,
          zernike=[0, 0],
          seed=None):
    if seed is None:
        seed = np.random.randint(2147483647)

    model1 = ShearInterferometerExample(
        r0=r0,
        seed=seed,
        rotation_angle=rotation_angle)
    model1.set_input_wavefront_zernike(zernike)
    model1.set_m4_petals(petals)

    model2 = ShearInterferometerExample(
        r0=r0,
        seed=seed)
    model2.set_input_wavefront_zernike(zernike)
    model2.set_m4_petals(petals)

    i4 = Interferometer(model1, model2)
    i4.combine()
    i4.acquire()
    i4.display_interferogram()

    return model1, model2, i4


def compute_jumps(model1, i4):
    r = model1.pupil_rotation_angle
    image = i4.interferogram()
    angs = (-180, -180 + r, -120, -120 + r, -60, -
            60 + r, 0, r, 60, 60 + r, 120, 120 + r, 180)
    res = np.zeros(len(angs) - 1)
    for i in range(len(angs) - 1):
        ifm = _mask_ifgram(image, (angs[i], angs[i + 1]))
        res[i] = np.ma.median(ifm)
    return res


def _mask_ifgram(ifgram, angle_range):
    smask1 = sector_mask(ifgram.shape,
                         (angle_range[0], angle_range[1]))
    mask = np.ma.mask_or(ifgram.mask, ~smask1)
    return np.ma.masked_array(ifgram, mask=mask)


def test_kolmo(niter=3, r0_at_500=0.2, dz=1, wl=0.5e-6):
    telescope_radius = 19.5 * u.m
    r0 = r0_at_500 * (wl / 0.5e-6)**(6 / 5)
    for i in range(niter):
        ss = poppy.OpticalSystem(pupil_diameter=2 * telescope_radius)
        ss.add_pupil(poppy.KolmogorovWFE(
            r0=r0, dz=dz, outer_scale=40, inner_scale=0.05, kind='von Karman'))
        ss.add_pupil(poppy.CircularAperture(radius=telescope_radius,
                                            name='Entrance Pupil'))

        ss.add_detector(pixelscale=0.20, fov_arcsec=2.0)
        hdu = ss.calc_psf(wavelength=wl)[0]
        if i == 0:
            psfs = np.zeros((niter, hdu.shape[0], hdu.shape[1]))
        psfs[i] = hdu.data
    hdu.data = psfs.sum(axis=0)
    hdulist = fits.HDUList(hdu)
    print("0.98*l/r0 = %g  - FWHM %g " % (
        0.98 * wl / r0 / 4.848e-6, poppy.utils.measure_fwhm_radprof(hdulist)))
    plt.figure()
    poppy.utils.display_profiles(hdulist)
    plt.figure()
    poppy.utils.display_psf(hdulist)
    return psfs, hdulist, ss
