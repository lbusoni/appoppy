import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import astropy.units as u
import poppy
from appoppy.petaled_m4 import PetaledM4
from poppy.optics import ScalarOpticalPathDifference
from poppy.poppy_core import PlaneType
import skimage
from appoppy.mask import mask_from_median
from appoppy.elt_aperture import ELTAperture
import logging
from appoppy.maory_residual_wfe import MaoryResidualWavefront


class EltForPetalometry(object):

    def __init__(self,
                 zern_coeff=[0.0],
                 use_simulated_residual_wfe=True,
                 r0=999999,
                 tracking_number='20210518_223459.0',
                 rotation_angle=0,
                 kolm_seed=0,
                 residual_wavefront_start_from=100,
                 residual_wavefront_step=0,
                 residual_wavefront_average_on=1,
                 name=''):
        self.name = name
        self._log = logging.getLogger('EltForPetalometry-%s' % self.name)
        self.wavelength = 2.2e-6 * u.m
        self.telescope_radius = 19.8 * u.m
        self.lambda_over_d = (
            self.wavelength / (2 * self.telescope_radius)).to(
                u.arcsec, equivalencies=u.dimensionless_angles())
        self._kolm_seed = kolm_seed
        self.pupil_rotation_angle = rotation_angle
        self._use_sim_res = use_simulated_residual_wfe

        if not self._use_sim_res:
            r0l = r0 * u.m * (self.wavelength / (0.5e-6 * u.m))**(6 / 5)
            atmo_wfe = poppy.KolmogorovWFE(
                name='Turbulence',
                r0=r0l,
                dz=1 * u.m,
                seed=self._kolm_seed)
            npix = 480
            time_step = 60  # each kolmo layer is decorrelated from the previous one
            elt_aperture = ELTAperture()
        else:
            atmo_wfe = MaoryResidualWavefront(
                tracking_number,
                start_from=residual_wavefront_start_from,
                step=residual_wavefront_step,
                average_on=residual_wavefront_average_on)
            npix = atmo_wfe.shape[-1]
            time_step = atmo_wfe.time_step
            elt_aperture = ELTAperture(atmo_wfe.pupiltag)

        self.pixelsize = 2 * self.telescope_radius / npix
        self.time_step = time_step

        self._osys = poppy.OpticalSystem(
            oversample=2,
            npix=npix,
            pupil_diameter=2 * self.telescope_radius)
        self._osys.add_pupil(atmo_wfe)
        self._osys.add_pupil(poppy.ZernikeWFE(name='Zernike WFE',
                                              coefficients=zern_coeff,
                                              radius=self.telescope_radius))
        self._osys.add_pupil(PetaledM4())
#        self._osys.add_pupil(poppy.SecondaryObscuration(
#            secondary_radius=3.0, n_supports=6, support_width=50 * u.cm))
        self._osys.add_pupil(elt_aperture)
        self._osys.add_pupil(ScalarOpticalPathDifference(
            opd=0 * u.nm, planetype=PlaneType.pupil))
        self._osys.add_rotation(-1 * self.pupil_rotation_angle)
        self._osys.add_pupil(poppy.CircularAperture(radius=self.telescope_radius,
                                                    name='Entrance Pupil'))
        self._osys.add_detector(
            pixelscale=0.5 * self.lambda_over_d / (1 * u.pixel),
            fov_arcsec=1)

        self._turbulence_plane = 0
        self._zernike_wavefront_plane = 1
        self._m4_wavefront_plane = 2
        self._phase_shift_plane = -4
        self._exit_pupil_plane = -2
        self.display_intermediates = False
        self._reset_intermediate_wfs()

    def _reset_intermediate_wfs(self):
        self._intermediates_wfs = None

    # def set_atmospheric_wavefront(self, atmo_opd):
    #     self._reset_intermediate_wfs()
    #     pass

    def set_input_wavefront_zernike(self, zern_coeff):
        self._reset_intermediate_wfs()
        in_wfe = poppy.ZernikeWFE(name='Zernike WFE',
                                  coefficients=zern_coeff,
                                  radius=self.telescope_radius)
        self._osys.planes[self._zernike_wavefront_plane] = in_wfe

    def set_m4_petals(self, piston):
        self._reset_intermediate_wfs()
        in_wfe = PetaledM4(piston, name='Piston WFE')
        self._osys.planes[self._m4_wavefront_plane] = in_wfe

    def set_phase_shift(self, shift_in_lambda):
        self._reset_intermediate_wfs()
        in_wfe = ScalarOpticalPathDifference(
            opd=shift_in_lambda * self.wavelength,
            planetype=PlaneType.pupil,
            name='phase_shift')
        self._osys.planes[self._phase_shift_plane] = in_wfe

    def set_step_idx(self, step_idx):
        # advance residual phase screen only in the case of MORFEO
        # otherwise is Kolmogorov, every screen uncorrelated, no wind speed
        if self._use_sim_res:
            self._reset_intermediate_wfs()
            self._osys.planes[self._turbulence_plane].set_step_idx(step_idx)

    def propagate(self):
        self._log.info('propagating')
        _, self._intermediates_wfs = self._osys.propagate(
            self._osys.input_wavefront(self.wavelength),
            normalize='first',
            display_intermediates=self.display_intermediates,
            return_intermediates=True)

    def compute_psf(self):
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

    def pupil_opd_unwrapped(self):
        mask = self.pupil_mask()
        minv = np.logical_not(mask).astype(int)
        unwrap = skimage.restoration.unwrap_phase(
            self.pupil_phase() * minv)
        return np.ma.array(unwrap / 2 / np.pi * self.wavelength.to_value(u.nm),
                           mask=mask)

    def pupil_mask(self):
        return mask_from_median(self.pupil_intensity(), 10)

    def pupil_opd(self):
        osys = self._osys
        opd0 = osys.planes[self._turbulence_plane].get_opd(
            osys.input_wavefront(self.wavelength))
        opd1 = osys.planes[self._zernike_wavefront_plane].get_opd(
            osys.input_wavefront(self.wavelength))
        opd2 = osys.planes[self._m4_wavefront_plane].get_opd(
            osys.input_wavefront(self.wavelength))

        opdm = np.ma.MaskedArray(opd0 + opd1 + opd2, mask=self.pupil_mask())
        return opdm * 1e9

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
        self._display_on_plane('intensity', plane_number, scale=scale)

    def display_phase_on_plane(self, plane_number):
        self._display_on_plane('phase', plane_number, scale='linear')

    def display_pupil_intensity(self, **kw):
        self.display_intensity_on_plane(self._exit_pupil_plane, **kw)

    def display_pupil_phase(self, **kw):
        self.display_phase_on_plane(self._exit_pupil_plane, **kw)

    def display_pupil_opd(self, **kw):
        wave = self._wave(0)
        image = self.pupil_opd()
        norm = matplotlib.colors.Normalize()
        cmap = 'cividis'
        pc = wave.pupil_coordinates(image.shape, wave.pixelscale)
        extent = [pc[0].min(), pc[0].max(), pc[1].min(), pc[1].max()]

        plt.clf()
        plt.imshow(image, norm=norm, extent=extent, origin='lower', cmap=cmap)
        plt.title("Total OPD")
        plt.colorbar()

    def plot_pupil_intensity_row(self, row=None, scale='linear'):
        aa = self._wave(self._exit_pupil_plane)
        image = aa.intensity
        title = aa.location
        if row is None:
            row = int(image.shape[0] / 2)

        if scale == 'linear':
            plt.plot(image[row, :])
        elif scale == 'log':
            plt.semilogy(image[row, :])
        else:
            raise Exception('Unknown scale %s' % scale)
        plt.title(title)
        plt.grid()
