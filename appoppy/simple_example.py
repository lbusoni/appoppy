import numpy as np
import poppy
from poppy.wfe import WavefrontError
import astropy.units as u
from poppy import utils
import matplotlib
import matplotlib.pyplot as plt
from appoppy.pyramid_wfs import PyramidWFS
from appoppy.point_diffraction_interferometer import PointDiffractionInterferometer
from appoppy.zernike_mask import ZernikeMaskWFS
from poppy.poppy_core import PlaneType


class FocalPlaneWFSExample(object):

    def __init__(self, zern_coeff=[0.0], oversample=4, npix=256):
        self.wavelength = 1e-6 * u.m
        self.telescope_radius = 4.1 * u.m
        self.lambda_over_d = (
            self.wavelength / (2 * self.telescope_radius)).to(
                u.arcsec, equivalencies=u.dimensionless_angles())

        self._osys = poppy.OpticalSystem(oversample=oversample,
                                         npix=npix,
                                         pupil_diameter=2 * self.telescope_radius)
        self._osys.add_pupil(poppy.ZernikeWFE(name='Zernike WFE',
                                              coefficients=zern_coeff,
                                              radius=self.telescope_radius))
        self._osys.add_pupil(poppy.CircularAperture(radius=self.telescope_radius,
                                                    name='Entrance Pupil'))
        self._zernike_wavefront_plane = 0
        self._wfs_plane_from = 2
        self._exit_pupil_plane = -2
        self.display_intermediates = True

        # osys.add_pupil(poppy.CircularAperture(radius=4*self.telescope_radius, name='Exit Pupil'))
        self._osys.add_detector(
            pixelscale=0.1 * self.lambda_over_d / (1 * u.pixel),
            fov_arcsec=0.5)
        # self.propagate_wavefront(zern_coeff)

    def set_input_wavefront_zernike(self, zern_coeff):
        in_wfe = poppy.ZernikeWFE(name='Zernike WFE',
                                  coefficients=zern_coeff,
                                  radius=self.telescope_radius)
        self._osys.planes[self._zernike_wavefront_plane] = in_wfe

    def set_m4_petals(self, piston):
        in_wfe = SegmentedM4(piston, name='Piston WFE')
        self._osys.planes[self._zernike_wavefront_plane] = in_wfe

    def use_pyramid_wfs(self,
                        pyr_angle=2.5e-5 * u.m / u.arcsec,
                        pyr_edge_in_lambda_d=0.5):
        pyrwfs = PyramidWFS(
            name='My Pyramid',
            pyr_angle=pyr_angle,
            pyr_edge_thickness=pyr_edge_in_lambda_d * self.lambda_over_d)
        pyrwfs.add_to_system(self._osys, self._wfs_plane_from)

    def use_zernike_mask_wfs(self):
        zmwfs = ZernikeMaskWFS(
            name='My ZernikeMask')
        zmwfs.add_to_system(self._osys, self._wfs_plane_from)

    def use_pid(self,
                pinhole_radius_in_lambda_d=1.0,
                transmittance_outer_region=0.01):
        pinhole_radius = pinhole_radius_in_lambda_d * \
            self.lambda_over_d
        pid = PointDiffractionInterferometer(
            name='My PID',
            pinhole_radius=pinhole_radius,
            transmittance_outer_region=transmittance_outer_region)
        pid.add_to_system(self._osys, self._wfs_plane_from)

    def run(self):
        self._psf, self._intermediates_wfs = self._osys.calc_psf(
            self.wavelength,
            return_intermediates=True,
            display_intermediates=self.display_intermediates)

    def display_psf(self):
        poppy.display_psf(self._psf)

    def _pump_up_zero_for_log_display(self, image):
        ret = image * 1.0
        ret[np.where(image == 0)] = \
            np.min(image[np.where(image != 0)]) / 10
        return ret

    def pupil_intensity(self):
        return self._intermediates_wfs[self._exit_pupil_plane].intensity

    def display_intensity_on_plane(self, plane_number, scale='linear'):
        wave = self._intermediates_wfs[plane_number]
        image = wave.intensity
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
        plt.imshow(image, norm=norm, extent=extent)
        plt.title(title)
        plt.colorbar()

    def display_intensity_on_pyramid_tip(self, **kw):
        self.display_intensity_on_plane(self._wfs_plane_from + 1, **kw)

    def display_pupil_intensity(self, **kw):
        self.display_intensity_on_plane(self._exit_pupil_plane, **kw)

    def plot_pupil_intensity_row(self, row=None, scale='linear'):
        aa = self._intermediates_wfs[self._exit_pupil_plane]
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


class SegmentedM4(WavefrontError):

    @utils.quantity_input(piston=u.meter)
    def __init__(self, piston, name="Piston WFE", **kwargs):
        self._piston = piston
        kwargs.update({'name': name})
        super(SegmentedM4, self).__init__(**kwargs)

    def get_opd(self, wave):
        y, x = self.get_coordinates(wave)  # in meters
        opd = np.zeros(wave.shape, dtype=np.float64)
        opd[np.where(x > 0)] = self._piston.to(u.m).value
        return opd


class PyramidWFSExample(FocalPlaneWFSExample):

    def __init__(self,
                 pyr_angle=2.5e-5 * u.m / u.arcsec,
                 pyr_edge_in_lambda_d=0,
                 **kwargs):
        FocalPlaneWFSExample.__init__(self, **kwargs)
        pyrwfs = PyramidWFS(
            name='My Pyramid',
            pyr_angle=pyr_angle,
            pyr_edge_thickness=pyr_edge_in_lambda_d * self.lambda_over_d)
        pyrwfs.add_to_system(self._osys, self._wfs_plane_from)


def main_pyramid(zern_coeff=[0, 0, 0, 0e-9]):
    model = PyramidWFSExample(
        zern_coeff=zern_coeff,
        pyr_angle=2.5e-5 * u.m / u.arcsec,
        pyr_edge_in_lambda_d=0)
    model.run()
    plt.figure(2)
    model.display_pupil_intensity()
    return model


def main_zernike_mask(zern_coeff=[0, 0, 0, 0e-9]):
    model = FocalPlaneWFSExample(zern_coeff=zern_coeff)
    model.use_zernike_mask_wfs()
    model.run()
    plt.figure(2)
    model.display_pupil_intensity()
    return model


def main_pid(zern_coeff=[0, 0, 0, 0e-9],
             pinhole_radius_in_lambda_d=1.0,
             transmittance_outer_region=0.01):
    model = FocalPlaneWFSExample(zern_coeff=zern_coeff)
    model.use_pid(pinhole_radius_in_lambda_d=pinhole_radius_in_lambda_d,
                  transmittance_outer_region=transmittance_outer_region)
    model.run()
    plt.figure(2)
    model.display_pupil_intensity()
    return model
