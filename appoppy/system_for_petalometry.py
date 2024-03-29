import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import astropy.units as u
import poppy
from appoppy import elt_aperture
from appoppy.petaled_m4 import PetaledM4
from poppy.optics import ScalarOpticalPathDifference
from poppy.poppy_core import PlaneType
import skimage
from appoppy.mask import mask_from_median
from appoppy.elt_aperture import ELTAperture
import logging
from appoppy.maory_residual_wfe import MaoryResidualWavefront
from appoppy.low_wind_effect import LowWindEffectWavefront
from appoppy.snapshotable import Snapshotable, SnapshotPrefix


class EfpSnapshotEntry(object):
    NAME = "NAME"
    WAVELENGTH = "WL"
    TELESCOPE_RADIUS = "TELE_RAD"
    NPIX = "NPIX"
    KOLMOGOROV_SEED = "KOLM_SEED"
    PUPIL_ROTATION_ANGLE = "PUP_ROT_ANG"
    LWE_WIND_SPEED = "LWE_WIND_SPEED"
    ZERNIKE_COEFFICIENTS = "ZERN_COEFF"
    R0 = "R0"
    PASSATA_TRACKING_NUMBER = "TRACKNUM"
    PASSATA_START_FROM = "RES_START_FROM"


class BaseSystemForPetalometry(Snapshotable):

    PLANE_EXIT_PUPIL = 'exit pupil'

    def __init__(self, npix=256, oversample=1, wavelength=2.2*u.um, telescope_radius=4*u.m, name=''):
        self._npix = npix
        self._oversample = oversample
        self.wavelength = wavelength
        self.telescope_radius = telescope_radius
        self.name = name
        self.lambda_over_d = (
            self.wavelength / (2 * self.telescope_radius)).to(
                u.arcsec, equivalencies=u.dimensionless_angles())    
        self.pixelsize = 2 * self.telescope_radius / self._npix
        self._initialize_optical_system()
        self._reset_intermediate_wfs()

    def get_snapshot(self, prefix='EFP'):
        snapshot = {}
        snapshot[EfpSnapshotEntry.NAME] = self.name
        snapshot[EfpSnapshotEntry.WAVELENGTH] = self.wavelength.to_value(u.nm)
        snapshot[EfpSnapshotEntry.NPIX] = self._npix
        snapshot[EfpSnapshotEntry.TELESCOPE_RADIUS] = \
            self.telescope_radius.to_value(u.m)
        return Snapshotable.prepend(prefix, snapshot)

    @property
    def optical_system(self):
        return self._osys

    def _initialize_optical_system(self):
        self._log = logging.getLogger('BaseSystemForPetalometry-%s' % self.name)
        self._osys = poppy.OpticalSystem(oversample=self._oversample, npix=self._npix,
                                            pupil_diameter=self.telescope_radius*2)
        self._osys.add_pupil(poppy.CircularAperture(radius=self.telescope_radius, name=self.PLANE_EXIT_PUPIL))
        self._osys.add_detector(
            pixelscale=0.25 * self.lambda_over_d / (1 * u.pixel), fov_arcsec=1)
            
        self._planes_idx_dict = {x.name: i for i, x in enumerate(self._osys.planes)}
        self.display_intermediates = False
    
    def _reset_intermediate_wfs(self):
        self._intermediates_wfs = None
        self._psf = None

    def propagate(self):
        self._log.info('propagating')
        _, self._intermediates_wfs = self._osys.propagate(
            self._osys.input_wavefront(self.wavelength),
            normalize='first',
            display_intermediates=self.display_intermediates,
            return_intermediates=True)
        
    def set_phase_shift(self, shift_in_lambda):
        pass

    def _compute_psf(self):
        self._psf, self._intermediates_wfs = self.optical_system.calc_psf(
            self.wavelength,
            return_intermediates=True,
            display_intermediates=self.display_intermediates,
            normalize='first')
            
    def psf(self):
        if not self._psf:
            self._compute_psf()
        return self._psf

    def display_psf(self, **kwargs):
        poppy.display_psf(self.psf(), **kwargs)

    def _wave_at_plane(self, plane_no):
        if not self._intermediates_wfs:
            self.propagate()
        return self._intermediates_wfs[plane_no]
    
    def pupil_mask(self):
        return mask_from_median(self.pupil_intensity(), 10)

    def pupil_phase(self):
        return self.pupil_wavefront().phase

    def pupil_amplitude(self):
        return self.pupil_wavefront().amplitude

    def pupil_intensity(self):
        return self.pupil_wavefront().intensity

    def pupil_wavefront(self):
        return self._wave_at_plane(self._planes_idx_dict[self.PLANE_EXIT_PUPIL])
    
    def pupil_opd(self):
        pass
    
    def _wave_at_plane(self, plane_no):
        if not self._intermediates_wfs:
            self.propagate()
        return self._intermediates_wfs[plane_no]
    
    def _display_on_plane(self, what, plane_number, scale='linear'):
        wave = self._wave_at_plane(plane_number)
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
            norm = mpl.colors.Normalize()
        elif scale == 'log':
            image = self._pump_up_zero_for_log_display(image)
            vmax = np.max(image)
            vmin = np.maximum(np.min(image), np.max(image) / 1e4)
            norm = mpl.colors.LogNorm(vmin=vmin, vmax=vmax)
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
    
    def display_phase_on_plane(self, plane_number):
        self._display_on_plane('phase', plane_number, scale='linear')
    
    def display_pupil_opd(self, title='Total OPD', **kw):
        pass

    def display_pupil_phase(self, **kw):
        self.display_phase_on_plane(
            self._planes_idx_dict[self.PLANE_EXIT_PUPIL], **kw)
        
    def plot_pupil_intensity_row(self, row=None, scale='linear'):
        aa = self._wave(self._planes_idx_dict[self.PLANE_EXIT_PUPIL])
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


class EltForPetalometry(BaseSystemForPetalometry):
    '''
    Parameters
    ----------
    r0: float
        r0 of Kolmogorov turbulence. Set it to np.inf to disable. Default=np.inf
        Every temporal step generates a new Kolmogorov screen - no phase
        screen wind propagation.

    tracking_number: str
        AO residual wfe to use. Set it to None to disable. Default=None.

    zern_coeff: tuple or None
        Zernike coefficients of the static WFE to add to the pupil. Unit
        in meters, starting from piston. Set it to None to disable.
        Default=None.

    lwe_wind_speed: float or None
        Wind speed of low wind effect simulations in m/s.
        Possible values: (None, 0.5, 1.0). Set it to None to disable.
        Default=None.

    rotation_angle: float
        Rotation angle in degree of the petalometer. Default=0.

    kolm_seed: int
        Seed of random number generator for Kolmogorov turbulence. Default=0.

    residual_wavefront_start_from: int
        Index of AO residual wfe frame to start from.
        Used to skip the convergence. Default=100.
    '''

    PLANE_TURBULENCE = 'Turbulence'
    PLANE_AO_RESIDUAL = 'AO residual'
    PLANE_LOW_WIND_EFFECT = 'LWE'
    PLANE_ZERNIKE = 'Zernike'
    PLANE_M4_PETALS = 'M4 petals'
    PLANE_ELT_APERTURE = 'ELT aperture'
    PLANE_PHASE_SHIFT = 'phase shift'
    # PLANE_EXIT_PUPIL = 'exit pupil'

    def __init__(self,
                 npix=256,
                 oversample=2,
                 wavelength=2.2e-6 * u.m,
                 telescope_radius=19.8 * u.m,
                 name='',
                 r0=np.inf,
                 tracking_number=None,
                 zern_coeff=None,
                 lwe_speed=None,
                 rotation_angle=0,
                 kolm_seed=0,
                 residual_wavefront_start_from=100,
                 residual_wavefront_step=0,
                 residual_wavefront_average_on=1):
        
        self._r0 = r0
        self._tracknum = tracking_number
        self.zernike_coefficients = zern_coeff
        self._lwe_wind_speed = lwe_speed
        self.pupil_rotation_angle = rotation_angle
        self._kolm_seed = kolm_seed
        self._residual_wavefront_start_from = residual_wavefront_start_from
        self._residual_wavefront_step = residual_wavefront_step
        self._residual_wavefront_average_on = residual_wavefront_average_on

        # self._npix = npix
        # self.oversample = oversample
        # self.wavelength = wavelength
        # self.telescope_radius = telescope_radius

        BaseSystemForPetalometry.__init__(self, npix, oversample, wavelength, telescope_radius, name)

    def _initialize_optical_system(self): 
        self._log = logging.getLogger('EltForPetalometry-%s' % self.name)
        self._osys = poppy.OpticalSystem(
            oversample=self._oversample,
            npix=self._npix,
            pupil_diameter=2 * self.telescope_radius)

        if self._r0 != np.inf:
            r0l = self._r0 * u.m * \
                (self.wavelength / (0.5e-6 * u.m)) ** (6 / 5)
            kolmo_wfe = poppy.KolmogorovWFE(
                name=self.PLANE_TURBULENCE,
                r0=r0l,
                dz=1 * u.m,
                seed=self._kolm_seed)
            self._osys.add_pupil(kolmo_wfe)

        if self._tracknum is not None:
            self._aores_wfe = MaoryResidualWavefront(
                self._tracknum,
                start_from=self._residual_wavefront_start_from,
                step=self._residual_wavefront_step,
                average_on=self._residual_wavefront_average_on,
                name=self.PLANE_AO_RESIDUAL)
            self._osys.add_pupil(self._aores_wfe)

        if self._lwe_wind_speed is not None:
            self._osys.add_pupil(LowWindEffectWavefront(
                self._lwe_wind_speed, name=self.PLANE_LOW_WIND_EFFECT))

        if self.zernike_coefficients is not None:
            self._osys.add_pupil(poppy.ZernikeWFE(name=self.PLANE_ZERNIKE,
                                                  coefficients=self.zernike_coefficients,
                                                  radius=self.telescope_radius))
        self._osys.add_pupil(PetaledM4(name=self.PLANE_M4_PETALS))

        self._osys.add_pupil(ELTAperture(
            pupil_mask_tag=elt_aperture.PUPIL_MASK_DEFAULT, name=self.PLANE_ELT_APERTURE))

        self._osys.add_pupil(ScalarOpticalPathDifference(
            opd=0 * u.nm, planetype=PlaneType.pupil, name=self.PLANE_PHASE_SHIFT))

        if self.pupil_rotation_angle != 0:
            self._osys.add_rotation(-1 * self.pupil_rotation_angle)

        self._osys.add_pupil(poppy.CircularAperture(radius=self.telescope_radius,
                                                    name=self.PLANE_EXIT_PUPIL))
        self._osys.add_detector(
            pixelscale=0.5 * self.lambda_over_d / (1 * u.pixel),
            fov_arcsec=1)

        self._planes_idx_dict = {x.name: i for i,
                                 x in enumerate(self._osys.planes)}

        self.display_intermediates = False
        self._reset_intermediate_wfs()

    # @property
    # def optical_system(self):
    #     return self._osys

    def get_snapshot(self, prefix='EFP'):
        snapshot = {}
        snapshot[EfpSnapshotEntry.NAME] = self.name
        snapshot[EfpSnapshotEntry.WAVELENGTH] = self.wavelength.to_value(u.nm)
        snapshot[EfpSnapshotEntry.NPIX] = self._npix
        snapshot[EfpSnapshotEntry.TELESCOPE_RADIUS] = \
            self.telescope_radius.to_value(u.m)
        snapshot[EfpSnapshotEntry.KOLMOGOROV_SEED] = self._kolm_seed
        snapshot[EfpSnapshotEntry.PUPIL_ROTATION_ANGLE] = \
            self.pupil_rotation_angle
        snapshot[EfpSnapshotEntry.LWE_WIND_SPEED] = self._lwe_wind_speed

        if self.zernike_coefficients is not None:
            snapshot[EfpSnapshotEntry.ZERNIKE_COEFFICIENTS] = np.array2string(
                self.zernike_coefficients)
        else:
            snapshot[EfpSnapshotEntry.ZERNIKE_COEFFICIENTS] = None

        snapshot[EfpSnapshotEntry.R0] = self._r0
        snapshot[EfpSnapshotEntry.PASSATA_TRACKING_NUMBER] = self._tracknum
        if self._tracknum is not None:
            snapshot.update(
                self._aores_wfe.get_snapshot(SnapshotPrefix.PASSATA_RESIDUAL))
        return Snapshotable.prepend(prefix, snapshot)

    # def _reset_intermediate_wfs(self):
    #     self._intermediates_wfs = None
    #     self._psf = None

    # def set_atmospheric_wavefront(self, atmo_opd):
    #     self._reset_intermediate_wfs()
    #     pass

    def set_input_wavefront_zernike(self, zern_coeff):
        '''
        Set Zernike amplitudes
        
        If this object has been created with zern_coeff=None, this
        method will raise an exception
        
        An aberration corresponding to the standard Zernike polynomials
        with the specified amplitude and radius equal to telescope_radius
        is added to the system.
        The first element correspond to the piston. Ordered as in Noll '76 
        
        Parameters
        ----------
        zern_coeff: astropy.quantity equivalent to u.m of shape (N,)
            Zernike amplitudes

        '''
        self._reset_intermediate_wfs()
        in_wfe = poppy.ZernikeWFE(name=self.PLANE_ZERNIKE,
                                  coefficients=zern_coeff,
                                  radius=self.telescope_radius)
        self.optical_system.planes[self._planes_idx_dict[self.PLANE_ZERNIKE]] = in_wfe
        self.zernike_coefficients = zern_coeff.to_value(u.m)

    def set_m4_petals(self, petals):
        '''
        Set M4 petals

        Parameters
        ----------
        petals: astropy.quantity equivalent to u.m of shape (6,)
            petals to be applied on M4
        '''

        self._reset_intermediate_wfs()
        in_wfe = PetaledM4(petals, name=self.PLANE_M4_PETALS)
        self.optical_system.planes[self._planes_idx_dict[self.PLANE_M4_PETALS]] = in_wfe

    def set_phase_shift(self, shift_in_lambda):
        self._reset_intermediate_wfs()
        in_wfe = ScalarOpticalPathDifference(
            opd=shift_in_lambda * self.wavelength,
            planetype=PlaneType.pupil,
            name=self.PLANE_PHASE_SHIFT)
        self.optical_system.planes[self._planes_idx_dict[self.PLANE_PHASE_SHIFT]] = in_wfe

    def set_step_idx(self, step_idx):
        # advance residual phase screen
        self._reset_intermediate_wfs()
        self.optical_system.planes[self._planes_idx_dict[self.PLANE_AO_RESIDUAL]].set_step_idx(
            step_idx)

    def set_kolm_seed(self, seed):
        self._reset_intermediate_wfs()
        self.optical_system.planes[self._planes_idx_dict[self.PLANE_TURBULENCE]].seed = seed
        self._kolm_seed = seed

    # def propagate(self):
    #     self._log.info('propagating')
    #     _, self._intermediates_wfs = self.optical_system.propagate(
    #         self.optical_system.input_wavefront(self.wavelength),
    #         normalize='first',
    #         display_intermediates=self.display_intermediates,
    #         return_intermediates=True)

    # def compute_psf(self):
    #     self._psf, self._intermediates_wfs = self.optical_system.calc_psf(
    #         self.wavelength,
    #         return_intermediates=True,
    #         display_intermediates=self.display_intermediates,
    #         normalize='first')

    # def psf(self):
    #     if not self._intermediates_wfs:
    #         self.compute_psf()
    #     return self._psf

    # def display_psf(self, **kwargs):
    #     poppy.display_psf(self.psf(), **kwargs)

    # def _pump_up_zero_for_log_display(self, image):
    #     ret = image * 1.0
    #     ret[np.where(image == 0)] = \
    #         np.min(image[np.where(image != 0)]) / 10
    #     return ret

    # def _wave(self, plane_no):
    #     if not self._intermediates_wfs:
    #         self.propagate()
    #     return self._intermediates_wfs[plane_no]

    # def pupil_wavefront(self):
    #     return self._wave(self._planes_idx_dict[self.PLANE_EXIT_PUPIL])

    # def pupil_phase(self):
    #     return self.pupil_wavefront().phase

    # def pupil_amplitude(self):
    #     return self.pupil_wavefront().amplitude

    # def pupil_intensity(self):
    #     return self.pupil_wavefront().intensity

    def pupil_opd_unwrapped(self):
        mask = self.pupil_mask()
        minv = np.logical_not(mask).astype(int)
        unwrap = skimage.restoration.unwrap_phase(
            self.pupil_phase() * minv)
        return np.ma.array(unwrap / 2 / np.pi * self.wavelength.to_value(u.nm),
                           mask=mask)

    # def pupil_mask(self):
    #     return mask_from_median(self.pupil_intensity(), 10)

    def pupil_opd(self):
        if not self._intermediates_wfs:
            self.propagate()
        osys = self.optical_system
        wave = osys.input_wavefront(self.wavelength)
        opd = 0

        def _trick_to_get_resampled_opd(plane):
            _ = plane.get_phasor(wave)
            return plane._resampled_opd

        def _try_add_opd(plane_name, wave, use_trick):
            try:
                if use_trick:
                    return _trick_to_get_resampled_opd(
                        osys.planes[self._planes_idx_dict[plane_name]])
                else:
                    return osys.planes[self._planes_idx_dict[plane_name]
                                       ].get_opd(wave)
            except KeyError:
                return 0

        opd += _try_add_opd(self.PLANE_TURBULENCE, wave, False)
        opd += _try_add_opd(self.PLANE_AO_RESIDUAL, wave, True)
        opd += _try_add_opd(self.PLANE_LOW_WIND_EFFECT, wave, True)
        opd += _try_add_opd(self.PLANE_ZERNIKE, wave, False)
        opd += _try_add_opd(self.PLANE_M4_PETALS, wave, False)
        opd += _try_add_opd(self.PLANE_PHASE_SHIFT, wave, False)

        opdm = np.ma.MaskedArray(opd, mask=self.pupil_mask())
        return opdm * 1e9

    # def _display_on_plane(self, what, plane_number, scale='linear'):
    #     wave = self._wave(plane_number)
    #     if what == 'intensity':
    #         image = wave.intensity
    #         cmap = 'cividis'
    #     elif what == 'phase':
    #         image = wave.phase
    #         cmap = 'twilight'
    #     else:
    #         raise Exception('Unknown property to display: %s')
    #     title = wave.location

    #     if scale == 'linear':
    #         # norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
    #         norm = matplotlib.colors.Normalize()
    #     elif scale == 'log':
    #         image = self._pump_up_zero_for_log_display(image)
    #         vmax = np.max(image)
    #         vmin = np.maximum(np.min(image), np.max(image) / 1e4)
    #         norm = matplotlib.colors.LogNorm(vmin=vmin, vmax=vmax)
    #     else:
    #         raise Exception('Unknown scale %s' % scale)

    #     if wave.planetype == PlaneType.pupil:
    #         pc = wave.pupil_coordinates(image.shape, wave.pixelscale)
    #         extent = [pc[0].min(), pc[0].max(), pc[1].min(), pc[1].max()]
    #     elif wave.planetype == PlaneType.image:
    #         extent = [-1, 1, -1, 1]

    #     plt.clf()
    #     plt.imshow(image, norm=norm, extent=extent, origin='lower', cmap=cmap)
    #     plt.title(title)
    #     plt.colorbar()

    # def display_intensity_on_plane(self, plane_number, scale='linear'):
    #     self._display_on_plane('intensity', plane_number, scale=scale)

    # def display_phase_on_plane(self, plane_number):
    #     self._display_on_plane('phase', plane_number, scale='linear')

    # def display_pupil_intensity(self, **kw):
    #     self.display_intensity_on_plane(
    #         self._planes_idx_dict[self.PLANE_EXIT_PUPIL], **kw)

    # def display_pupil_phase(self, **kw):
    #     self.display_phase_on_plane(
    #         self._planes_idx_dict[self.PLANE_EXIT_PUPIL], **kw)

    def display_pupil_opd(self, title='Total OPD', **kw):
        wave = self._wave_at_plane(0)
        image = self.pupil_opd()
        norm = mpl.colors.Normalize()
        cmap = 'cividis'
        pc = wave.pupil_coordinates(image.shape, wave.pixelscale)
        extent = [pc[0].min(), pc[0].max(), pc[1].min(), pc[1].max()]

        plt.clf()
        plt.imshow(image, norm=norm, extent=extent, origin='lower', cmap=cmap)
        plt.title(title)
        plt.colorbar()


class LabSystemForCiaoCiao(BaseSystemForPetalometry):
    '''
    Parameters
    ----------
    rotation_angle: float
        Rotation angle in degree of the petalometer. Default=0.

    zernike_coeff: tuple or None
        Zernike coefficients of the static WFE to add to the pupil. Unit
        in meters, starting from piston. Set it to None to disable.
        Default=None.

    segmented_pupil: TBD

    '''

    PLANE_ZERNIKE = 'Zernike'
    PLANE_PETALS = 'Petals'
    PLANE_TELESCOPE_APERTURE = 'Telescope aperture'

    def __init__(self,
                 npix=256,
                 oversample=1,
                 wavelength=2.2*u.um,
                 telescope_radius=4*u.m,
                 name='',
                 rotation_angle=0,
                 zernike_coeff=[0, 0]*u.nm,
                 segmented_pupil=None):
        self.pupil_rotation_angle = rotation_angle
        self.zernike_coefficients = zernike_coeff
        self._segmented_pupil = segmented_pupil

        BaseSystemForPetalometry.__init__(self, npix, oversample, wavelength, telescope_radius, name)

    def get_snapshot(self, prefix='EFP'):
        snapshot = {}
        snapshot[EfpSnapshotEntry.NAME] = self.name
        snapshot[EfpSnapshotEntry.WAVELENGTH] = self.wavelength.to_value(u.nm)
        snapshot[EfpSnapshotEntry.NPIX] = self._npix
        snapshot[EfpSnapshotEntry.TELESCOPE_RADIUS] = \
            self.telescope_radius.to_value(u.m)
        snapshot[EfpSnapshotEntry.PUPIL_ROTATION_ANGLE] = \
            self.pupil_rotation_angle

        if self.zernike_coefficients is not None:
            snapshot[EfpSnapshotEntry.ZERNIKE_COEFFICIENTS] = np.array2string(
                self.zernike_coefficients)
        else:
            snapshot[EfpSnapshotEntry.ZERNIKE_COEFFICIENTS] = None
        return Snapshotable.prepend(prefix, snapshot)

    def _initialize_optical_system(self):
        self._log = logging.getLogger('LabSystemForCiaoCiao-%s' % self.name)
        self._osys = poppy.OpticalSystem(oversample=self._oversample, npix=self._npix,
                                            pupil_diameter=self.telescope_radius*2)
        if self.zernike_coefficients is not None:
            self._osys.add_pupil(poppy.ZernikeWFE(name=self.PLANE_ZERNIKE,
                                                  coefficients=self.zernike_coefficients,
                                                  radius=self.telescope_radius))
        if self._segmented_pupil:
            self._osys.add_pupil(self._segmented_pupil(name=self.PLANE_PETALS))

        if self.pupil_rotation_angle != 0:
            self._osys.add_rotation(-1 * self.pupil_rotation_angle)
        self._osys.add_pupil(poppy.CircularAperture(radius=self.telescope_radius, name=self.PLANE_EXIT_PUPIL))
        self._osys.add_detector(pixelscale=0.25*(self.wavelength / (self.telescope_radius*2)).to(
            u.arcsec, equivalencies=u.dimensionless_angles()) /(1*u.pixel), fov_arcsec=1)
    
        self._planes_idx_dict = {x.name: i for i, x in enumerate(self._osys.planes)}
        self.display_intermediates = False

    def pupil_opd(self):
            if not self._intermediates_wfs:
                self.propagate()
            osys = self.optical_system
            wave = osys.input_wavefront(self.wavelength)
            opd = 0

            def _trick_to_get_resampled_opd(plane):
                _ = plane.get_phasor(wave)
                return plane._resampled_opd

            def _try_add_opd(plane_name, wave, use_trick):
                try:
                    if use_trick:
                        return _trick_to_get_resampled_opd(
                            osys.planes[self._planes_idx_dict[plane_name]])
                    else:
                        return osys.planes[self._planes_idx_dict[plane_name]
                                        ].get_opd(wave)
                except KeyError:
                    return 0

            opd += _try_add_opd(self.PLANE_ZERNIKE, wave, False)
            opd += _try_add_opd(self.PLANE_PETALS, wave, False)

            opdm = np.ma.MaskedArray(opd, mask=self.pupil_mask())
            return opdm * 1e9

    def display_pupil_opd(self, title='Total OPD', **kw):
        wave = self._wave_at_plane(0)
        image = self.pupil_opd()
        norm = mpl.colors.Normalize()
        cmap = 'cividis'
        pc = wave.pupil_coordinates(image.shape, wave.pixelscale)
        extent = [pc[0].min(), pc[0].max(), pc[1].min(), pc[1].max()]

        plt.clf()
        plt.imshow(image, norm=norm, extent=extent, origin='lower', cmap=cmap)
        plt.title(title)
        plt.colorbar()