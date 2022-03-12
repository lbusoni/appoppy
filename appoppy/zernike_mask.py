import numpy as np
import poppy
from poppy.optics import AnalyticImagePlaneElement
from poppy.poppy_core import Wavefront, PlaneType, BaseWavefront
from astropy import units as u


class ZernikeMaskWFS(object):

    def __init__(self,
                 name='Zernike Mask WFS',
                 radius_in_arcsec=0.010,
                 phase_delay=np.pi / 2):
        self.name = name
        self._radius_in_arcsec = radius_in_arcsec
        self._phase_delay = phase_delay

    def add_to_system(self, osys, index):
        osys.add_pupil(poppy.FQPM_FFT_aligner(), index=index)
        osys.add_image(index=index + 1)
        osys.add_image(ZernikeMask(
            name=self.name,
            phase_delay=self._phase_delay,
            radius=self._radius_in_arcsec),
            index=index + 2)
        osys.add_pupil(poppy.FQPM_FFT_aligner(direction='backward'),
                       index=index + 3)


class ZernikeMask(AnalyticImagePlaneElement):
    """ Defines a Zernike Mask field stop with an inner region of a given _radius and phase delay

    Parameters
    ------------
    name : string
        Descriptive name
    phase_delay : float, default=pi/2
        phase delay of the inner region in radians
    radius : float, default=1.0
        Radius of the phase delayed region in arcsec
    """

    def __init__(self,
                 name="unnamed pinhole field stop",
                 phase_delay=np.pi / 2,
                 radius=1.0,
                 **kwargs):
        AnalyticImagePlaneElement.__init__(self, **kwargs)
        self.name = name
        self._phase_delay = phase_delay
        self._radius = radius
        self._default_display_size = 10 * u.arcsec  # radius_outer

    def get_opd(self, wave):
        """ Compute the OPD [m] appropriate for a Zernike Mask
        """

        if not isinstance(wave, Wavefront):  # pragma: no cover
            raise ValueError("get_opd must be called with "
                             "a Wavefront to define the spacing")
        assert (wave.planetype == PlaneType.image)

        print(wave)

        if isinstance(wave, BaseWavefront):
            wavelength = wave.wavelength
        else:
            wavelength = wave
        radians2meter = wavelength.to(u.meter).value / (2. * np.pi)

        y, x = self.get_coordinates(wave)
        r = np.sqrt(x ** 2 + y ** 2)

        opd = np.zeros(wave.shape, dtype=np.float)
        w_inside = np.where(r <= self._radius)
        opd[w_inside] = self._phase_delay * radians2meter

        return opd

