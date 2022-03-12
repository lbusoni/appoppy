
import numpy as np
import poppy
from poppy.optics import AnalyticImagePlaneElement
from poppy.poppy_core import Wavefront, PlaneType
from astropy import units as u


class PointDiffractionInterferometer(object):

    def __init__(self,
                 name='Point Diffraction Interferometer',
                 pinhole_radius=0.010 * u.arcsec,
                 transmittance_outer_region=0.01):
        self.name = name
        self.pinhole_radius = pinhole_radius
        self.transmittance_outer_region = transmittance_outer_region

    def add_to_system(self, osys, index):
        osys.add_pupil(poppy.FQPM_FFT_aligner(), index=index)
        osys.add_image(index=index + 1)
        osys.add_image(PointDiffractionPinHole(
            name=self.name,
            transmittance_outer=self.transmittance_outer_region,
            radius_outer=self.pinhole_radius),
            index=index + 2)
        osys.add_pupil(poppy.FQPM_FFT_aligner(direction='backward'),
                       index=index + 3)


class PointDiffractionPinHole(AnalyticImagePlaneElement):
    """ Defines a PDI field stop with an opaque outer region of a given radius and transmittance

    Parameters
    ------------
    name : string
        Descriptive name
    transmittance_outer : float
        Transmittance of the outside region. Default is 0.01 (best for PDI according to Smartt 1975)
    radius_outer : float
        Radius of the circular field stop outer edge in arcsec. Default is 1.0.
    """

    def __init__(self,
                 name="unnamed pinhole field stop",
                 transmittance_outer=0.01,
                 radius_outer=1.0 * u.arcsec,
                 **kwargs):
        AnalyticImagePlaneElement.__init__(self, **kwargs)
        self.name = name
        self.transmittance_outer = transmittance_outer
        self.radius_outer = radius_outer  # radius of circular field stop in arcseconds.
        self._default_display_size = 10 * u.arcsec  # radius_outer

    def get_transmission(self, wave):
        """ Compute the transmission inside/outside of the field stop.
        """
        if not isinstance(wave, Wavefront):  # pragma: no cover
            raise ValueError("get_transmission must be called with a Wavefront to define the spacing")
        assert (wave.planetype == PlaneType.image)

        y, x = self.get_coordinates(wave)
        r = np.sqrt(x ** 2 + y ** 2)

        self.transmission = np.ones(wave.shape, dtype=np.float)
        w_outside = np.where(r >= self.radius_outer.to(u.arcsec).value)
        self.transmission[w_outside] = self.transmittance_outer

        return self.transmission

