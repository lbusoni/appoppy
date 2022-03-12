import numpy as np
import poppy
from poppy.optics import AnalyticImagePlaneElement
from poppy.poppy_core import Wavefront, PlaneType
from astropy import units as u


class Pyramid(AnalyticImagePlaneElement):

    def __init__(self, name, opd_per_arcsec, edge_thickness, **kwargs):
        AnalyticImagePlaneElement.__init__(self, **kwargs)
        self.name = name
        self.opdPerArcsec = opd_per_arcsec.to(u.m / u.arcsec).value
        self.edgeThicknessInArcsec = edge_thickness.to(u.arcsec).value

    def get_opd(self, wave):
        """ Compute the OPD [m] appropriate for a pyramid for some given pixel spacing
        corresponding to the supplied Wavefront
        """

        if not isinstance(wave, Wavefront):  # pragma: no cover
            raise ValueError("Prism get_opd must be called with "
                             "a Wavefront to define the spacing")
        assert (wave.planetype == PlaneType.image)

        y, x = self.get_coordinates(wave)
        phase = (np.abs(x) + np.abs(y)) * self.opdPerArcsec

        return phase

    def get_transmission(self, wave):
        if not isinstance(wave, Wavefront):  # pragma: no cover
            raise ValueError("Pyramid get_opd must be called with"
                             " a Wavefront to define the spacing")
        assert (wave.planetype == PlaneType.image)

        y, x = self.get_coordinates(wave)
        t = np.ones(x.shape)
        t[np.where(np.abs(x) < self.edgeThicknessInArcsec)] = 0
        t[np.where(np.abs(y) < self.edgeThicknessInArcsec)] = 0
        return t


class PyramidWFS(object):

    def __init__(self,
                 name='Pyramid WFS',
                 pyr_angle=2.5e-5 * u.m / u.arcsec,
                 pyr_edge_thickness=0.010 * u.arcsec):
        self.name = name
        self.pyr_angle = pyr_angle
        self.pyr_edge_in_arcsec = pyr_edge_thickness

    def add_to_system(self, osys, index):
        osys.add_pupil(poppy.FQPM_FFT_aligner(), index=index)
        osys.add_image(poppy.AnnularFieldStop(radius_outer=1,
                                              radius_inner=0,
                                              name='pyramid field stop'),
                       index=index + 1)
        osys.add_image(Pyramid(
            name=self.name,
            opd_per_arcsec=self.pyr_angle,
            edge_thickness=self.pyr_edge_in_arcsec),
            index=index + 2)

        pupil_plane = poppy.FQPM_FFT_aligner(direction='backward',
                                             name="Pyramid WFS pupil plane")
        pupil_plane.wavefront_display_hint = 'intensity'
        osys.add_pupil(
            pupil_plane,
            index=index + 3)

