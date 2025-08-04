import numpy as np
import poppy
from poppy.optics import AnalyticImagePlaneElement
from poppy.poppy_core import Wavefront, PlaneType
from astropy import units as u
import logging

class IdealCoronograph(object):
    """
    Ideal coronograph: introduces a 180° phase shift (multiplies amplitude by -1)
    everywhere in the focal plane except at the central pixel (0,0), where the amplitude
    is attenuated by a given factor (default=1).
    """

    def __init__(self, name='Ideal Coronograph', **kwargs):
        self.name = name
        self._logger = logging.getLogger(name)

    def add_to_system(self, osys, index):
        osys.add_image(
            IdealCoronographMask(name=self.name, **kwargs),
            index=index)


class IdealCoronographMask(AnalyticImagePlaneElement):
    """
    Ideal coronograph mask for the focal plane.
    """

    def __init__(self, name="Ideal Coronograph Mask", central_transmission=1.0, outer_transmission=1.0, radius=0.0, **kwargs):
        super().__init__(name=name, **kwargs)
        self.central_transmission = central_transmission
        self.outer_transmission = outer_transmission
        self.radius = radius  # raggio in coordinate focali (stesse unità di x, y)
        self._logger = logging.getLogger(name)

    def _central_area_coordinates(self, wave):
        # Seleziona tutti i pixel entro una certa distanza radius dal centro
        y, x = self.get_coordinates(wave)
        r = np.sqrt(x**2 + y**2)
        mask = r <= self.radius
        coords = np.where(mask)
        self._logger.debug(f"Central area mask: {coords} (radius={self.radius})")
        return coords

    def _central_pixel_coordinates(self, wave):
        # Get the coordinates of the central pixel in the wavefront
        y, x = self.get_coordinates(wave)
        idx_central = (np.abs(x) + np.abs(y)).argmin()
        cp = np.unravel_index(idx_central, x.shape)
        self._logger.debug(f"Central pixel coordinates: {cp} - corresponding to {x[cp]}, {y[cp]}")
        return cp

    def get_transmission(self, wave):
        # Attenua tutti i pixel entro radius dal centro
        transmission = np.ones(wave.shape, dtype=float) * self.outer_transmission
        central_coords = self._central_area_coordinates(wave)
        transmission[central_coords] = self.central_transmission
        return transmission

    def get_opd(self, wave):
        # OPD = λ/2 nell'area centrale, nullo altrove
        opd = np.zeros(wave.shape, dtype=float)
        central_coords = self._central_area_coordinates(wave)
        opd[central_coords] = (wave.wavelength.to_value(u.meter) / 2)
        return opd