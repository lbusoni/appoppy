import numpy as np
from poppy.wfe import WavefrontError
from astropy import units as u


class PetaledM4(WavefrontError):
    '''
    A simplified 6-petals deformable mirror (like E-ELT's M4) with
    piston control on each petal
    '''

    N_PETALS = 6

    def __init__(self, piston=None, name="M4", **kwargs):
        if piston is None:
            piston = np.zeros(self.N_PETALS) * u.nm
        self._piston = piston
        kwargs.update({'name': name})
        super(PetaledM4, self).__init__(**kwargs)

    def _mask_for_petal(self, x, y, petal_idx):
        return np.logical_and(
            np.arctan2(-x, -y) < (petal_idx - 2) * np.pi / 3,
            np.arctan2(-x, -y) > (petal_idx - 3) * np.pi / 3)

    def get_opd(self, wave):
        y, x = self.get_coordinates(wave)  # in meters
        opd = np.zeros(wave.shape, dtype=np.float64)
        for petal_idx in range(self.N_PETALS):
            mask = self._mask_for_petal(x, y, petal_idx)
            opd[mask] = self._piston[petal_idx].to(u.m).value
        return opd
