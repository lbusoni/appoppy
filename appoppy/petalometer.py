import astropy.units as u
import numpy as np
from appoppy.phase_shift_interferometer import PhaseShiftInterferometer
from appoppy.elt_for_petalometry import EltForPetalometry
from appoppy.sector_mask import sector_mask
from appoppy.circular_math import wrap_around_zero, zero_mean, difference
from appoppy import circular_math


class Petalometer():

    def __init__(self,
                 r0=np.inf,
                 petals=np.array([0, 0, 0, 0, 0, 0]) * u.nm,
                 rotation_angle=15,
                 zernike=[0, 0],
                 seed=None):
        if seed is None:
            seed = np.random.randint(2147483647)

        self._model1 = EltForPetalometry(
            r0=r0,
            seed=seed,
            rotation_angle=rotation_angle)

        self._model2 = EltForPetalometry(
            r0=r0,
            seed=seed)

        self.set_m4_petals(petals)
        self.set_zernike_wavefront(zernike)

        self._i4 = PhaseShiftInterferometer(self._model1,  self._model2)
        self._i4.combine()
        self.sense_wavefront_jumps()

    @property
    def wavelength(self):
        return self._model1.wavelength

    def set_zernike_wavefront(self, zernike_coefficients):
        self._model1.set_input_wavefront_zernike(zernike_coefficients)
        self._model2.set_input_wavefront_zernike(zernike_coefficients)

    def set_m4_petals(self, petals):
        self._petals = zero_mean(petals, self.wavelength)
        self._model1.set_m4_petals(self._petals)
        self._model2.set_m4_petals(self._petals)

    @property
    def petals(self):
        return self._petals.to(u.nm)

    def sense_wavefront_jumps(self):
        self._i4.acquire()
        self._i4.display_interferogram()
        self._compute_jumps()
        return self.error_jumps()

    def _expected_jumps(self):
        dd = np.repeat(self.petals, 2)
        return wrap_around_zero(np.roll(dd, 1) - dd,
                                self.wavelength)

    def error_jumps(self):
        return wrap_around_zero(
            (self._expected_jumps() - self._jumps).to(u.nm),
            self.wavelength)

    def error_petals(self):
        return difference(self.estimated_petals(), self.petals, self.wavelength)

    def _compute_jumps(self):
        r = self._model1.pupil_rotation_angle
        image = self._i4.interferogram()
        angs = (-180, -180 + r, -120, -120 + r, -60, -
                60 + r, 0, r, 60, 60 + r, 120, 120 + r, 180)
        res = np.zeros(len(angs) - 1)
        for i in range(len(angs) - 1):
            ifm = self._mask_ifgram(image, (angs[i], angs[i + 1]))
            res[i] = np.ma.median(ifm)

        self._jumps = wrap_around_zero(
            res * u.nm, self.wavelength)

    def _mask_ifgram(self, ifgram, angle_range):
        smask1 = sector_mask(ifgram.shape,
                             (angle_range[0], angle_range[1]))
        mask = np.ma.mask_or(ifgram.mask, ~smask1)
        return np.ma.masked_array(ifgram, mask=mask)

    def _jumps_to_petals_matrix(self):
        # last line forces global piston to zero
        mm = np.array([
            [-1, 0, 0, 0, 0, 1],
            [1, -1, 0, 0, 0, 0],
            [0, 1, -1, 0, 0, 0],
            [0, 0, 1, -1, 0, 0],
            [0, 0, 0, 1, -1, 0],
            [0, 0, 0, 0, 1, -1],
            [1, 1, 1, 1, 1, 1]
        ])
        return np.linalg.pinv(mm)

    def estimated_petals_wrong(self):
        return wrap_around_zero(
            np.dot(self._jumps_to_petals_matrix(),
                   np.append(self._jumps[::2], 0)),
            self.wavelength)

    def estimated_petals(self):
        res = -1 * np.cumsum(self._jumps[::2])
        return zero_mean(res, self.wavelength)
