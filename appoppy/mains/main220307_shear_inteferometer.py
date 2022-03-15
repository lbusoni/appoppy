import matplotlib.pyplot as plt
import astropy.units as u
import numpy as np
from appoppy.phase_shift_interferometer import PhaseShiftInterferometer
from appoppy.elt_for_petalometry import EltForPetalometry
import functools


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


class Petalometer():

    def __init__(self,
                 r0=np.inf,
                 petals=np.array([800, 0, 0, 0, 0, 0]) * u.nm,
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

    def set_zernike_wavefront(self, zernike_coefficients):
        self._model1.set_input_wavefront_zernike(zernike_coefficients)
        self._model2.set_input_wavefront_zernike(zernike_coefficients)

    def set_m4_petals(self, petals):
        self._model1.set_m4_petals(petals)
        self._model2.set_m4_petals(petals)
        self._petals = petals

    def sense_wavefront_jumps(self):
        self._i4.acquire()
        self._i4.display_interferogram()
        self._compute_jumps()
        return self.error()

    def _expected_jumps(self):
        dd = np.repeat(self._petals, 2).to_value(u.nm)
        return np.roll(dd, 1) - dd

    def error(self):
        return (self.estimated_petals() - self._zero_mean(self._petals)).to(u.nm)

    def _compute_jumps(self):
        r = self._model1.pupil_rotation_angle
        image = self._i4.interferogram()
        angs = (-180, -180 + r, -120, -120 + r, -60, -
                60 + r, 0, r, 60, 60 + r, 120, 120 + r, 180)
        res = np.zeros(len(angs) - 1)
        for i in range(len(angs) - 1):
            ifm = self._mask_ifgram(image, (angs[i], angs[i + 1]))
            res[i] = np.ma.median(ifm)

        self._jumps = self._around_zero_angle(
            res * u.nm, self._model1.wavelength)

    def _mask_ifgram(self, ifgram, angle_range):
        smask1 = sector_mask(ifgram.shape,
                             (angle_range[0], angle_range[1]))
        mask = np.ma.mask_or(ifgram.mask, ~smask1)
        return np.ma.masked_array(ifgram, mask=mask)

    @staticmethod
    def _zero_mean(image):
        return image - image.mean()

    @staticmethod
    def _around_zero_angle(data, wavelength):
        dpi = 2 * np.pi
        a = data.to_value(u.nm) / wavelength.to_value(u.nm) * dpi
        return np.arctan2(np.sin(a), np.cos(a)) * wavelength.to(u.nm) / dpi

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

    def estimated_petals(self):
        return np.dot(self._jumps_to_petals_matrix(),
                      np.append(self._jumps[::2], 0))


def main(r0=np.inf,
         petals=np.random.uniform(-1000, 1000, 6) * u.nm,
         rotation_angle=10):
    p = Petalometer(r0=r0, petals=petals, rotation_angle=rotation_angle)
    return p


def apply_petals(p, petals=np.random.uniform(-1000, 1000, 6) * u.nm):
    p.set_m4_petals(petals)
    p.sense_wavefront_jumps()


# def test_turb_variance(niter=10, r0=1, rotation_angle=10):
#     res = []
#     for _ in range(niter):
#         pet = main(r0=r0,
#                    rotation_angle=rotation_angle,
#                    petals=np.array([0, 0, 0, 0, 0, 0]) * u.nm)
#         res.append(pet._i4.interferogram())
#     dd = np.ma.array(res)
#     plt.clf()
#     plt.imshow(dd.std(axis=0), origin='lower', cmap='twilight')
#     plt.colorbar()
#     return dd

# def test_elt_pupil():
#     telescope_radius = 19.5 * u.m
#     wl = 1e-6 * u.m
#     ss = poppy.OpticalSystem(pupil_diameter=2 * telescope_radius)
#
#     # 3 rings of 2 m segments yields 14.1 m circumscribed diameter
#     ap = poppy.MultiHexagonAperture(rings=14, flattoflat=2 * u.m)
#     sec = poppy.SecondaryObscuration(
#         secondary_radius=1.5, n_supports=4, support_width=0.1)   # secondary with spiders
#     atlast = poppy.CompoundAnalyticOptic(
#         opticslist=[ap, sec], name='Mock ATLAST')           # combine into one optic
#
#     ss.add_pupil(atlast)
#
#     ss.add_detector(pixelscale=0.20, fov_arcsec=2.0)
#     psf = ss.calc_psf(wavelength=wl)
#     poppy.display_psf(psf)
#     return ss
