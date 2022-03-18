import matplotlib.pyplot as plt
import astropy.units as u
import numpy as np
from appoppy.petalometer import Petalometer


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
