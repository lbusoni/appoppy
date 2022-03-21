import matplotlib.pyplot as plt
import astropy.units as u
import numpy as np
from appoppy.petalometer import Petalometer
from appoppy.package_data import data_root_dir
import os
import scipy.io
import poppy
from appoppy.elt_aperture import restore_elt_pupil_mask, ELTAperture


def main(r0=np.inf,
         petals=np.random.uniform(-1100, 1100, 6) * u.nm,
         rotation_angle=10):
    p = Petalometer(r0=r0, petals=petals, rotation_angle=rotation_angle)
    return p


def error(r0=np.inf,
          petals=np.random.uniform(-1100, 1100, 6) * u.nm,
          rotation_angle=10):
    res = []
    for i in range(100):
        p = Petalometer(r0=r0, petals=petals, rotation_angle=rotation_angle)
        res.append(p.error_petals())
        print(p.error_petals())
    ret = np.array(res)
    print(ret.std())
    return ret


def apply_petals(p, petals=np.random.uniform(-1000, 1000, 6) * u.nm):
    p.set_m4_petals(petals)
    p.sense_wavefront_jumps()


def restore_residual_wavefront():

    fname = os.path.join(data_root_dir(),
                         '20210518_223459.0',
                         'CUBE_OLCUBE_CL_coo0.0_0.0.sav')
    idl_dict = scipy.io.readsav(fname)
    phase_screen = np.moveaxis(idl_dict['cube_k'], 2, 0)
    mask = restore_elt_pupil_mask().data
    cmask = np.tile(mask, (phase_screen.shape[0], 1, 1))
    return np.ma.masked_array(phase_screen, mask=cmask)


def test_fits_optical_element():
    telescope_radius = 20
    osys = poppy.OpticalSystem(oversample=2,
                               pupil_diameter=2 * telescope_radius)
    osys.add_pupil(ELTAperture())
    osys.add_detector(0.001 * u.arcsec / u.pixel, fov_arcsec=0.4 * u.arcsec)
    osys.calc_psf(1 * u.um, display=True, display_intermediates=True)
    return osys


def reload():
    import importlib
    from appoppy import elt_for_petalometry
    from appoppy import phase_shift_interferometer
    from appoppy import petalometer
    from appoppy.mains import main220307_shear_interferometer
    importlib.reload(elt_for_petalometry)
    importlib.reload(phase_shift_interferometer)
    importlib.reload(petalometer)
    importlib.reload(main220307_shear_interferometer)


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
