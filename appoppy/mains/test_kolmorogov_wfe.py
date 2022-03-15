import poppy
from astropy import units as u
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits


def test_kolmo(niter=3, r0_at_500=0.2, dz=1, wl=0.5e-6):
    telescope_radius = 19.5 * u.m
    r0 = r0_at_500 * (wl / 0.5e-6)**(6 / 5)
    for i in range(niter):
        ss = poppy.OpticalSystem(pupil_diameter=2 * telescope_radius)
        ss.add_pupil(poppy.KolmogorovWFE(
            r0=r0, dz=dz, outer_scale=40, inner_scale=0.05, kind='von Karman'))
        ss.add_pupil(poppy.CircularAperture(radius=telescope_radius,
                                            name='Entrance Pupil'))

        ss.add_detector(pixelscale=0.20, fov_arcsec=2.0)
        hdu = ss.calc_psf(wavelength=wl)[0]
        if i == 0:
            psfs = np.zeros((niter, hdu.shape[0], hdu.shape[1]))
        psfs[i] = hdu.data
    hdu.data = psfs.sum(axis=0)
    hdulist = fits.HDUList(hdu)
    print("0.98*l/r0 = %g  - FWHM %g " % (
        0.98 * wl / r0 / 4.848e-6, poppy.utils.measure_fwhm_radprof(hdulist)))
    plt.figure()
    poppy.utils.display_profiles(hdulist)
    plt.figure()
    poppy.utils.display_psf(hdulist)
    return psfs, hdulist, ss
