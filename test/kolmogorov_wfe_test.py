#!/usr/bin/env python
import unittest
import numpy as np
from astropy.io import fits
from astropy import units as u
import poppy


class KolmogorovWFETest(unittest.TestCase):

    def test_fwhm_seeing_limited_image(self):
        niter = 10
        r0_at_500 = 0.2 * u.m
        dz = 1 * u.m
        wl = 0.5e-6 * u.m
        telescope_radius = 4 * u.m
        r0 = r0_at_500 * (wl / (0.5e-6 * u.m))**(6 / 5)

        wants_fwhm = (0.98 * wl / r0 / 4.848e-6).value
        for i in range(niter):
            ss = poppy.OpticalSystem(pupil_diameter=2 * telescope_radius)
            ss.add_pupil(poppy.KolmogorovWFE(
                r0=r0, dz=dz, kind='Kolmogorov'))
            ss.add_pupil(poppy.CircularAperture(radius=telescope_radius,
                                                name='Entrance Pupil'))

            ss.add_detector(pixelscale=0.20, fov_arcsec=2.0)
            hdu = ss.calc_psf(wavelength=wl)[0]
            if i == 0:
                psfs = np.zeros((niter, hdu.shape[0], hdu.shape[1]))
            psfs[i] = hdu.data
        hdu.data = psfs.sum(axis=0)
        hdulist = fits.HDUList(hdu)
        got_fwhm = poppy.utils.measure_fwhm_radprof(hdulist)
        print("0.98*l/r0 = %g  - FWHM %g " % (wants_fwhm, got_fwhm))
        self.assertAlmostEqual(wants_fwhm, got_fwhm, delta=wants_fwhm / 5)

    def test_wavefront_std(self):
        niter = 10
        r0_at_500 = 2 * u.m
        dz = 1e3 * u.m
        wl = 2.2e-6 * u.m
        telescope_radius = 19.5 * u.m
        r0 = r0_at_500 * (wl / (0.5e-6 * u.m))**(6 / 5)

        ss = poppy.OpticalSystem(pupil_diameter=2 * telescope_radius)
        wv = ss.input_wavefront(wl)
        wants_std = np.sqrt(1.03 * (2 * telescope_radius / r0)
                            ** (5. / 3)) / 2 / np.pi * wl
        for i in range(niter):
            kwfe = poppy.KolmogorovWFE(r0=r0, dz=dz, kind='Kolmogorov')
            opd = kwfe.get_opd(wv)
            print("sqrt(1.03 D/r0 ^ (5/3): %g std: %g nm" % (
                wants_std.to_value(u.nm), opd.std() * 1e9))


if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()
