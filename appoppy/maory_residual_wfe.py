from poppy.poppy_core import ArrayOpticalElement
import os
import numpy as np
import scipy.io
from appoppy.package_data import data_root_dir
from appoppy.elt_aperture import restore_elt_pupil_mask
from astropy.io import fits
import astropy.units as u


def convert_residual_wavefront():

    fname_sav = os.path.join(data_root_dir(),
                             '20210518_223459.0',
                             'CUBE_OLCUBE_CL_coo0.0_0.0.sav')
    fname_fits = os.path.join(data_root_dir(),
                              '20210518_223459.0',
                              'CUBE_OLCUBE_CL_coo0.0_0.0.fits')

    idl_dict = scipy.io.readsav(fname_sav)
    phase_screen = np.moveaxis(idl_dict['cube_k'], 2, 0)
    maskhdu = restore_elt_pupil_mask()
    mask = maskhdu.data
    maskhdr = maskhdu.header
    cmask = np.tile(mask, (phase_screen.shape[0], 1, 1))
    res_wfs = np.ma.masked_array(phase_screen, mask=cmask)
    header = fits.Header()
    header['TN'] = idl_dict['tn'].decode("utf-8")
    header['DIR'] = idl_dict['dir'].decode("utf-8")
    header['COO_RO'] = float(idl_dict['polar_coordinates_k'][0])
    header['COO_TH'] = float(idl_dict['polar_coordinates_k'][1])
    header['PIXELSCL'] = maskhdr['PIXELSCL']
    fits.writeto(fname_fits, res_wfs.data, header)
    fits.append(fname_fits, mask.astype(int))


def restore_residual_wavefront():

    fname_fits = os.path.join(data_root_dir(),
                              '20210518_223459.0',
                              'CUBE_OLCUBE_CL_coo0.0_0.0.fits')
    dat, hdr = fits.getdata(fname_fits, 0, header=True)
    mas = fits.getdata(fname_fits, 1).astype(bool)
    cmask = np.tile(mas, (dat.shape[0], 1, 1))
    return np.ma.masked_array(dat, mask=cmask), hdr


class MaoryResidualWavefront():

    def __init__(self):
        self._res_wf, hdr = restore_residual_wavefront()
        self._pxscale = hdr['PIXELSCL'] * u.m / u.pixel
        self._step_idx = 100  # skip first 100 frames

    def as_optical_element(self, step=None, average_on=1):
        if step is None:
            step = self._step_idx
            self._step_idx += average_on
        opd = np.mean(self._res_wf[step:step + average_on].data, axis=0) * 1e-9
        transmission = (~self._res_wf[step].mask).astype(int)
        return ArrayOpticalElement(
            opd=opd,
            transmission=transmission,
            pixelscale=self._pxscale)
