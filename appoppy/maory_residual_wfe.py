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


class MaoryResidualWavefront(ArrayOpticalElement):
    '''
    Return an OpticalElement whose opd represents the MAORY residual wavefront
    simulated by PASSATA

    Data consists of 1275 frames sampled at 500Hz of residual wavefront on-axis
    in median conditions

    Parameters
    ----------
    start_from: int (optional, default=100)
        first frame to use
    step: int (optional, default=1)
        how many frames to jump at each get_opd call
    average_on: int (optional, default=1)
        how many frames to average on each get_opd call
    adjust_factor: float (optional, default=1)
        multiply residual by

    Example
    -------
    MaoryResidualWavefron(start_from=200, step=0, average_on=1) returns always
    the 200th frame

    MaoryResidualWavefron(start_from=200, step=50, average_on=10) yields
    (mean(200:210), mean(250:260), mean(300:310)...)
    '''
    _START_FROM = 100

    def __init__(self, 
                 start_from=None, 
                 step=1, 
                 average_on=1,
                 adjust_factor=0.1, 
                 **kwargs):
        self._res_wf, hdr = restore_residual_wavefront()
        self._pxscale = hdr['PIXELSCL'] * u.m / u.pixel
        self._nframes = self._res_wf.shape[0]
        self._adjust_factor = adjust_factor

        if start_from is None:
            self._start_from = self._START_FROM
        else:
            self._start_from = start_from
        self._step = step
        self._average_on = average_on
        self._step_idx = np.maximum(
            0, np.minimum(self._start_from, self._nframes))
        self.amplitude = (~self._res_wf[self._START_FROM].mask).astype(int)
        super(MaoryResidualWavefront, self).__init__(
            transmission=self.amplitude, pixelscale=self._pxscale, **kwargs)

    def get_opd(self, wave):
        first = self._step_idx
        last = np.minimum(self._step_idx + self._average_on, self._nframes)
        self.opd = np.mean(self._res_wf[first:last].data, axis=0) * 1e-9
        self._step_idx = (self._step_idx + self._step) % self._nframes
        return self.opd * self._adjust_factor

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
