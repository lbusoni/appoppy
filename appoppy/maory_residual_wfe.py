from poppy.poppy_core import ArrayOpticalElement
import os
import numpy as np
import scipy.io
from appoppy.package_data import data_root_dir
from appoppy.elt_aperture import restore_elt_pupil_mask
from astropy.io import fits
import astropy.units as u
from appoppy import elt_aperture
from scipy.ndimage import rotate
from appoppy.snapshotable import Snapshotable


def transfer_from_faramir_to_drive(tracking_number,
                                   tag_profile='SteSci2021/P10'):
    main_dir1 = '/Volumes/GoogleDrive/Drive condivisi/MORFEO-OAA/Petalometro Ciao Ciao/Simulazioni PASSATA/'
    main_dir2 = '/Volumes/GoogleDrive/Drive\ condivisi/MORFEO-OAA/Petalometro\ Ciao\ Ciao/Simulazioni\ PASSATA/'
    os.mkdir(os.path.join(main_dir1, tracking_number))
    remote_path = "/raid1/guido/results/MAORY/%s/mcaoFull/" % tag_profile
    os.system("scp " + "gcarla@faramir:" + remote_path + tracking_number +
              "/params.txt " + main_dir2 + tracking_number + "/")
    os.system("scp -r " + "gcarla@faramir:" + remote_path +
              tracking_number + "_oaCUBEs/* " + main_dir2 + tracking_number + "/")


class PASSATASimulationConverter():
    '''
    Convert data from PASSATA simulations in the standard used inside
    "MaoryResidualWavefront" class.
    '''

    def __init__(self):
        pass

    def convert_from_sav(self):

        fname_sav = os.path.join(data_root_dir(),
                                 'passata_simulations',
                                 '20210518_223459.0',
                                 'CUBE_CL_coo0.0_0.0.sav')

        fname_fits = os.path.join(data_root_dir(),
                                  'passata_simulations_converted',
                                  '20210518_223459.0',
                                  'CUBE_CL_converted.fits')

        pupilmasktag = elt_aperture.PUPIL_MASK_480
        idl_dict = scipy.io.readsav(fname_sav)
        phase_screen = np.moveaxis(idl_dict['cube_k'], 2, 0)
        maskhdu = restore_elt_pupil_mask(pupilmasktag)
        mask = maskhdu.data
        maskhdr = maskhdu.header
        cmask = np.tile(mask, (phase_screen.shape[0], 1, 1))
        res_wfs = np.ma.masked_array(phase_screen, mask=cmask)
        header = fits.Header()
        header[AoResSnapshotEntry.TRACKING_NUMBER] = idl_dict['tn'].decode(
            "utf-8")
        header[AoResSnapshotEntry.COORDINATE_RHO] = \
            float(idl_dict['polar_coordinates_k'][0])
        header[AoResSnapshotEntry.COORDINATE_THETA] = \
            float(idl_dict['polar_coordinates_k'][1])
        header[AoResSnapshotEntry.PIXEL_SCALE] = maskhdr['PIXELSCL']
        header[AoResSnapshotEntry.PUPIL_TAG] = pupilmasktag
        header[AoResSnapshotEntry.TIME_STEP] = 0.002
        fits.writeto(fname_fits, res_wfs.data, header)
        fits.append(fname_fits, mask.astype(int))

    def convert_from_fits_data(self, tracking_number, rho, theta, pupilmasktag,
                               timestep):
        fname_orig_fits = os.path.join(data_root_dir(),
                                       'passata_simulations',
                                       tracking_number,
                                       'CUBE_CL_coo%s_%s.fits' % (rho, theta))
        fname_newdir = os.path.join(data_root_dir(), 'passata_simulations_converted',
                                    tracking_number + '_coo%s_%s' % (rho, theta))
        os.makedirs(fname_newdir, exist_ok=True)
        fname_fits = os.path.join(fname_newdir, 'CUBE_CL_converted.fits')
        dat, hdr = fits.getdata(fname_orig_fits, 0, header=True)
        maskhdu = restore_elt_pupil_mask(pupilmasktag)
        maskhdr = maskhdu.header
        mask = maskhdu.data
        rotation = float(maskhdr['ROTATION'])

        phase_screen = rotate(
            dat, rotation, reshape=False, cval=0,
            mode='constant', axes=(1, 0))

        phase_screen = np.moveaxis(phase_screen, -1, 0)

        cmask = np.tile(mask, (phase_screen.shape[0], 1, 1))
        res_wfs = np.ma.masked_array(phase_screen, mask=cmask)

        header = fits.Header()
        header[AoResSnapshotEntry.TRACKING_NUMBER] = hdr['TN']
        header[AoResSnapshotEntry.COORDINATE_RHO] = float(hdr['POCOO0'])
        header[AoResSnapshotEntry.COORDINATE_THETA] = float(hdr['POCOO1'])
        header[AoResSnapshotEntry.PIXEL_SCALE] = maskhdr['PIXELSCL']
        header[AoResSnapshotEntry.PUPIL_TAG] = pupilmasktag
        header[AoResSnapshotEntry.TIME_STEP] = timestep
        fits.writeto(fname_fits, res_wfs.data, header)
        fits.append(fname_fits, mask.astype(int))

    def create_none_tracknum(self):
        mask = np.zeros((500, 64, 64), dtype=bool)
        res_wf = np.ma.masked_array(
            np.zeros((500, 64, 64)),
            mask=mask)
        header = fits.Header()
        header[AoResSnapshotEntry.TRACKING_NUMBER] = 'None'
        header[AoResSnapshotEntry.COORDINATE_RHO] = float(0)
        header[AoResSnapshotEntry.COORDINATE_THETA] = float(0)
        header[AoResSnapshotEntry.PIXEL_SCALE] = 1
        header[AoResSnapshotEntry.PUPIL_TAG] = 'None'
        header[AoResSnapshotEntry.TIME_STEP] = 1
        fname_newdir = os.path.join(
            data_root_dir(), 'passata_simulations_converted', 'none')
        os.makedirs(fname_newdir, exist_ok=True)
        fname_fits = os.path.join(fname_newdir, 'CUBE_CL_converted.fits')
        fits.writeto(fname_fits, res_wf.data, header)
        fits.append(fname_fits, mask[0].astype(int))


def restore_residual_wavefront(tracking_number):
    #  '20210518_223459.0'

    fname_fits = os.path.join(data_root_dir(),
                              'passata_simulations_converted',
                              tracking_number,
                              'CUBE_CL_converted.fits')
    dat, hdr = fits.getdata(fname_fits, 0, header=True)
    mas = fits.getdata(fname_fits, 1).astype(bool)
    cmask = np.tile(mas, (dat.shape[0], 1, 1))
    return np.ma.masked_array(dat, mask=cmask), hdr


class AoResSnapshotEntry(object):
    TRACKING_NUMBER = 'TN'
    PIXEL_SCALE = 'PIXELSCL'
    PUPIL_TAG = 'PUPILTAG'
    TIME_STEP = 'TIME_STEP'
    STEP_IDX = 'STEP_IDX'
    START_FROM_FRAME = 'START_FROM'
    AVERAGE_HOW_MANY_FRAMES = 'AVE_ON'
    ADVANCE_BY_FRAMES = 'STEP'
    COORDINATE_RHO = 'COO_RO'
    COORDINATE_THETA = 'COO_TH'


class MaoryResidualWavefront(ArrayOpticalElement, Snapshotable):
    '''
    Return an OpticalElement whose opd represents the MAORY residual wavefront
    simulated by PASSATA

    Data consists of 1375 frames sampled at 500Hz of residual wavefront on-axis
    in median conditions

    Selected data are available in the shared gdrive
    "MORFEO-OAA/Petalometro Ciao Ciao/Simulazioni PASSATA" where a README.gdoc
    explains the simulated configuration.
    The user must copy the corresponding folder (e.g. 20210518_223459.0)
    in appoppy/data and pip install again to copy the data in the proper folder.


    Parameters
    ----------
    tracking_number: string or None
        tracking number of the PASSATA simulation. None returns null opds
    start_from: int (optional, default=100)
        first frame to use
    step: int (optional, default=0)
        how many frames to jump at each get_opd call
    average_on: int (optional, default=1)
        how many frames to average on each get_opd call

    Example
    -------
    MaoryResidualWavefront(start_from=200, step=0, average_on=1) returns always
    the 200th frame

    MaoryResidualWavefront(start_from=200, step=50, average_on=10) yields
    (mean(200:210), mean(250:260), mean(300:310)...)
    '''
    _START_FROM = 100

    def __init__(self,
                 tracking_number,
                 start_from=None,
                 step=0,
                 average_on=1,
                 **kwargs):
        self._tracknum = tracking_number
        self._res_wf, hdr = self._restore_residual_wavefront_or_none(
            self._tracknum)
        self._pxscale = hdr[AoResSnapshotEntry.PIXEL_SCALE] * u.m / u.pixel
        self._pupiltag = hdr[AoResSnapshotEntry.PUPIL_TAG]
        self._time_step = hdr[AoResSnapshotEntry.TIME_STEP]
        self._nframes = self._res_wf.shape[0]

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

    def _restore_residual_wavefront_or_none(self, tracknum):
        if tracknum is None:
            res_wf = np.ma.masked_array(
                np.zeros((500, 64, 64)),
                mask=np.zeros((500, 64, 64), dtype=bool))
            hdr = {}
            hdr[AoResSnapshotEntry.PIXEL_SCALE] = 1
            hdr[AoResSnapshotEntry.PUPIL_TAG] = 'None'
            hdr[AoResSnapshotEntry.TIME_STEP] = 1
        else:
            res_wf, hdr = restore_residual_wavefront(tracknum)
        return res_wf, hdr

    def get_snapshot(self, prefix='AORES'):
        snapshot = {}
        snapshot[AoResSnapshotEntry.ADVANCE_BY_FRAMES] = self._step
        snapshot[AoResSnapshotEntry.AVERAGE_HOW_MANY_FRAMES] = self._average_on
        snapshot[AoResSnapshotEntry.PIXEL_SCALE] = self._pxscale.to_value(
            u.m / u.pixel)
        snapshot[AoResSnapshotEntry.PUPIL_TAG] = self._pupiltag
        snapshot[AoResSnapshotEntry.START_FROM_FRAME] = self._start_from
        snapshot[AoResSnapshotEntry.STEP_IDX] = self._step_idx
        snapshot[AoResSnapshotEntry.TIME_STEP] = self._time_step
        snapshot[AoResSnapshotEntry.TRACKING_NUMBER] = self._tracknum
        return Snapshotable.prepend(prefix, snapshot)

    @property
    def time_step(self):
        return self._time_step

    @property
    def pupiltag(self):
        return self._pupiltag

    @property
    def shape(self):
        return self._res_wf.shape

    def set_step_idx(self, step_idx):
        self._step_idx = step_idx

    def get_opd(self, wave):
        first = self._step_idx
        last = np.minimum(self._step_idx + self._average_on, self._nframes)
        self.opd = np.mean(self._res_wf[first:last].data, axis=0) * 1e-9
        self._remove_global_piston()
        self._step_idx = (self._step_idx + self._step) % self._nframes
        return self.opd

    def _remove_global_piston(self):
        self.opd -= self.opd.mean()

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
