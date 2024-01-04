import numpy as np
import astropy.units as u
from pathlib import Path
import os
from appoppy.elt_for_petalometry import EltForPetalometry
from appoppy.ao_residuals import AOResidual
from appoppy.package_data import data_root_dir
from appoppy.petalometer import PetalComputer, Petalometer
from astropy.io import fits
from appoppy.snapshotable import Snapshotable, SnapshotPrefix


class LepSnapshotEntry(object):
    NITER = 'NITER'
    ROT_ANGLE = 'ROTANG'
    TRACKNUM = 'TRACKNUM'
    LPE_TRACKNUM = 'LPE_TRACKNUM'
    PIXELSZ = 'PIXELSZ'
    STARTSTEP = 'STARTSTEP'
    M4_PETALS = 'PETALS'
    WAVELENGTH = 'WAVELENGTH'


def long_exposure_tracknum(tn, code):
    return "%s_%s" % (tn, code)


def long_exposure_filename(lep_tracknum):
    return os.path.join(data_root_dir(),
                        'long_exposure',
                        lep_tracknum,
                        'long_exp.fits')


def animation_folder(lep_tracknum):
    return os.path.join(data_root_dir(),
                        'appoppy_anim',
                        lep_tracknum)


class LongExposureSimulation(Snapshotable):
    SNAPSHOT_PREFIX = 'LPE'

    def __init__(self,
                 lpe_tracking_number,
                 passata_tracking_number,
                 rot_angle=10,
                 petals=np.array([0, 0, 0, 0, 0, 0]) * u.nm,
                 wavelength=2.2e-6 * u.m,
                 lwe_speed=None,
                 start_from_step=100,
                 n_iter=1000):
        self._start_from_step = start_from_step
        self._niter = n_iter
        self._rot_angle = rot_angle
        self._m4_petals = petals
        self._wavelength = wavelength
        self._lwe_speed = lwe_speed
        self._jpg_root = animation_folder(lpe_tracking_number)
        self._reconstructed_phase = None
        self._pixelsize = None
        self._lpe_tracking_number = lpe_tracking_number
        self._passata_tracking_number = passata_tracking_number
        if self._passata_tracking_number:
            aores = AOResidual(self._passata_tracking_number)
            self._time_step = aores.time_step
        self._pet = None
        self._meas_petals = None
        self._input_opd = None
        self._corrected_opd = None

    def run(self):
        self._pet = Petalometer(
            tracking_number=self._passata_tracking_number,
            lwe_speed=self._lwe_speed,
            petals=self._m4_petals,
            residual_wavefront_start_from=self._start_from_step,
            rotation_angle=self._rot_angle,
            wavelength=self._wavelength,
            should_display=False)
        self._pixelsize = self._pet.pixelsize.to_value(u.m)
        self._pet.sense_wavefront_jumps()
        sh = (self._niter,
              self._pet.reconstructed_phase.shape[0],
              self._pet.reconstructed_phase.shape[1])
        self._reconstructed_phase = np.ma.zeros(sh)
        self._meas_petals = np.zeros((self._niter, 6))
        self._input_opd = np.ma.zeros(sh)
        self._corrected_opd = np.ma.zeros(sh)
        self._pet.set_step_idx(0)
        for i in range(self._niter):
            print("step %d" % self._pet.step_idx)
            self._pet.sense_wavefront_jumps()
            self._meas_petals[self._pet.step_idx] = self._pet.estimated_petals
            self._reconstructed_phase[self._pet.step_idx] = self._pet.reconstructed_phase
            self._input_opd[self._pet.step_idx] = self._pet.pupil_opd
            self._corrected_opd[self._pet.step_idx] = self._input_opd[self._pet.step_idx] - \
                self.opd_correction(self._pet.reconstructed_phase,
                                    self._rot_angle,
                                    self._input_opd.shape[1])
            self._pet.advance_step_idx()

    def get_snapshot(self, prefix='LEP'):
        snapshot = {}
        snapshot[LepSnapshotEntry.NITER] = self._niter
        snapshot[LepSnapshotEntry.ROT_ANGLE] = self._rot_angle
        snapshot[LepSnapshotEntry.TRACKNUM] = self._passata_tracking_number
        snapshot[LepSnapshotEntry.LPE_TRACKNUM] = self._lpe_tracking_number
        snapshot[LepSnapshotEntry.PIXELSZ] = self._pixelsize
        snapshot[LepSnapshotEntry.STARTSTEP] = self._start_from_step
        snapshot[LepSnapshotEntry.M4_PETALS] = self._m4_petals
        snapshot[LepSnapshotEntry.WAVELENGTH] = self._wavelength.to_value(u.m)
        snapshot.update(self._pet.get_snapshot(SnapshotPrefix.PETALOMETER))
        return Snapshotable.prepend(prefix, snapshot)

    def save(self):
        filename = long_exposure_filename(self._lpe_tracking_number)
        rootpath = Path(filename).parent.absolute()
        rootpath.mkdir(parents=True, exist_ok=True)
        hdr = Snapshotable.as_fits_header(
            self.get_snapshot(self.SNAPSHOT_PREFIX))
        fits.writeto(filename, self._reconstructed_phase.data,
                     header=hdr, overwrite=True)
        fits.append(filename, self._reconstructed_phase.mask.astype(int))
        fits.append(filename, self._meas_petals)
        fits.append(filename, self._input_opd.data)
        fits.append(filename, self._input_opd.mask.astype(int))
        fits.append(filename, self._corrected_opd.data)

    @classmethod
    def opd_correction(cls, reconstructed_phase_map, rot_angle, npix):
        pc = PetalComputer(reconstructed_phase_map, rot_angle)
        pp = pc.estimated_petals_zero_mean
        efp = EltForPetalometry(npix=npix)
        efp.set_m4_petals(pp)
        opd = efp.pupil_opd()
        return opd
