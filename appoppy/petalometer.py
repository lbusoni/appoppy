import astropy.units as u
import numpy as np
from appoppy.phase_shift_interferometer import PhaseShiftInterferometer
from appoppy.elt_for_petalometry import EltForPetalometry
from appoppy.mask import sector_mask
from appoppy.circular_math import wrap_around_zero
import logging
from appoppy.snapshotable import SnapshotPrefix, Snapshotable
from arte.utils.timestamp import Timestamp


class PetalometerSnapshotEntry(object):
    TIMESTAMP = "TIMESTAMP"
    STEP_IDX = "STEP_IDX"
    AVERAGE_RESIDUAL_ON = "AVE_RES_ON"
    PETALS = 'PETALS'
    SHOULD_DISPLAY = 'SHOULD_DISPLAY'


class Petalometer(Snapshotable):

    def __init__(self,
                 r0=np.inf,
                 tracking_number=None,
                 lwe_speed=None,
                 residual_wavefront_start_from=100,
                 npix=256,
                 petals=np.array([0, 0, 0, 0, 0, 0]) * u.nm,
                 rotation_angle=15,
                 zernike=[0, 0],
                 kolm_seed=None,
                 should_display=True):
        if kolm_seed is None:
            seed = np.random.randint(2147483647)
        # if residual_wavefront_index:
        #    residual_wavefront_index = np.random.randint(100, 1000)

        self._log = logging.getLogger('appoppy')
        self._step_idx = 0
        self._res_average_on = 1
        self._should_display = should_display
        self._petals = petals

        self._model1 = EltForPetalometry(
            r0=r0,
            tracking_number=tracking_number,
            kolm_seed=seed,
            lwe_speed=lwe_speed,
            rotation_angle=rotation_angle,
            npix=npix,
            residual_wavefront_start_from=residual_wavefront_start_from,
            residual_wavefront_average_on=self._res_average_on,
            residual_wavefront_step=0,
            name='M1')

        self._model2 = EltForPetalometry(
            r0=r0,
            tracking_number=tracking_number,
            kolm_seed=seed,
            lwe_speed=lwe_speed,
            rotation_angle=0,
            npix=npix,
            residual_wavefront_start_from=residual_wavefront_start_from,
            residual_wavefront_average_on=self._res_average_on,
            residual_wavefront_step=0,
            name='M2')

        self.set_m4_petals(self._petals)
        self.set_zernike_wavefront(zernike)

        self._i4 = PhaseShiftInterferometer(self._model1,  self._model2)
        self._i4.combine()
        self.sense_wavefront_jumps()

    def get_snapshot(self, prefix=SnapshotPrefix.PETALOMETER):
        snapshot = {}
        snapshot[PetalometerSnapshotEntry.TIMESTAMP] = Timestamp.nowUSec()
        snapshot[PetalometerSnapshotEntry.STEP_IDX] = self._step_idx
        snapshot[PetalometerSnapshotEntry.AVERAGE_RESIDUAL_ON] = \
            self._res_average_on
        snapshot[PetalometerSnapshotEntry.SHOULD_DISPLAY] = \
            self._should_display
        snapshot[PetalometerSnapshotEntry.PETALS] = np.array2string(
            self.petals)
        snapshot.update(
            self._model1.get_snapshot(SnapshotPrefix.PATH1))
        snapshot.update(
            self._model2.get_snapshot(SnapshotPrefix.PATH2))
        snapshot.update(
            self._i4.get_snapshot(SnapshotPrefix.PHASE_SHIFT_INTERFEROMETER))
        return Snapshotable.prepend(prefix, snapshot)

    def should_display(self, true_or_false):
        self._should_display = true_or_false

    @property
    def wavelength(self):
        return self._model1.wavelength

    @property
    def pixelsize(self):
        return self._model1.pixelsize

    @property
    def phase_difference_map(self):
        '''
        Map of phase difference between subapertures overlapped by the
        rotational shearing.

        The point of polar coordinates (rho, theta) contains the
        phase difference between the points (rho, theta) and
        (rho, theta - rotation_angle)


        Returns
        -------
        phase_map: numpy array
            last computed phase map in nm, wrapped in
            the range (-wavelength/2, wavelength/2)
        '''
        return self._i4.interferogram()

    def set_zernike_wavefront(self, zernike_coefficients_in_m):
        self._model1.set_input_wavefront_zernike(zernike_coefficients_in_m)
        self._model2.set_input_wavefront_zernike(zernike_coefficients_in_m)

    def set_m4_petals(self, petals):
        '''
        Set M4 petals

        Parameters
        ----------
        petals: astropy.quantity equivalent to u.m of shape (6,)
            petals to be applied on M4
        '''
        #self._petals = zero_mean(petals, self.wavelength)
        self._petals = petals
        self._model1.set_m4_petals(self._petals)
        self._model2.set_m4_petals(self._petals)

    # def set_atmospheric_wavefront(self, atmospheric_wavefront):
    #     self._atmo_opd = atmospheric_wavefront
    #     self._model1.set_atmospheric_wavefront(self._atmo_opd)
    #     self._model2.set_atmospheric_wavefront(self._atmo_opd)

    @property
    def petals(self):
        return self._petals.to(u.nm)

    def sense_wavefront_jumps(self):
        self._i4.acquire()
        if self._should_display:
            self._i4.display_interferogram()
        self._compute_jumps_of_interferogram()
        return self.error_jumps

    def advance_step_idx(self):
        self.set_step_idx(self._step_idx + self._res_average_on)

    def set_step_idx(self, step_idx):
        self._step_idx = step_idx
        self._log.info('set time step to %g' % self._step_idx)
        self._model1.set_step_idx(self._step_idx)
        self._model2.set_step_idx(self._step_idx)

    @property
    def _expected_jumps(self):
        dd = np.repeat(self.petals, 2)
        # return wrap_around_zero(np.roll(dd, 1) - dd,
        #                        self.wavelength)
        return np.roll(dd, 1) - dd

    @property
    def error_jumps(self):
        # return wrap_around_zero(
        #    (self._expected_jumps() - self._jumps).to(u.nm),
        #    self.wavelength)
        return (self._expected_jumps - self.all_jumps).to(u.nm)

    @property
    def error_petals(self):
        # return difference(self.estimated_petals, self.petals,
        # self.wavelength)
        diff = self.estimated_petals - self.petals
        return diff - diff[0]

    @property
    def across_islands_jumps(self):
        '''
        Measure OPD between sectors separated by a spider 
        '''
        return self._jumps[::2]

    @property
    def all_jumps(self):
        '''
        Measured OPD between all sectors.

        Even-th jumps correspond to the interference of a segment with itself, so
        they should be nominally zero.
        Odd-th jumps correspond to OPD across islands, what we are interested in. 
        '''
        return self._jumps

    def _compute_jumps_of_interferogram(self):
        r = self._model1.pupil_rotation_angle
        image = self._i4.interferogram()
        self._jumps = self.compute_jumps(image, r)

    @classmethod
    def compute_jumps(cls, image, r):
        angs = (90, 90 - r, 30, 30 - r, -30, -30 - r, -
                90, -90 - r, -150, -150 - r, -210, -210 - r, -270)
        res = np.zeros(len(angs) - 1)
        for i in range(len(angs) - 1):
            ifm = cls._mask_ifgram(image, (angs[i + 1], angs[i]))
            res[i] = np.ma.median(ifm)
        #res = wrap_around_zero(res * u.nm, self.wavelength)
        return res * u.nm

    @classmethod
    def _mask_ifgram(cls, ifgram, angle_range):
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

    @property
    def estimated_petals_wrong(self):
        return wrap_around_zero(
            np.dot(self._jumps_to_petals_matrix(),
                   np.append(self.across_islands_jumps, 0)),
            self.wavelength)

    @property
    def estimated_petals(self):
        res = -1 * np.cumsum(self.across_islands_jumps)
        # return zero_mean(res, self.wavelength)
        return res
