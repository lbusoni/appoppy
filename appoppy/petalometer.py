import astropy.units as u
import numpy as np
from appoppy.interferometer import PhaseShiftInterferometer
from appoppy.system_for_petalometry import EltForPetalometry
from appoppy.mask import sector_mask
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
    '''
    Rotational Shearing Interferometer for the ELT
    
    
    
    '''

    def __init__(self,
                 r0=np.inf,
                 tracking_number=None,
                 lwe_speed=None,
                 residual_wavefront_start_from=100,
                 npix=256,
                 petals=np.array([0, 0, 0, 0, 0, 0]) * u.nm,
                 rotation_angle=15,
                 zernike=np.array([0]) * u.nm,
                 wavelength=2.2e-6 * u.m,
                 kolm_seed=None,
                 should_display=True,
                 should_unwrap=False):
        if kolm_seed is None:
            seed = np.random.randint(2147483647)

        self._log = logging.getLogger('appoppy')
        self._step_idx = 0
        self._res_average_on = 1
        self._should_display = should_display
        self._petals = petals

        self._jumps = None

        self._model1 = EltForPetalometry(
            r0=r0,
            tracking_number=tracking_number,
            kolm_seed=seed,
            lwe_speed=lwe_speed,
            rotation_angle=rotation_angle,
            zern_coeff=zernike,
            npix=npix,
            wavelength=wavelength,
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
            zern_coeff=zernike,
            npix=npix,
            wavelength=wavelength,
            residual_wavefront_start_from=residual_wavefront_start_from,
            residual_wavefront_average_on=self._res_average_on,
            residual_wavefront_step=0,
            name='M2')

        self.set_m4_petals(self._petals)
        self.set_zernike_wavefront(zernike)

        self._i4 = PhaseShiftInterferometer(self._model1, self._model2,
                                            should_unwrap)

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
    def pupil_opd(self):
        return self._model2.pupil_opd()

    @property
    def reconstructed_phase(self):
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
        # self._petals = zero_mean(petals, self.wavelength)
        self._model1.set_m4_petals(petals)
        self._model2.set_m4_petals(petals)
        self._petals = petals

    # def set_atmospheric_wavefront(self, atmospheric_wavefront):
    #     self._atmo_opd = atmospheric_wavefront
    #     self._model1.set_atmospheric_wavefront(self._atmo_opd)
    #     self._model2.set_atmospheric_wavefront(self._atmo_opd)

    @property
    def petals(self):
        '''
        Return M4 petals in nm
        
        The array corresponds to the one set via set_m4_petals().
        It doesn't account for any other possible petaling included
        e.g. in the atmospheric residuals or LWE. 

        See estimated_petals() for the Petalometer 
        measurement of the OPD between petals

        Returns
        ----------
        petals: astropy.quantity equivalent to u.nm of shape (6,)
            petals applied on M4
        '''
        return self._petals.to(u.nm)

    def sense_wavefront_jumps(self):
        self._i4.acquire()
        if self._should_display:
            self._i4.display_interferogram()
        self._pc = PetalComputer(self.reconstructed_phase,
                                 self._model1.pupil_rotation_angle)

    def advance_step_idx(self):
        self.set_step_idx(self._step_idx + self._res_average_on)

    def set_step_idx(self, step_idx):
        self._step_idx = step_idx
        self._log.info('set time step to %g' % self._step_idx)
        self._model1.set_step_idx(self._step_idx)
        self._model2.set_step_idx(self._step_idx)

    @property
    def step_idx(self):
        return self._step_idx

#    @property
#    def _expected_jumps(self):
#        dd = np.repeat(self.petals, 2)
#        # return wrap_around_zero(np.roll(dd, 1) - dd,
#        #                        self.wavelength)
#        return np.roll(dd, 1) - dd

#    @property
#    def error_jumps(self):
#        return (self._expected_jumps - self._pc.all_jumps).to(u.nm)

    @property
    def difference_between_estimated_petals_and_m4_petals(self):
        '''
        Difference between the estimateded petals and the petals
        set with set_m4_petals.
        
        Warning:  AO residual or other aberrations can also 
        create petals that the WFS correctly senses.

        Returns
        -------
        difference: numpy array (6)
            difference between estimated petals and M4 petals [nm]
        '''
        diff = self.estimated_petals - self.petals
        return diff - diff[0]

    @property
    def estimated_petals(self):
        return self._pc.estimated_petals


class PetalComputer():

    def __init__(self, reconstructed_phase, rotation_angle):
        self._rot_angle = rotation_angle
        self._reconstructed_phase = reconstructed_phase
        self._compute_jumps_from_reconstructed_phase()

    @property
    def estimated_petals(self):
        res = -1 * np.cumsum(self.across_islands_jumps)
        # return zero_mean(res, self.wavelength)
        res -= res.mean()
        return res

    @property
    def estimated_petals_zero_mean(self):
        pp = self.estimated_petals
        return pp - pp.mean()

    @property
    def across_islands_jumps(self):
        '''
        Measure OPD between sectors separated by a spider 
        
        See all_jumps() and compute_jumps()
        '''
        return self._jumps[::2]

    @property
    def all_jumps(self):
        '''
        Measured OPD between all sectors.

        See compute_jumps() with parameter reconstructed_phase() and
        rotation_angle 
        '''
        return self._jumps

    def _compute_jumps_from_reconstructed_phase(self):
        self._jumps = self.compute_jumps(
            self._reconstructed_phase, self._rot_angle)

    @classmethod
    def compute_jumps(cls, image, r):
        '''
        Compute jumps between pupil sectors.

        In the case of ELT, having six 60° circular sectors, 
        the rotational shearing interferometer output pupil image is made of
        12 sectors obtained by interfering the pupil with a copy of the pupil 
        rotated by R degrees.   

        The jumps are computed as median value of the interferogram in 

        Even-th jumps correspond to the interference of a segment with itself, so
        they should be nominally zero.
        Odd-th jumps correspond to OPD across pupil islands, what we are interested in.
        
        This method is provided for offline computation of the jumps, in case
        only the reconstructed_phase is available and 
        the Petalometer object is not valid anymore.
        
        Parameters
        ----------
        image: numpy masked array
            interferometer on which 
        
        rotation_angle: float
            rotation angle of the RSI  
        
        Returns
        -------
        jumps: numpy array (6)
            jumps difference between measured petals and actual petals [nm]
                 
        '''
        angs = (90, 90 - r, 30, 30 - r, -30, -30 - r, -
                90, -90 - r, -150, -150 - r, -210, -210 - r, -270)
        res = np.zeros(len(angs) - 1)
        for i in range(len(angs) - 1):
            ifm = cls._mask_ifgram(image, (angs[i + 1], angs[i]))
            res[i] = np.ma.median(ifm)
        # res = wrap_around_zero(res * u.nm, self.wavelength)
        return res * u.nm

    @classmethod
    def _mask_ifgram(cls, ifgram, angle_range):
        smask1 = sector_mask(ifgram.shape,
                             (angle_range[0], angle_range[1]))
        mask = np.ma.mask_or(ifgram.mask, ~smask1)
        return np.ma.masked_array(ifgram, mask=mask)
