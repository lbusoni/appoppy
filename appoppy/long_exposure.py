import numpy as np
import astropy.units as u
from pathlib import Path
import os
from appoppy.ao_residuals import AOResidual
from appoppy.petalometer import Petalometer
from appoppy.gif_animator import Gif2DMapsAnimator
from astropy.io import fits
import matplotlib.pyplot as plt
from appoppy.snapshotable import Snapshotable, SnapshotPrefix


class LepSnapshotEntry(object):
    NITER = 'NITER'
    ROT_ANGLE = 'ROTANG'
    TRACKNUM = 'TRACKNUM'
    LPE_TRACKNUM = 'LPE_TRACKNUM'
    PIXELSZ = 'PIXELSZ'
    STARTSTEP = 'STARTSTEP'
    PETALS = 'PETALS'
    WAVELENGTH = 'WAVELENGTH'
    

class LongExposurePetalometer(Snapshotable):

    def __init__(self,
                 lpe_tracking_number,
                 passata_tracking_number,
                 rot_angle=10,
                 petals=np.array([0, 0, 0, 0, 0, 0]) * u.nm,
                 wavelength=2.2e-6 * u.m,
                 lwe_speed=None,
                 jpeg_root_folder=None,
                 start_from_step=100,
                 n_iter=1000):
        self._start_from_step = start_from_step
        self._niter = n_iter
        self._rot_angle = rot_angle
        self._petals = petals
        self._wavelength = wavelength
        self._lwe_speed = lwe_speed
        if jpeg_root_folder is None:
            home = str(Path.home())
            jpeg_root_folder = os.path.join(
                home, 'appoppy_anim',
                lpe_tracking_number if lpe_tracking_number else "none")
        self._jpg_root = jpeg_root_folder
        self._phase_diff_cumave = None
        self._phase_diff = None
        self._meas_petals = None
        self._meas_petals_no_global_pist = None
        self._meas_petals_cumave = None
        self._pixelsize = None
        self._lpe_tracking_number = lpe_tracking_number
        self._passata_tracking_number = passata_tracking_number
        if self._passata_tracking_number:
            self._aores = AOResidual(self._passata_tracking_number)
        self._pet = None


    def run(self):
        self._pet = Petalometer(
            tracking_number=self._passata_tracking_number,
            lwe_speed=self._lwe_speed,
            petals=self._petals,
            residual_wavefront_start_from=self._start_from_step,
            rotation_angle=self._rot_angle,
            wavelength=self._wavelength,
            should_display=False)
        self._pixelsize = self._pet.pixelsize.to_value(u.m)
        sh = (self._niter,
              self._pet.phase_difference_map.shape[0],
              self._pet.phase_difference_map.shape[1])
        self._phase_diff = np.ma.zeros(sh)
        self._meas_petals = np.zeros((self._niter, 6))
        self._phase_screen = np.ma.zeros(sh)
        self._pet.set_step_idx(0)
        for i in range(self._niter):
            print("step %d" % self._pet._step_idx)
            self._pet.sense_wavefront_jumps()
            self._meas_petals[self._pet._step_idx] = self._pet.error_petals
            self._phase_diff[self._pet._step_idx] = self._pet.phase_difference_map
            self._phase_screen[self._pet._step_idx] = self._pet.pupil_opd
            self._pet.advance_step_idx()

    def phase_screen(self):
        '''
        Phase screens
        
        It returns a cube of frames where the i-th frame corresponds to
        the Optical Path Difference map of the i-th temporal step

        The OPD is obtained as sum of the aberration contributed by any
        optical element in the system (atmospheric turbulence/AO residual,
        low-wind-effect, M4 petals, Zernike aberration)  
        
        It is not guaranteed that each screen has null global piston, i.e.
        phase_screen.mean(axis=(1,2)) != 0

        Returns
        -------
        phase_screen: numpy array (n_iter, n_pixel, n_pixel)
            pupil opd [nm]
        
        '''
        return self._phase_screen
    
    def phase_screen_cumave(self):
        self._phase_screen_cumave = self._cumaverage(self.phase_screen())
        return self._phase_screen_cumave
    
    def phase_screen_ave(self):
        return self.phase_screen().mean(axis=0)

    # def phase_screen(self):
    #     return self._aores.phase_screen
    #
    # def phase_screen_cumave(self):
    #     return self._aores.phase_screen_cumave
    #
    # def phase_screen_ave(self):
    #     return self._aores.phase_screen_ave

    def petals(self):
        '''
        Petals
        
        Returns petals measured at every temporal step
        The global piston is removed, i.e. mean(petals, axis=1) == 0 
        
        ! It is the petal error, i.e. the difference between the petals set and the measured ones.
        
        Returns
        -------
        petals: numpy array (n_iter, 6)
            measured petals as a function of time [nm]

        '''
        if self._meas_petals_no_global_pist is None:
            self._meas_petals_no_global_pist = self._meas_petals - np.broadcast_to(
                self._meas_petals.mean(axis=1), (6, self._niter)).T
        return self._meas_petals_no_global_pist

    def petals_cumave(self):
        if self._meas_petals_cumave is None:
            self._meas_petals_cumave = self._cumaverage(self.petals())
        return self._meas_petals_cumave

    def phase_difference(self):
        '''
        Phase difference maps

        It returns a cube of frames where the i-th frame corresponds to
         the phase difference estimated from the short exposure at the i-th
         temporal step

        The phase difference map is the interferogram of the Rotational
        Shearing Interferometer, so every point of the map contains the
        measured phase difference of the two subapertures overlapped by the
        RSI  

        Returns
        -------
        phase_difference: numpy array (n_iter, n_pixel, n_pixel)
            petalometer phase difference maps.
        '''

        return self._phase_diff

    def phase_difference_cumave(self):
        '''
        Cumulative average of phase difference maps

        It returns a cube of frames where the i-th frame corresponds to
        a long exposure of the phase difference from frame 0 to frame i

        Returns
        -------
        phase_difference_cumave: numpy array
            cumulative temporal average of petalometer phase difference map.
        '''
        if self._phase_diff_cumave is None:
            self._phase_diff_cumave = self._cumaverage(self.phase_difference())
        return self._phase_diff_cumave

    def phase_difference_ave(self):
        '''
        Temporal average of phase difference maps

        It corresponds to a long exposure of the petalometer detector

        Returns
        -------
        phase_difference_ave: numpy array
            temporal average of petalometer phase difference map.
        '''
        return self.phase_difference().mean(axis=0)

    def _cumaverage(self, cube):
        cs = cube.cumsum(axis=0)
        res = np.array([cs[i] / (i + 1) for i in np.arange(cube.shape[0])])
        return res

    # def display_phase_difference_ave(self):
    #     self.display_map(self.phase_difference_ave())
    #
    # def display_phase_difference_std(self):
    #     self.display_map(self.phase_difference().std(axis=0))

    def _animate_generic(self, what, label, step, cmap='twilight', **kwargs):
        Gif2DMapsAnimator(
            os.path.join(self._jpg_root, label),
            what,
            deltat=self._aores.time_step,
            pixelsize=self._pixelsize, **kwargs).make_gif(step=step, cmap=cmap)

    def animate_phase_difference(self, vminmax=(-1100, 1100), remove_jpg=True):
        self._animate_generic(self.phase_difference(), 'phase_difference', 20,
                              vminmax=vminmax, remove_jpg=remove_jpg)

    def animate_phase_difference_cumulative_average(self, vminmax=(-1100, 1100), remove_jpg=True):
        self._animate_generic(self.phase_difference_cumave(), 'phase_difference_cum', 20,
                              vminmax=vminmax, remove_jpg=remove_jpg)

    def animate_phase_screens_cumulative_average(self, vminmax=(-300, 300), remove_jpg=True):
        self._animate_generic(self.phase_screen_cumave(), 'phase_screen_cum', 10,
                              cmap='cividis', vminmax=vminmax, remove_jpg=remove_jpg)

    def animate_phase_screens(self, vminmax=(-300, 300), remove_jpg=True):
        self._animate_generic(self.phase_screen(), 'phase_screen', 10,
                              cmap='cividis', vminmax=vminmax, remove_jpg=remove_jpg)

    def get_snapshot(self, prefix='LEP'):
        snapshot = {}
        snapshot[LepSnapshotEntry.NITER] = self._niter
        snapshot[LepSnapshotEntry.ROT_ANGLE] = self._rot_angle
        snapshot[LepSnapshotEntry.TRACKNUM] = self._passata_tracking_number
        snapshot[LepSnapshotEntry.LPE_TRACKNUM] = self._lpe_tracking_number
        snapshot[LepSnapshotEntry.PIXELSZ] = self._pixelsize
        snapshot[LepSnapshotEntry.STARTSTEP] = self._start_from_step
        snapshot[LepSnapshotEntry.PETALS] = self._petals
        snapshot[LepSnapshotEntry.WAVELENGTH] = self._wavelength.to_value(u.m)
        snapshot.update(self._pet.get_snapshot(SnapshotPrefix.PETALOMETER))
        return Snapshotable.prepend(prefix, snapshot)

    def save(self, filename):
        rootpath = Path(filename).parent.absolute()
        rootpath.mkdir(parents=True, exist_ok=True)
        hdr = Snapshotable.as_fits_header(self.get_snapshot('LPE'))
        fits.writeto(filename, self.phase_difference().data,
                     header=hdr, overwrite=True)
        fits.append(filename, self.phase_difference().mask.astype(int))
        fits.append(filename, self._meas_petals)
        fits.append(filename, self.phase_screen().data)
        fits.append(filename, self.phase_screen().mask.astype(int))

    @staticmethod
    def load(filename):
        hdr = fits.getheader(filename)
        try:
            res = hdr['LPE.' + LepSnapshotEntry.WAVELENGTH]
            if isinstance(res, str):
                wavelength = eval(res.split()[0]) * u.m
            else:
                wavelength = res * u.m
        except KeyError:
            wavelength = 24e-6 * u.m
        # TODO: fix wavelength
        soi = LongExposurePetalometer(
            hdr['LPE.' + LepSnapshotEntry.LPE_TRACKNUM],
            passata_tracking_number=hdr['LPE.' + LepSnapshotEntry.TRACKNUM],
            rot_angle=hdr['LPE.' + LepSnapshotEntry.ROT_ANGLE],
            petals=hdr['LPE.' + LepSnapshotEntry.PETALS],
            wavelength=wavelength,
            start_from_step=hdr['LPE.' + LepSnapshotEntry.STARTSTEP],
            n_iter=hdr['LPE.' + LepSnapshotEntry.NITER])
        soi._pixelsize = hdr['LPE.' + LepSnapshotEntry.PIXELSZ]
        res_map_d = fits.getdata(filename, 0)
        res_map_m = fits.getdata(filename, 1).astype(bool)
        soi._phase_diff = np.ma.masked_array(data=res_map_d, mask=res_map_m)
        soi._meas_petals = fits.getdata(filename, 2)
        phase_screen_d = fits.getdata(filename, 3)
        phase_screen_m = fits.getdata(filename, 4)
        soi._phase_screen = np.ma.masked_array(data=phase_screen_d, mask=phase_screen_m)
        return soi

    def plot_petals(self):
        t = np.arange(0,
                      self._niter * self._aores.time_step,
                      self._aores.time_step)

        dd = self.petals()
        plt.figure()
        plt.plot(t, dd, label=['Segment %s' % i for i in range(6)])
        plt.legend()
        plt.grid()
        plt.ylabel("Estimated piston [nm]")
        plt.xlabel("Time [s]")
        print("mean %s" % str(dd.mean(axis=0)))
        print("std  %s" % str(dd.std(axis=0)))

    def _compute_jumps_from_phase_difference_average(self, rotation_angle):
        return Petalometer.compute_jumps(
            self.phase_difference_ave(), rotation_angle)

    def petals_from_phase_difference_ave(self):
        jj = self._compute_jumps_from_phase_difference_average(self._rot_angle)
        pp = -1 * np.cumsum(jj[::2])
        pp = pp - pp.mean()
        return pp, jj

    def phase_correction_from_petalometer(self):
        pp, _ = self.petals_from_phase_difference_ave()
        pet = Petalometer(
            petals=pp,
            rotation_angle=self._rot_angle,
            npix=self.phase_screen_ave().shape[0],
            wavelength=self._wavelength,
            should_display=False)
        opd = pet.pupil_opd
        return opd

    def phase_screen_petal_corrected(self):
        phase_correction_cube = np.tile(self.phase_correction_from_petalometer(),
                                        (self.phase_screen().shape[0], 1, 1))
        return self.phase_screen()-phase_correction_cube


    @property
    def rot_angle(self):
        return self._rot_angle

