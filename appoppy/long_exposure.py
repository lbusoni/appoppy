import numpy as np
import astropy.units as u
from pathlib import Path
import os
from appoppy.ao_residuals import AOResidual
from appoppy.elt_for_petalometry import EltForPetalometry
from appoppy.package_data import data_root_dir
from appoppy.petalometer import PetalComputer, Petalometer
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
    

def long_exposure_tracknum(tn, code):
    return "%s_%s" % (tn, code)


def long_exposure_filename(lep_tracknum):
    return os.path.join(data_root_dir(),
                        'long_exposure',
                        lep_tracknum,
                        'long_exp.fits')


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
        self._reconstructed_phase = None
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
            self._meas_petals[self._pet.step_idx] = self._pet.error_petals
            self._reconstructed_phase[self._pet.step_idx] = self._pet.reconstructed_phase
            self._input_opd[self._pet.step_idx] = self._pet.pupil_opd
            self._corrected_opd[self._pet.step_idx] = self._input_opd[self._pet.step_idx] - \
                self._opd_correction(self._pet.reconstructed_phase)
            self._pet.advance_step_idx()

    def input_opd(self):
        '''
        Pupil OPD before petal correction
        
        It returns a cube of frames where the i-th frame corresponds to
        the Optical Path Difference map of the i-th temporal step

        The OPD is obtained as sum of the aberration contributed by any
        optical element in the system (atmospheric turbulence/AO residual,
        low-wind-effect, M4 petals, Zernike aberration)  
        
        It is not guaranteed that each screen has null global piston, i.e.
        phase_screen.mean(axis=(1,2)) != 0

        Returns
        -------
        input_opd: numpy array (n_iter, n_pixel, n_pixel)
            pupil opd [nm]
        
        '''
        return self._input_opd
    
    def input_opd_cumave(self):
        self._phase_screen_cumave = self._cumaverage(self.input_opd())
        return self._phase_screen_cumave
    
    def input_opd_ave(self):
        return self.input_opd().mean(axis=0)

    def input_opd_std(self):
        '''
        Stdev of pupil OPD before petal correction
        
        Computed as temporal average of the spatial std of
        the OPD before the petal correction
        
        Returns
        -------
        phase_screen_std: float
            std of pupil opd before petal correction [nm]        
        '''
        return self.input_opd().std(axis=(1, 2)).mean()

    def corrected_opd(self):
        '''
        Residual pupil OPD after instantaneus petal correction
        
        It returns a cube of frames where the i-th frame corresponds to
        the Optical Path Difference map of the i-th temporal step after
        instantaneus petal correction

        The petal correction is based on the short-exposure reconstructed 
        phase, i.e. on the phase reconstructed from the i-th frame  
        
        Returns
        -------
        corrected_opd: numpy array (n_iter, n_pixel, n_pixel)
            pupil opd after petal correction [nm]
        
        '''
        return self._corrected_opd

    def corrected_opd_cumave(self):
        self._corrected_opd_cumave = self._cumaverage(self.corrected_opd())
        return self._corrected_opd_cumave

    def corrected_opd_ave(self):
        return self.corrected_opd().mean(axis=0)

    def corrected_opd_std(self):
        '''
        Stdev of residual opd after petal correction
        
        Computed as temporal average of the spatial std of
        the corrected OPD
        
        Returns
        -------
        corrected_opd_std: float
            std of pupil opd after petal correction [nm]        
        '''
        return self.corrected_opd().std(axis=(1, 2)).mean()

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

    def reconstructed_phase(self):
        '''
        Reconstructed phase maps

        It returns a cube of frames where the i-th frame corresponds to
         the phase difference estimated from the short exposure at the i-th
         temporal step

        The phase difference map is the phase reconstructed from the
        interferogram of the Rotational Shearing Interferometer, 
        so every point of the map contains the
        measured phase difference of the two subapertures overlapped by the
        RSI  

        Returns
        -------
        reconstructed_phase: numpy array (n_iter, n_pixel, n_pixel)
            petalometer reconstructed phase maps.
        '''

        return self._reconstructed_phase

    def reconstructed_phase_cumave(self):
        '''
        Cumulative average of reconstructed phase maps

        It returns a cube of frames where the i-th frame corresponds to
        a long exposure of the reconstructed phase from frame 0 to frame i

        Returns
        -------
        reconstructed_phase_cumave: numpy array (n_iter, n_pixel, n_pixel)
            cumulative temporal average of petalometer reconstructed phase.
        '''
        if self._phase_diff_cumave is None:
            self._phase_diff_cumave = self._cumaverage(
                self.reconstructed_phase())
        return self._phase_diff_cumave

    def reconstructed_phase_ave(self):
        '''
        Temporal average of reconstructed phase maps

        Returns
        -------
        reconstructed_phase_ave: numpy array (n_pixel, n_pixel)
            temporal average of petalometer reconstructed phase maps.
        '''
        return self.reconstructed_phase().mean(axis=0)

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

    def animate_reconstructed_phase(self, vminmax=(-1100, 1100), remove_jpg=True):
        self._animate_generic(self.reconstructed_phase(), 'reconstructed_phase', 20,
                              vminmax=vminmax, remove_jpg=remove_jpg)

    def animate_reconstructed_phase_cumulative_average(self, vminmax=(-1100, 1100), remove_jpg=True):
        self._animate_generic(self.reconstructed_phase_cumave(), 'reconstructed_phase_cum', 20,
                              vminmax=vminmax, remove_jpg=remove_jpg)

    def animate_input_opd_cumulative_average(self, vminmax=(-300, 300), remove_jpg=True):
        self._animate_generic(self.input_opd_cumave(), 'input_opd_cum', 10,
                              cmap='cividis', vminmax=vminmax, remove_jpg=remove_jpg)

    def animate_input_opd(self, vminmax=(-300, 300), remove_jpg=True):
        self._animate_generic(self.input_opd(), 'input_opd', 10,
                              cmap='cividis', vminmax=vminmax, remove_jpg=remove_jpg)

    def animate_corrected_opd_cumulative_average(self, vminmax=(-300, 300), remove_jpg=True):
        self._animate_generic(self.corrected_opd_cumave(), 'corrected_opd_cum', 10,
                              cmap='cividis', vminmax=vminmax, remove_jpg=remove_jpg)

    def animate_corrected_opd(self, vminmax=(-300, 300), remove_jpg=True):
        self._animate_generic(self.corrected_opd(), 'corrected_opd', 10,
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

    def save(self, lep_tracknum):
        filename = long_exposure_filename(lep_tracknum)
        rootpath = Path(filename).parent.absolute()
        rootpath.mkdir(parents=True, exist_ok=True)
        hdr = Snapshotable.as_fits_header(self.get_snapshot('LPE'))
        fits.writeto(filename, self.reconstructed_phase().data,
                     header=hdr, overwrite=True)
        fits.append(filename, self.reconstructed_phase().mask.astype(int))
        fits.append(filename, self._meas_petals)
        fits.append(filename, self.input_opd().data)
        fits.append(filename, self.input_opd().mask.astype(int))
        fits.append(filename, self.corrected_opd().data)

    @staticmethod
    def load(lep_tracknum):
        filename = long_exposure_filename(lep_tracknum)
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
        soi._reconstructed_phase = np.ma.masked_array(
            data=res_map_d, mask=res_map_m)
        soi._meas_petals = fits.getdata(filename, 2)
        phase_screen_d = fits.getdata(filename, 3)
        phase_screen_m = fits.getdata(filename, 4)
        soi._input_opd = np.ma.masked_array(
            data=phase_screen_d, mask=phase_screen_m)
        corrected_opd_d = fits.getdata(filename, 5)
        soi._corrected_opd = np.ma.masked_array(
            data=corrected_opd_d, mask=phase_screen_m)

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

    def petals_from_reconstructed_phase_map(self, reconstructed_phase_map):
        pc = PetalComputer(reconstructed_phase_map, self._rot_angle)
        pp = pc.estimated_petals
        pp = pp - pp.mean()
        return pp, pc.all_jumps

    def _opd_correction(self, reconstructed_phase_map):
        pp, _ = self.petals_from_reconstructed_phase_map(
            reconstructed_phase_map)
        efp = EltForPetalometry(
            npix=self.input_opd().shape[1])
        efp.set_m4_petals(pp)
        opd = efp.pupil_opd()
        return opd

    def opd_correction_from_reconstructed_phase_ave(self):
        return self._opd_correction(self.reconstructed_phase_ave())

    def corrected_opd_from_reconstructed_phase_ave(self):
        '''
        Residual OPD after petal correction using long exposure
        reconstructed phase
        
        It returns a cube of frames where the i-th frame corresponds to
        the Optical Path Difference map of the i-th temporal step after
        petal correction

        The petal correction is computed from the long-exposure reconstructed
        phase (see reconstructed_phase_ave), i.e. the command to the corrector
        is the same for all the n_iter frames.
        
        Returns
        -------
        corrected_opd: numpy array (n_iter, n_pixel, n_pixel)
            pupil opd after long-exposure petal correction [nm]
        
        '''

        phase_correction_cube = np.tile(self.opd_correction_from_reconstructed_phase_ave(),
                                        (self.input_opd().shape[0], 1, 1))
        return self.input_opd()-phase_correction_cube


    @property
    def rot_angle(self):
        return self._rot_angle

