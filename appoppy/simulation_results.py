from appoppy.ao_residuals import AOResidual
from appoppy.gif_animator import Gif2DMapsAnimator
from appoppy.long_exposure_simulation import LepSnapshotEntry, LongExposureSimulation, animation_folder, long_exposure_filename
from appoppy.petalometer import PetalComputer
import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits

import os


class SimulationResults():
    SNAPSHOT_PREFIX = 'LPE'

    def __init__(self,
                 hdr,
                 reconstructed_phase,
                 meas_petals,
                 input_opd,
                 corrected_opd,
                 ):
        self._hdr = hdr
        self._niter = self._hdr_value(LepSnapshotEntry.NITER)
        self._rot_angle = self._hdr_value(LepSnapshotEntry.ROT_ANGLE)
        self._lpe_tracking_number = self._hdr_value(
            LepSnapshotEntry.LPE_TRACKNUM)
        self._jpg_root = animation_folder(self._lpe_tracking_number)
        self.passata_tracking_number = self._hdr_value(
            LepSnapshotEntry.TRACKNUM)
        aores = AOResidual(self.passata_tracking_number)
        self._time_step = aores.time_step
        self.wavelength = self._get_wavelength()
        self.pixelsize = self._hdr_value(LepSnapshotEntry.PIXELSZ)
        self.m4_petals = self._hdr_value(LepSnapshotEntry.M4_PETALS)
        self.start_from_step = self._hdr_value(LepSnapshotEntry.STARTSTEP)

        self._reconstructed_phase = reconstructed_phase
        self._meas_petals = meas_petals
        self._input_opd = input_opd
        self._corrected_opd = corrected_opd

        self._phase_diff_cumave = None
        self._meas_petals_no_global_pist = None
        self._meas_petals_cumave = None

    def _hdr_value(self, key):
        return self._hdr[LongExposureSimulation.SNAPSHOT_PREFIX+'.'+key]

    def _get_wavelength(self):
        try:
            res = self._hdr_value(LepSnapshotEntry.WAVELENGTH)
            if isinstance(res, str):
                wavelength = eval(res.split()[0]) * u.m
            else:
                wavelength = res * u.m
        except KeyError:
            wavelength = 24e-6 * u.m
        # TODO: fix wavelength
        return wavelength

    @staticmethod
    def load(lep_tracknum):
        filename = long_exposure_filename(lep_tracknum)
        hdr = fits.getheader(filename)
        res_map_d = fits.getdata(filename, 0)
        res_map_m = fits.getdata(filename, 1).astype(bool)
        reconstructed_phase = np.ma.masked_array(
            data=res_map_d, mask=res_map_m)
        meas_petals = fits.getdata(filename, 2)
        phase_screen_d = fits.getdata(filename, 3)
        phase_screen_m = fits.getdata(filename, 4)
        input_opd = np.ma.masked_array(
            data=phase_screen_d, mask=phase_screen_m)
        corrected_opd_d = fits.getdata(filename, 5)
        corrected_opd = np.ma.masked_array(
            data=corrected_opd_d, mask=phase_screen_m)

        return SimulationResults(hdr, reconstructed_phase,
                                 meas_petals, input_opd, corrected_opd)

    @property
    def time_step(self):
        return self._time_step

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

    def petals(self):
        '''
        Estimated petals

        Returns the estimated petals at every temporal step
        The global piston is removed, i.e. mean(petals, axis=1) == 0 

        Returns
        -------
        petals: numpy array (n_iter, 6)
            esitmated petals as a function of time [nm]

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

    def _animate_generic(self, what, label, step, cmap='twilight', **kwargs):
        Gif2DMapsAnimator(
            os.path.join(self._jpg_root, label),
            what,
            deltat=self.time_step,
            pixelsize=self.pixelsize, **kwargs).make_gif(step=step, cmap=cmap)

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

    def plot_petals(self):
        t = np.arange(0,
                      self._niter * self.time_step,
                      self.time_step)

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
        pc = PetalComputer(reconstructed_phase_map, self.rot_angle)
        pp = pc.estimated_petals_zero_mean
        return pp, pc.all_jumps

    def opd_correction_from_reconstructed_phase_ave(self):
        return LongExposureSimulation.opd_correction(self.reconstructed_phase_ave(),
                                                     self.rot_angle,
                                                     self.input_opd().shape[1])

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
