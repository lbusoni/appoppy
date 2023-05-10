import numpy as np
import astropy.units as u
from pathlib import Path
import os
from appoppy.ao_residuals import AOResidual
from appoppy.petalometer import Petalometer
from appoppy.gif_animator import Gif2DMapsAnimator
from astropy.io.fits.header import Header
from astropy.io import fits
import matplotlib.pyplot as plt


class LongExposurePetalometer():

    def __init__(self,
                 tracking_number,
                 rot_angle=10,
                 jpeg_root_folder=None):
        self._niter = 1000
        self._start_from_step = 100
        self._rot_angle = rot_angle
        if jpeg_root_folder is None:
            home = str(Path.home())
            jpeg_root_folder = os.path.join(
                home, 'appoppy_anim', tracking_number)
        self._jpg_root = jpeg_root_folder
        self._phase_diff_cumave = None
        self._phase_diff = None
        self._meas_petals = None
        self._meas_petals_no_global_pist = None
        self._meas_petals_cumave = None
        self._pixelsize = None
        self._tracking_number = tracking_number
        self._aores = AOResidual(self._tracking_number)

    def run(self):
        self._pet = Petalometer(
            use_simulated_residual_wfe=True,
            tracking_number=self._tracking_number,
            petals=np.array([0, 0, 0, 0, 0, 0]) * u.nm,
            residual_wavefront_start_from=self._start_from_step,
            rotation_angle=self._rot_angle)
        self._pixelsize = self._pet.pixelsize.to_value(u.m)
        sh = (self._niter,
              self._pet.phase_difference_map.shape[0],
              self._pet.phase_difference_map.shape[1])
        self._phase_diff = np.ma.zeros(sh)
        self._meas_petals = np.zeros((self._niter, 6))
        self._pet.set_step_idx(0)
        for i in range(self._niter):
            print("step %d" % self._pet._step_idx)
            self._pet.sense_wavefront_jumps()
            self._meas_petals[self._pet._step_idx] = self._pet.error_petals
            self._phase_diff[self._pet._step_idx] = self._pet.phase_difference_map
            self._pet.advance_step_idx()

    def phase_screen(self):
        return self._aores.phase_screen

    def phase_screen_cumave(self):
        return self._aores.phase_screen_cumave

    def phase_screen_ave(self):
        return self._aores.phase_screen_ave

    def petals(self):
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

    def animate_phase_difference(self, vminmax=(-1100, 1100)):
        Gif2DMapsAnimator(
            os.path.join(self._jpg_root, 'phase_difference'),
            self.phase_difference(),
            deltat=self._aores.time_step,
            pixelsize=self._pixelsize,
            vminmax=vminmax).make_gif(step=20)

    def animate_phase_difference_cumulative_average(self, vminmax=(-1100, 1100)):
        Gif2DMapsAnimator(
            os.path.join(self._jpg_root, 'phase_difference_cum'),
            self.phase_difference_cumave(),
            deltat=self._aores.time_step,
            pixelsize=self._pixelsize,
            vminmax=vminmax).make_gif(step=20)

    def animate_phase_screens_cumulative_average(self, vminmax=(-300, 300)):
        Gif2DMapsAnimator(
            os.path.join(self._jpg_root, 'phase_screen_cum'),
            self.phase_screen_cumave(),
            deltat=self._aores.time_step,
            pixelsize=self._pixelsize,
            vminmax=vminmax).make_gif(step=10, cmap='cividis')

    def animate_phase_screens(self, vminmax=(-300, 300)):
        Gif2DMapsAnimator(
            os.path.join(self._jpg_root, 'phase_screen'),
            self.phase_screen(),
            deltat=self._aores.time_step,
            pixelsize=self._pixelsize,
            vminmax=vminmax).make_gif(step=10, cmap='cividis')

    def save(self, filename):
        hdr = Header()
        hdr['NITER'] = self._niter
        hdr['ROTANG'] = self._rot_angle
        hdr['TRACKNUM'] = self._tracking_number
        hdr['PIXELSZ'] = self._pixelsize
        hdr['STARTSTEP'] = self._start_from_step
        fits.writeto(filename, self.phase_difference().data,
                     header=hdr, overwrite=True)
        fits.append(filename, self.phase_difference().mask.astype(int))
        fits.append(filename, self._meas_petals)

    @staticmethod
    def load(filename):
        hdr = fits.getheader(filename)
        soi = LongExposurePetalometer(hdr['TRACKNUM'],
                                      rot_angle=hdr['ROTANG'])
        soi._niter = hdr['NITER']
        soi._pixelsize = hdr['PIXELSZ']
        soi._start_from_step = hdr['STARTSTEP']
        res_map_d = fits.getdata(filename, 0)
        res_map_m = fits.getdata(filename, 1).astype(bool)
        soi._phase_diff = np.ma.masked_array(data=res_map_d, mask=res_map_m)
        soi._meas_petals = fits.getdata(filename, 2)
        return soi

    def plot_petals(self):
        t = np.arange(0,
                      self._niter * self._aores.time_step,
                      self._aores.time_step)

        dd = self.petals()
        plt.plot(t, dd)
        plt.ylabel("petals [nm]")
        plt.xlabel("time [s]")
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
        pp, jj = self.petals_from_phase_difference_ave()
        pet = Petalometer(
            use_simulated_residual_wfe=False,
            r0=999999,
            petals=pp,
            rotation_angle=self._rot_angle)
        opd = pet._model2.pupil_opd()
        return opd

    def phase_screen_petal_corrected(self):
        return self.phase_screen_ave() - \
            self.phase_correction_from_petalometer()