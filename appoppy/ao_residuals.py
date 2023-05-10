import numpy as np
import matplotlib.pyplot as plt
from appoppy.maory_residual_wfe import restore_residual_wavefront
from appoppy.mask import sector_mask


class AOResidual():

    def __init__(self,
                 tracking_number,
                 start_from=100):
        self._start_from = start_from
        resscreen, hdr = restore_residual_wavefront(tracking_number)
        self._pxscale = float(hdr['PIXELSCL'])
        self._screens = resscreen[self._start_from:, :, :]
        self._shape = self._screens.shape
        self._spider_x_coord = 240
        self._valid_y_min = 20
        self._valid_y_max = 160
        self._dt = hdr['TIME_STEP']
        self._t_steps = self._screens.shape[0]
        self._t = np.arange(0, self._t_steps * self._dt, self._dt)

        self._phase_screens_cumave = None
        self._phase_screens_ave = None
        self._phase_screens_no_global_pist = None
        self._petals_median = None
        self._spider_jumps = None

    @property
    def time_step(self):
        return self._dt

    def difference_across_spider_at(self, separation_in_meter, rows=None):
        raise Exception("Fix it - it assumes a 480 px pupil")
        if rows is None:
            rows = np.arange(self._valid_y_min, self._valid_y_max)
        if np.isscalar(rows):
            rows = np.repeat(rows, 2)
        sep_px = round(0.5 * separation_in_meter / self._pxscale)
        idx_x = (self._spider_x_coord - sep_px, self._spider_x_coord + sep_px)
        two_cols = self._screens[np.ix_(np.arange(self._t_steps), rows, idx_x)]
        return two_cols[:, :, 1] - two_cols[:, :, 0]

    def _rfft(self, signal, dt):
        '''
        Return power spectrum of input array

        Parameters
        ----------
        signal: np.array
            axis=0 is the temporal dimension
            axis=1 is a spatial dimension.

        Returns
        -------
        temporal power spectrum of signal, averaged along the spatial dimension
        '''
        if isinstance(signal, np.ma.MaskedArray):
            if np.any(signal.mask == True):
                raise Exception('masked signal. fft output is unreliable')
            signal = signal.filled()
        ft = np.fft.rfft(signal, norm='ortho', axis=0)
        power_spectrum = np.mean(np.abs(ft**2), axis=1)
        frequencies = np.fft.rfftfreq(signal.shape[0], d=dt)
        return power_spectrum, frequencies

    def _structure_function_sep(self, sep_px, a_moment=None):
        if a_moment is None:
            a_moment = 500
        scr2 = np.roll(self._screens, sep_px, axis=2)
        dd = (self._screens[a_moment] - scr2[a_moment])[:, sep_px:]
        return np.mean(dd**2)

    def structure_function(self, a_moment=500):
        '''
        Structure function of residual phase, computed on the full pupil

        $D_{\phi}(\rho) = average( (\phi(r+\rho)-\phi(r))^2) )

        Returns
        -------
        stf, rho: tuple of `numpy.array`
            First element contains the structure function in nm**2
            Second elements contains the separation at which the
            structure function has been computed in m

        '''
        seps_px = np.array([1, 2, 5, 10, 20, 50, 100])
        stf = np.zeros_like(seps_px)
        for i, sep in enumerate(seps_px):
            stf[i] = self._structure_function_sep(sep, a_moment)
        return stf, seps_px

    def plot_spectrum(self, separation_in_meter=2, **kwargs):
        dres_nm = self.difference_across_spider_at(
            separation_in_meter, **kwargs)
        dres_sp, dres_fr = self._rfft(dres_nm, self._dt)
        plt.loglog(dres_fr, dres_sp, label='AO residual difference')
        plt.ylabel(r'Power spectrum of $D_{\phi}$ of AO residuals at 2m [AU]')
        plt.xlabel('Frequency [Hz]')
        plt.grid(True)

    def plot_structure_function(self):
        stf_rad2_0, stf_x = self.structure_function(a_moment=0)
        stf_rad2_1, stf_x = self.structure_function(a_moment=500)
        plt.figure()
        plt.semilogy(stf_x * self._pxscale, stf_rad2_0, label='t=0s')
        plt.semilogy(stf_x * self._pxscale, stf_rad2_1, label='t=1s')
        plt.xlabel(r'separation $\rho$  [m]')
        plt.ylabel(r'$D_{\phi}$ of residual phase $[nm^2]$')
        plt.grid(True)
        plt.legend()
        return stf_rad2_0, stf_rad2_1, stf_x

    def _cumaverage(self, cube):
        cs = cube.cumsum(axis=0)
        res = np.array([cs[i] / (i + 1) for i in np.arange(cube.shape[0])])
        return res

    @property
    def phase_screen(self):
        if self._phase_screens_no_global_pist is None:
            dd = self._screens.mean(axis=(1, 2))
            self._phase_screens_no_global_pist = self._screens - \
                np.rollaxis(
                    np.tile(dd, (self._shape[1], self._shape[2], 1)), 2)
        return self._phase_screens_no_global_pist

    @property
    def phase_screen_cumave(self):
        if self._phase_screens_cumave is None:
            self._phase_screens_cumave = self._cumaverage(self.phase_screen)
        return self._phase_screens_cumave

    @property
    def phase_screen_ave(self):
        if self._phase_screens_ave is None:
            self._phase_screens_ave = self.phase_screen.mean(axis=0)
        return self._phase_screens_ave

    def mask_phase_screen(self, angle_range):
        smask1 = sector_mask(self.phase_screen[0].shape,
                             (angle_range[0], angle_range[1]))
        mask = np.ma.mask_or(self.phase_screen[0].mask, ~smask1)
        return np.ma.masked_array(
            self.phase_screen, mask=np.broadcast_to(
                mask, self.phase_screen.shape))

    def _mask_sector(self, ifgram, angle_range):
        smask1 = sector_mask(ifgram.shape,
                             (angle_range[0], angle_range[1]))
        mask = np.ma.mask_or(ifgram.mask, ~smask1)
        return np.ma.masked_array(ifgram, mask=mask)

    @property
    def petals(self):
        if self._petals_median is None:
            angs = (90, 30, -30, -90, -150, -210, -270)
            self._petals_median = np.zeros((self.phase_screen.shape[0], 6))
            for i in range(6):
                psm = self.mask_phase_screen((angs[i + 1], angs[i]))
                self._petals_median[:, i] = np.ma.median(psm, axis=(1, 2))
        return self._petals_median

    @property
    def petals_average(self):
        mpet = np.zeros(6)
        for i, a in enumerate([30, -30, -90, -150, -210, -270]):
            mpet[i] = np.ma.median(
                self._mask_sector(self.phase_screen_ave, (a, a + 60)))
        return mpet - mpet.mean()

    @property
    def spider_jumps_average(self):
        if self._spider_jumps is None:
            r = 10
            ifgram = self.phase_screen_ave
            self._spider_jumps = self._compute_spider_jumps_average(ifgram, r)
        return self._spider_jumps

    def _compute_spider_jumps_average(self, ifgram, r):
        angs = (90 + r, 90, 90 - r, 30 + r, 30, 30 - r, -30 + r, -30, -30 - r,
                -90 + r, -90, -90 - r, -150 + r, -150, -150 - r,
                -210 + r, -210, -210 - r, -270 + r, -270, -270 - r)
        jumps = np.zeros(len(angs) - 1)
        for i in range(len(angs) - 1):
            psm = self._mask_sector(ifgram, (angs[i + 1], angs[i]))
            jumps[i] = np.ma.median(psm)
        return jumps
