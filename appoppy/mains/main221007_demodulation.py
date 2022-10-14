import matplotlib.pyplot as plt
import numpy as np
import scipy.signal
from appoppy.maory_residual_wfe import restore_residual_wavefront

def plot_spectrum(s):
    f = np.fft.rfftfreq(len(s))
    plt.loglog(f, np.abs(np.fft.rfft(s)))


def noise_psd(N, psd = lambda f: 1):
        X_white = np.fft.rfft(np.random.randn(N));
        S = psd(np.fft.rfftfreq(N))
        # Normalize S
        S = S / np.sqrt(np.mean(S**2))
        X_shaped = X_white * S;
        return np.fft.irfft(X_shaped);

def PSDGenerator(f):
    return lambda N: noise_psd(N, f)

@PSDGenerator
def white_noise(f):
    return 1;

@PSDGenerator
def blue_noise(f):
    return np.sqrt(f);

@PSDGenerator
def violet_noise(f):
    return f;

@PSDGenerator
def brownian_noise(f):
    return 1/np.where(f == 0, float('inf'), f)

@PSDGenerator
def pink_noise(f):
    return 1/np.where(f == 0, float('inf'), np.sqrt(f))

@PSDGenerator
def white_noise_10(f):
    return np.where(f<0.001, 1, 0);


def main_plot_noise_generators():
    plt.figure(figsize=(8, 8))
    for G in [brownian_noise, pink_noise, white_noise, blue_noise, violet_noise, white_noise_10]:
        plot_spectrum(G(2**14))
    plt.legend(['brownian', 'pink', 'white', 'blue', 'violet', '<10'])
    plt.ylim([1e-3, None]);


class PowerSpectralDensityNotFinished():
    
    def __init__(self, signal, dt):
        self._s = signal
        self._dt = dt
        self._use_welch()
        
    def _use_periodogram(self):
        fs = 1/ self._dt
        self._freqs, self._psd = scipy.signal.periodogram(self._s, fs, scaling='density')

    def _use_welch(self):
        fs = 1/ self._dt
        self._freqs, self._psd = scipy.signal.welch(self._s, fs)

    def _freq_range(self, freq_from, freq_to):
        indexFreqL= np.max(np.argwhere(self._freqs <= freq_from))
        indexFreqH= np.min(np.argwhere(self._freqs >= freq_to))
        return (indexFreqL, indexFreqH)

    def power_from_to(self, freq_from, freq_to):
        (indexFreqL, indexFreqH) = self._freq_range(freq_from, freq_to)
        power= np.sum(self._psd[indexFreqL: indexFreqH]) / self._psd.size
        ampl= np.sqrt(2*power)
        print("Integrated power in [%g, %g) Hz: %g" % (
            self._freqs[indexFreqL], self._freqs[indexFreqH], power))
        print("Corresponding to a spectral component of amplitude %g" % ampl)
        return ampl


    def plot(self):
        plt.semilogy(self._freqs, self._psd)
        plt.xlabel('frequency [Hz]')
        plt.ylabel('PSD [V**2/Hz]')
        plt.show()


class AOResidual():
    
    def __init__(self):
        resscreen, hdr = restore_residual_wavefront()
        self._pxscale = float(hdr['PIXELSCL'])
        self._screens = resscreen
        self._spider_x_coord = 240
        self._valid_y_min = 20 
        self._valid_y_max = 160 
        self._dt = 0.002
        self._t_steps = self._screens.shape[0]
        self._t = np.arange(0, self._t_steps*self._dt, self._dt)
        
    def difference_across_spider_at(self, separation_in_meter, rows=None):
        if rows is None:
            rows = np.arange(self._valid_y_min, self._valid_y_max)
        if np.isscalar(rows):
            rows = np.repeat(rows,2) 
        sep_px = round(0.5 * separation_in_meter / self._pxscale)
        idx_x = (self._spider_x_coord - sep_px, self._spider_x_coord + sep_px)
        two_cols =  self._screens[np.ix_(np.arange(self._t_steps), rows, idx_x)]  
        return two_cols[:,:,1] - two_cols[:,:,0]

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
        ft= np.fft.rfft(signal, norm='ortho', axis=0)
        power_spectrum= np.mean(np.abs(ft**2), axis=1)
        frequencies = np.fft.rfftfreq(signal.shape[0], d=dt)
        return power_spectrum, frequencies

    def _structure_function_sep(self, sep_px, a_moment=None, wavelength_nm=500):
        if a_moment is None:
            a_moment = 500
        scr2 = np.roll(self._screens, sep_px, axis=2)
        dd = (self._screens[a_moment]-scr2[a_moment])[:,sep_px:]
        return np.mean( (dd*2*np.pi/wavelength_nm)**2) 

    def structure_function(self, a_moment=500):
        '''
        Structure function of residual phase, computed on the full pupil
        '''
        seps_px = np.array([1,2,5,10,20,50,100])
        stf= np.zeros_like(seps_px)
        for i, sep in enumerate(seps_px) :
            stf[i] = self._structure_function_sep(sep, a_moment)
        return stf, seps_px


    def plot_spectrum(self, separation_in_meter=2, **kwargs):
        dres_nm = self.difference_across_spider_at(separation_in_meter, **kwargs)
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
        plt.ylabel(r'$D_{\phi}$ of residual phase $[rad^2]$')
        plt.grid(True)
        plt.legend()
        return stf_rad2_0, stf_rad2_1, stf_x

class Demodulation():

    def __init__(self, piston, noise):

        self._noise_amp = noise
        self._piston_amp = piston
        self._dt = 0.001
        self._t=np.arange(0, 1, self._dt)
        self._sz = len(self._t)
        
        self._carrier()
        
        self._p=np.zeros_like(self._t)+ self._piston_amp
        self._n=pink_noise(2*self._sz)[0:self._sz]* self._noise_amp
        #self._n=white_noise_10(2*self._sz)[0:self._sz]* self._noise_amp
        self._s=self._p+self._n
        self._r = self._p* self._cc + self._n
        
        #self._modem_fft(self._r)
        self._modem(self._r)
        #self._result()
    
    def _carrier(self):
        self._fm = 200 
        self._cc=np.cos(2*np.pi*self._fm*self._t)
        self._ss=np.sin(2*np.pi*self._fm*self._t)
    
    def _modem(self, modulated_signal):
        self._rc = modulated_signal * self._cc
        self._rs= modulated_signal * self._ss
        self._estimated_amp = 2*np.sqrt(self._rc.mean()**2 + self._rs.mean()**2)
        self._estimated_phase = np.arctan2(self._rs.mean(), self._rc.mean())
        #self._estimated_amp = 2* rc.mean()

    def _rfft(self, signal, dt):
        spectrum= np.fft.rfft(signal, norm='ortho')
        frequencies = np.fft.rfftfreq(signal.size, d=dt)
        return spectrum, frequencies

    def _modem_fft(self, modulated_signal):
        self._spe, self._freq = self._rfft(modulated_signal, self._dt)
        self._estimated_amp = self._power_from_to(
            self._spe, self._freq, 0.999*self._fm, 1.001*self._fm)
        self._estimated_phase = 0

    def _power_from_to(self, spe, freq, freq_from, freq_to):
        indexFreqL= np.max(np.argwhere(freq<=freq_from))
        indexFreqH= np.min(np.argwhere(freq>=freq_to))
        power= np.sum(np.absolute(spe[indexFreqL: indexFreqH])**2) / spe.size
        ampl= np.sqrt(2*power)
        print("Integrated power in [%g, %g) Hz: %g" % (
            freq[indexFreqL], freq[indexFreqH], power))
        print("Corresponding to a spectral component of amplitude %g" % ampl)
        return ampl

        
    @property
    def s(self):
        return self._s.mean()

    @property
    def p(self):
        return self._estimated_amp
        

    def _result(self):
        print("estimated amp %g" % self._estimated_amp)
        print("estimated phase %g" % self._estimated_phase)
        print("_p %g" % self._p.mean())
        print("_n %g" % self._n.mean())
        print("_s %g" % self._s.mean())
        print("2 <p*cc**2> %g" % (2*self._p*self._cc**2).mean())
        print("2 <n*cc**2> %g" % (2*self._n*self._cc**2).mean())
        print("std(_n) %g" % self._n.std())
        
    def _plot(self):
        plt.figure()
        plt.plot(self._t, self._s, label='s')
        plt.plot(self._t, self._r, label='modulated piston')
        plt.plot(self._t, self._cc, label='carrier')
        plt.legend()
        


def main_stat(noise):
    n = 1000
    res = np.zeros((2, n))
    for i in range(n):
        dd = Demodulation(1, noise)
        res[:,i] = [dd.s, dd.p]
    print('stdev mean/demod  %s'  % np.std(res, axis=1)) 
    return res