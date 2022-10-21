import matplotlib.pyplot as plt
import numpy as np
import scipy.signal

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

@PSDGenerator
def mcao_residual(f):
    cutf=0.005
    powerlaw = -1.
    return np.where(f<=cutf, 1, f**powerlaw / (cutf**powerlaw))


def main_plot_noise_generators():
    plt.figure(figsize=(8, 8))
    for G in [brownian_noise, pink_noise, white_noise, blue_noise, violet_noise, mcao_residual]:
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



class Demodulation():

    def __init__(self, piston, noise):

        self._noise_amp = noise
        self._piston_amp = piston
        self._dt = 0.001
        self._t=np.arange(0, 1, self._dt)
        self._sz = len(self._t)
        self._r = None 
        
        self._carrier()
        
        self._p=np.zeros_like(self._t)+ self._piston_amp
        
        #self._n=pink_noise(2*self._sz)[0:self._sz]* self._noise_amp
        self._n=mcao_residual(2*self._sz)[0:self._sz]* self._noise_amp
        #self._n=white_noise_10(2*self._sz)[0:self._sz]* self._noise_amp
        self._s=self._p+self._n
        
        # standard AM 
        #self._compute_amplitude_modulated_signal()
        #self._am_demodulation(self._r)
        # piston modulation
        self._compute_piston_modulated_signal()
        self._piston_demodulation(self._r)
        
        #self._modem_fft(self._r)
        #self.print_result()

    def _compute_amplitude_modulated_signal(self):
        self._r = self._p* self._cc + self._n

    def _compute_piston_modulated_signal(self):
        self._wavelen = 2200
        self._pm = 20
        self._r = np.cos(self._nm2rad(self._p+self._n + self._pm * self._ss)) 

    def _nm2rad(self, nm):
        return 2*np.pi/self._wavelen*nm

    def _rad2nm(self, rad):
        return self._wavelen/(2*np.pi)*rad
    
    def _carrier(self):
        self._fm = 100 
        self._cc=np.cos(2*np.pi*self._fm*self._t)
        self._ss=np.sin(2*np.pi*self._fm*self._t)
    
    def _am_demodulation(self, modulated_signal):
        self._rc = modulated_signal * self._cc
        self._rs= modulated_signal * self._ss
        self._estimated_amp = 2*np.sqrt(self._rc.mean()**2 + self._rs.mean()**2)
        self._estimated_phase = np.arctan2(self._rs.mean(), self._rc.mean())
        #self._estimated_amp = 2* rc.mean()

    def _piston_demodulation(self, modulated_signal):
        self._rc = modulated_signal * self._cc
        self._rs= modulated_signal * self._ss
        amp= self._rad2nm(np.arcsin(-2 * self._rs.mean() / self._nm2rad(self._pm)))
        self._estimated_amp = amp
        self._estimated_phase = 0


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
        

    def print_result(self):
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
        


def main_stat(noise, piston=0):
    n = 1000
    res = np.zeros((2, n))
    for i in range(n):
        dd = Demodulation(piston, noise)
        res[:,i] = [dd.s, dd.p]
    print('mean  mean/demod  %s'  % np.mean(res, axis=1)) 
    print('stdev mean/demod  %s'  % np.std(res, axis=1)) 
    return res