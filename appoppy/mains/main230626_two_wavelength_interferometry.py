import numpy as np
from appoppy.two_wavelength_interferometry import fractional_wavelength, twi1


def main_twi1():
    x = np.arange(-30000, 30000)
    wv1 = 1500
    wv2 = 1600

    # the sensor measure the opd in fraction of wavelength
    meas1 = fractional_wavelength(x, wv1)
    meas2 = fractional_wavelength(x, wv2)

    opd = twi1(meas1, meas2, wv1, wv2)
    import matplotlib.pyplot as plt
    plt.plot(x, opd)
    return opd


def main_twi1_with_noise(noise_lambda=0.01, wv1=1500, wv2=1600):
    opd = np.arange(-30000, 30000)

    # Assume the sensor measure opd as fractional wavelength
    # i.e. (opd % wv) / wv
    # Add guassian noise of noise_lambda rms
    ff1 = fractional_wavelength(
        opd, wv1) + np.random.randn(len(opd)) * noise_lambda
    ff2 = fractional_wavelength(
        opd, wv2) + np.random.randn(len(opd)) * noise_lambda

    # retrieve opd estimate
    opd_meas = twi1(ff1, ff2, wv1, wv2)

    # display result
    import matplotlib.pyplot as plt
    plt.figure()
    plt.plot(opd, opd_meas)
    plt.xlabel('opd [nm]')
    plt.ylabel('measured opd [nm]')
    plt.figure()
    plt.plot(opd[30000 - 12000:30000 + 12000],
             (opd_meas - opd)[30000 - 12000:30000 + 12000])
    plt.xlabel('opd [nm]')
    plt.ylabel('error [nm]')
    print("opd_meas error (in the central 2000 points): %g rms" %
          (opd_meas - opd)[29000:31000].std())
    return opd_meas, opd


def main_twi1_with_ciaciao_data():
    from appoppy.mains.main230519_petals_on_mcao import \
        main230519_petalometer_on_MORFEO_residuals_with_LWE

    le1500, _, _, _ = \
        main230519_petalometer_on_MORFEO_residuals_with_LWE(
            wv_in_um=1.5)
    le1600, _, _, _ = \
        main230519_petalometer_on_MORFEO_residuals_with_LWE(
            wv_in_um=1.6)
    le24, _, _, _ = main230519_petalometer_on_MORFEO_residuals_with_LWE(
        wv_in_um=24)
    NFRAME = 100
    fr1500 = le1500.phase_difference()[NFRAME] / 1500
    fr1600 = le1600.phase_difference()[NFRAME] / 1600
    opd = twi1(fr1500, fr1600, 1500, 1600)

    # plot difference between dw opd and 24um
    import matplotlib.pyplot as plt
    plt.figure()
    plt.imshow(le24.phase_difference()[NFRAME] - opd)
    plt.colorbar()

    return opd, le1500, le1600, le24


def main_twi2_hene(p=343, q=350, k=157, eps_i=0.001):
    opd = np.arange(-300000, 300000)
    wv1 = 604.613
    wv2 = 632.816

    # p = 343
    # q = 359
    # k = 157
    eps_r = 0  # -1.55e-6
    # eps_i = 0.0001

    meas1 = fractional_wavelength(
        opd, wv1) + np.random.randn(len(opd)) * eps_i
    meas2 = fractional_wavelength(
        opd, wv2) + np.random.randn(len(opd)) * eps_i

    def wrap(x):
        return x - np.round(x)

    def find_p_q(rl_hat):
        q = np.arange(2, 1000)
        p = np.round(rl_hat * q)
        coprime_idx = np.nonzero(np.gcd(p.astype(int), q.astype(int)) == 1)
        p = p[coprime_idx]
        q = q[coprime_idx]
        eps_f = rl_hat - p / q
        return p, q, eps_f

    def find_k(p, q):
        k = np.zeros(q.shape)
        for i in range(len(q)):
            ks = np.arange(0, q[i])
            k[i] = ks[np.nonzero(ks * p[i] % q[i] == 1)][0]
        return k

    def twi2(meas1, meas2, wv1, wv2, p, q, k, eps_r):
        print("UR: %g" % (q * wv1))
        rl = wv1 / wv2
        rl_hat = rl + eps_r
        m1barhat = q * wrap(k / q * np.round(q * (meas2 - rl_hat * meas1)))
        opd = wv1 * (m1barhat + meas1)

        eps_f = rl_hat - (p / q)
        eps_margin = (1 - q**2 * (np.abs(eps_f) +
                                  np.abs(eps_r))) / (2 * (p + q))
        return opd, eps_f, eps_margin

    res = twi2(meas1, meas2, wv1, wv2, p, q, k, eps_r)
    opd_meas = res[0]

    import matplotlib.pyplot as plt
    plt.plot(opd, opd_meas)
    return opd_meas, opd, res


def main_twi1_hene():
    opd = np.arange(-30000, 30000)
    wv1 = 604.613
    wv2 = 632.816

    meas1 = fractional_wavelength(opd, wv1)
    meas2 = fractional_wavelength(opd, wv2)

    opd_meas = twi1(meas1, meas2, wv1, wv2)
    import matplotlib.pyplot as plt
    plt.plot(opd, opd_meas)
    return opd_meas, opd
