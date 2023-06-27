import numpy as np
from appoppy.mains.main230519_petals_on_mcao import main230519_petalometer_on_MORFEO_residuals_with_LWE


def fractional_fringe(x, wv):
    return (x % wv) / wv


def synthetic_lambda(wv1, wv2):
    return wv1 * wv2 / np.abs(wv2 - wv1)


def dual_wavelength_3(ff1, ff2, wv1, wv2):
    # see e.g. https://wp.optics.arizona.edu/jcwyant/wp-content/uploads/sites/13/2016/08/ExtendedRangeTwo-WavelengthInterferometry.pdf
    # method 3
    swv = synthetic_lambda(wv1, wv2)
    opd = swv * (ff1 - ff2)
    opd[opd < 0] = opd[opd < 0] + swv

    # swing it back in range [-swv/2, swv2]
    # because we don't like jumps around 0
    opd[opd > (swv / 2)] = opd[opd > (swv / 2)] - swv
    return opd


def main_dw3():
    x = np.arange(-30000, 30000)
    wv1 = 1500
    wv2 = 1600

    # the sensor measure the fractional fringe
    ff1 = fractional_fringe(x, wv1)
    ff2 = fractional_fringe(x, wv2)

    opd = dual_wavelength_3(ff1, ff2, wv1, wv2)
    import matplotlib.pyplot as plt
    plt.plot(x, opd)
    return opd


def main_with_ciaciao_data():
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
    opd = dual_wavelength_3(fr1500, fr1600, 1500, 1600)

    # plot difference between dw opd and 24um
    import matplotlib.pyplot as plt
    plt.figure()
    plt.imshow(le24.phase_difference()[NFRAME] - opd)
    plt.colorbar()

    return opd, le1500, le1600, le24
