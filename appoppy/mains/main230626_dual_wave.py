import numpy as np
from appoppy.mains.main230519_petals_on_mcao import main230519_petalometer_on_MORFEO_residuals_with_LWE


def fractional_wavelength(x, wv):
    return (x % wv) / wv


def synthetic_lambda(wv1, wv2):
    return wv1 * wv2 / np.abs(wv2 - wv1)


def dual_wavelength_opd(meas1, meas2, wv1, wv2):
    '''
    Dual Wavelength OPD reconstruction

    Estimate the OPD from 2 measurements done at 2 wavelength wv1 and wv2
    The measurement at wv1 is periodic with period wv1 (and similar for wv2)

    The DW algorithm allows to extract OPD in the range (-W/2, W/2) where W is
    the synthetic wavelength for wv1 and wv2

    The input measurements meas1 and meas2 must be expressed
    in fraction of wavelength.
    E.g. an OPD of 1756 is be measured by the sensor at wv1=1600 as 156, and
    the corresponding meas1 is 156/1600.
    The same OPD at wv2=1500 will give meas2=(1756%1500)/1500=256/1500

    '''
    # see e.g. https://wp.optics.arizona.edu/jcwyant/wp-content/uploads/sites/13/2016/08/ExtendedRangeTwo-WavelengthInterferometry.pdf
    # method 3
    swv = synthetic_lambda(wv1, wv2)
    opd = swv * (meas1 - meas2)
    opd[opd < 0] = opd[opd < 0] + swv

    # swing it back in range [-swv/2, swv2]
    # because we don't like jumps around 0
    opd[opd > (swv / 2)] = opd[opd > (swv / 2)] - swv
    return opd


def main_dw3():
    x = np.arange(-30000, 30000)
    wv1 = 1500
    wv2 = 1600

    # the sensor measure the opd in fraction of wavelength
    ff1 = fractional_wavelength(x, wv1)
    ff2 = fractional_wavelength(x, wv2)

    opd = dual_wavelength_opd(ff1, ff2, wv1, wv2)
    import matplotlib.pyplot as plt
    plt.plot(x, opd)
    return opd


def main_dw3_with_noise(noise_lambda=0.01, wv1=1500, wv2=1600):
    opd = np.arange(-30000, 30000)

    # Assume the sensor measure opd as fractional wavelength
    # i.e. (opd % wv) / wv
    # Add guassian noise of noise_lambda rms
    ff1 = fractional_wavelength(
        opd, wv1) + np.random.randn(len(opd)) * noise_lambda
    ff2 = fractional_wavelength(
        opd, wv2) + np.random.randn(len(opd)) * noise_lambda

    # retrieve opd estimate
    opd_meas = dual_wavelength_opd(ff1, ff2, wv1, wv2)

    # display result
    import matplotlib.pyplot as plt
    plt.figure()
    plt.plot(opd, opd_meas)
    plt.xlabel('real opd [nm]')
    plt.ylabel('measured opd [nm]')
    plt.figure()
    plt.plot(opd[30000 - 12000:30000 + 12000],
             (opd_meas - opd)[30000 - 12000:30000 + 12000])
    plt.xlabel('real opd [nm]')
    plt.ylabel('error [nm]')
    print("opd_meas error: %g rms" % (opd_meas - opd)[29000:31000].std())
    return opd_meas, opd


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
    opd = dual_wavelength_opd(fr1500, fr1600, 1500, 1600)

    # plot difference between dw opd and 24um
    import matplotlib.pyplot as plt
    plt.figure()
    plt.imshow(le24.phase_difference()[NFRAME] - opd)
    plt.colorbar()

    return opd, le1500, le1600, le24
