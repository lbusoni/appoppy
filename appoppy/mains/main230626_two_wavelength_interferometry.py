import numpy as np
from appoppy.two_wavelength_interferometry import fractional_wavelength, twi1,\
    twi2


def _compute_measurements(opd, wv1, wv2, eps_m):
    '''
    The single wavelength sensor returns a measure
    the `opd` in fraction of wavelength, i.e. (opd % wv) / wv

    A gaussian noise of stdev eps_m is added to the measurement
    '''
    meas1 = fractional_wavelength(
        opd, wv1) + np.random.randn(len(opd)) * eps_m
    meas2 = fractional_wavelength(
        opd, wv2) + np.random.randn(len(opd)) * eps_m
    return meas1, meas2


def _display_results(opd, opd_meas, unambiguous_range,
                     algo, wv1, wv2, eps_r, eps_m):
    nel2 = int(len(opd) / 2)
    L2 = np.minimum(int(unambiguous_range / 2), nel2)
    rmin, rmax = int(nel2 - 0.9 * L2), int(nel2 + 0.9 * L2)
    err = opd_meas - opd
    err_center_std = err[nel2 - 1000:nel2 + 1000].std()

    import matplotlib.pyplot as plt
    fig, ax1 = plt.subplots()
    ax1.plot(opd, opd_meas, 'b.-', label='opd_meas')
    ax1.set_xlabel('opd [nm]')
    ax1.set_ylabel('measured opd [nm]', color='b')
    ax1.tick_params('y', colors='b')

    ax2 = ax1.twinx()
    ax2.plot(opd[rmin:rmax], err[rmin:rmax], 'r-', label='error')
    ax2.set_ylabel('error [nm]', color='r')
    ax2.tick_params('y', colors='r')

    lines = ax1.get_lines() + ax2.get_lines()
    labels = [line.get_label() for line in lines]
    ax1.legend(lines, labels, loc='best')
    ax1.grid(True)

    title_str0 = str(
        "Algo %s. wv1/2 %g/%g. eps_r %g, eps_m %g\n" %
        (algo, wv1, wv2, eps_r, eps_m))
    title_str1 = str("unambiguous range %g\n" % unambiguous_range)
    title_str2 = str("opd error (in the central 2000 points): %g rms" %
                     err_center_std)
    title = str(title_str0 + title_str1 + title_str2)
    ax1.set_title(title)
    print(title)


def _generic_main(wv1=1500, wv2=1600,
                  eps_m=0, eps_r=0,
                  algo='twi1', opd_max=100000):
    opd = np.arange(-opd_max, opd_max)

    meas1, meas2 = _compute_measurements(opd, wv1, wv2, eps_m)

    if algo == 'twi1':
        opd_meas, ur = twi1(meas1, meas2, wv1, wv2)
    elif algo == 'twi2':
        opd_meas, ur = twi2(meas1, meas2, wv1, wv2, eps_r, eps_m)
    else:
        raise ValueError("Unknown algorithm %s" % algo)

    _display_results(opd, opd_meas, ur, algo, wv1, wv2, eps_r, eps_m)
    return opd_meas, opd


def main_twi1():
    return _generic_main(wv1=1500, wv2=1600, algo='twi1', opd_max=30000)


def main_twi2():
    return _generic_main(wv1=1500, wv2=1600, algo='twi2', opd_max=30000)


def main_twi1_with_noise(eps_m=0.005):
    return _generic_main(wv1=1500, wv2=1600, algo='twi1',
                         opd_max=30000, eps_m=eps_m)


def main_twi2_with_noise(eps_m=0.005, eps_r=0):
    return _generic_main(wv1=1500, wv2=1600, algo='twi2',
                         opd_max=30000, eps_m=eps_m, eps_r=eps_r)


def main_twi2_hene(eps_m=1e-4, eps_r=-1e-7):
    return _generic_main(
        wv1=604.613, wv2=632.816, opd_max=150000,
        algo='twi2', eps_m=eps_m, eps_r=eps_r)


def main_twi2_section_6(eps_m=0.005, eps_r=1e-4):
    return _generic_main(
        wv1=824, wv2=1332, opd_max=30000,
        algo='twi2', eps_m=eps_m, eps_r=eps_r)


def main_twi1_hene(eps_m=1e-4):
    return _generic_main(
        wv1=604.613, wv2=632.816, opd_max=150000,
        algo='twi1', eps_m=eps_m)


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
