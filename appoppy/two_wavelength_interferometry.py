import numpy as np


def fractional_wavelength(x, wv):
    return (x % wv) / wv


def synthetic_lambda(wv1, wv2):
    return wv1 * wv2 / np.abs(wv2 - wv1)


def twi1(meas1, meas2, wv1, wv2):
    '''
    Two Wavelength Interferometry OPD reconstruction

    Estimate the OPD from 2 measurements done at 2 wavelength wv1 and wv2
    It is assumed that sensor 1 and sensor 2 have a perfectly linear
    response to the OPD in the range [0, wv1) and [0, wv2) respectively.
    The measurement of sensor 1 is periodic with period wv1 (and similar for
    sensor 2)

    The `twi1` algorithm allows to extract OPD in the range (-UR/2, UR/2)
    where UR is the Unambiguous Range corresponding to the
    beat-wavelength for wv1 and wv2 i.e. wv1*wv2/(wv2-wv1)

    The input measurements meas1 and meas2 must be expressed
    in fraction of wavelength i.e. meas = (opd % wv) / wv
    E.g. an OPD of 1756 is be measured by the sensor at wv1=1500 as 256, and
    the corresponding meas1 is 256/1600.
    The same OPD at wv2=1600 will give meas2=(1756%1600)/1600=156/1600

    Args:
    ----------
    meas1, meas2: float or float-array
        Measurement of the opd at wv1 and wv2 respectively.
        They must be expressed as fractional wavelength (see above)

    wv1, wv2: float
        measurement wavelength of meas1 and meas2
        They must be expressed in the same length unit of the return value OPD.
        wv2 must be greater than wv1.

    Returns:
    -------
    opd: float or float-array
        the DW estimated OPD in meas1 units in the range [-W, W)
        with W synthetic_lambda(wv1, wv2)

    ur: float
        the unambiguous range in wv1 units


    Reference:
    ---------
    Algorithm TWI-1 in
    Kamel Houairi and Frédéric Cassaing, "Two-wavelength interferometry:
    extended range and accurate optical path difference analytical estimator,"
    J. Opt. Soc. Am. A 26, 2503-2511 (2009).

    '''
    if wv1 >= wv2:
        raise ValueError("wv1 must be smaller than wv2")

    # see e.g. https://wp.optics.arizona.edu/jcwyant/wp-content/uploads/sites/13/2016/08/ExtendedRangeTwo-WavelengthInterferometry.pdf
    # method 3
    ur = synthetic_lambda(wv1, wv2)
    opd = ur * (meas1 - meas2)
    opd[opd < 0] = opd[opd < 0] + ur

    # swing it back in range [-ur/2, ur/2] because we like it more
    opd[opd > (ur / 2)] = opd[opd > (ur / 2)] - ur

    return opd, ur


def _wrap(x):
    return x - np.round(x)


def _compute_p_q(rl_hat):
    q_max = 1000
    q = np.arange(2, q_max)
    p = np.round(rl_hat * q)
    coprime_idx = np.nonzero(np.gcd(p.astype(int), q.astype(int)) == 1)
    p = p[coprime_idx]
    q = q[coprime_idx]
    eps_f = rl_hat - p / q
    return p, q, eps_f


def _compute_k(p, q):
    q1 = np.atleast_1d(q)
    p1 = np.atleast_1d(p)
    n_el = len(q1)
    k = np.zeros(n_el, dtype=int)
    for i in range(n_el):
        ks = np.arange(0, q1[i], dtype=int)
        k[i] = ks[np.nonzero(ks * p1[i] % q1[i] == 1)][0]
    return k[0] if n_el == 1 else k


def _maxeps(p, q, eps_f, eps_r, eps_m):
    maxeps = (q**2 * np.abs(eps_f) +
              q**2 * np.abs(eps_r) +
              2 * (p + q) * np.abs(eps_m) +
              q * np.abs(eps_r)) * 0.5
    return maxeps


def _find_optimal_q(rl_hat, eps_r, eps_m):
    TOLERABLE_EPS = 0.4
    p, q, eps_f = _compute_p_q(rl_hat)
    maxeps = _maxeps(p, q, eps_f, eps_r, eps_m)
    idx = np.nonzero(maxeps < TOLERABLE_EPS)[0].max()
    return p[idx], q[idx], eps_f[idx], maxeps[idx]


def _fix_numerical_rounding(opd_meas, ur):
    opd = opd_meas.copy()
    opd[opd_meas > (ur / 2)] = opd_meas[opd_meas > (ur / 2)] - ur
    opd[opd_meas < (-ur / 2)] = opd_meas[opd_meas < (-ur / 2)] + ur
    return opd


def twi2(meas1, meas2, wv1, wv2, eps_r=0, eps_m=1e-4):
    '''
    Two Wavelength Interferometry OPD reconstruction with extended unambigous range (UR)

    Estimate the OPD from 2 measurements done at 2 wavelength wv1 and wv2
    It is assumed that sensor 1 and sensor 2 have a perfectly linear
    response to the OPD in the range [0, wv1) and [0, wv2) respectively.
    The measurement of sensor 1 is periodic with period wv1 (and similar for
    sensor 2)

    The `twi2` algorithm optimize the UR taking into account the meausurement
    errors and the wavelength calibration errors. With the proper choice of wv1
    and wv2 and keeping the errors well constrained one can obtain an UR much
    larger than the `synthetic_lambda` of the algorithm `twi1`

    The input measurements meas1 and meas2 must be expressed
    in fraction of wavelength i.e. meas = (opd % wv) / wv
    E.g. an OPD of 1756 is be measured by the sensor at wv1=1500 as 256, and
    the corresponding meas1 is 256/1600.
    The same OPD at wv2=1600 will give meas2=(1756%1600)/1600=156/1600

    Args:
    ----------
    meas1, meas2: float or float-array
        Measurement of the opd at wv1 and wv2 respectively.
        They must be expressed as fractional wavelength (see above)

    wv1, wv2: float
        measurement wavelength of meas1 and meas2
        They must be expressed in the same length unit of the return value OPD.
        wv2 must be greater than wv1.

    eps_r: float
        error in the estimate of the ratio wv1/wv2 (e.g. wavelength
        calibration errors) Default to 0

    eps_m: float
        error in the estimate of meas1 and meas2 (e.g. single-wavelength
        OPD errors) Default to 0


    Returns:
    -------
    opd: float or float-array
        the DW estimated OPD in meas1 units in the range [-UR, UR)

    UR: float
        the unambiguous range in wv1 units


    Reference:
    ---------
    Implements algorithm TWI-2 in
    Kamel Houairi and Frédéric Cassaing, "Two-wavelength interferometry:
    extended range and accurate optical path difference analytical estimator,"
    J. Opt. Soc. Am. A 26, 2503-2511 (2009).
    '''
    rl = wv1 / wv2
    rl_hat = rl + eps_r
    p, q, eps_f, max_eps = _find_optimal_q(rl_hat, eps_r, eps_m)
    k = _compute_k(p, q)
    ur = q * wv1

    eps_margin = (1 - q**2 * (np.abs(eps_f) +
                              np.abs(eps_r))) / (2 * (p + q))
    print("p, q, k, eps_f, eps_margin, max_eps: %d %d %d %g %g %g" % (
        p, q, k, eps_f, eps_margin, max_eps))

    m1barhat = q * _wrap(k / q * np.round(q * (meas2 - rl_hat * meas1)))
    opd = wv1 * (m1barhat + meas1)

    opd_r = _fix_numerical_rounding(opd, ur)

    return opd_r, ur
