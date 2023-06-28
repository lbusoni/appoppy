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

    The DW algorithm allows to extract OPD in the range (-W/2, W/2) where W is
    the synthetic wavelength for wv1 and wv2

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
    swv = synthetic_lambda(wv1, wv2)
    opd = swv * (meas1 - meas2)
    opd[opd < 0] = opd[opd < 0] + swv

    # swing it back in range [-swv/2, swv2] because we like it more
    opd[opd > (swv / 2)] = opd[opd > (swv / 2)] - swv
    return opd


def twi2(meas1, meas2, wv1, wv2):
    pass
