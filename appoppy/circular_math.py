import numpy as np
from astropy import units as u
from astropy.stats.circstats import circmean


class _CircularMath():

    @classmethod
    def zero_mean(cls, opd_data, wavelength):
        ang_a = cls.to_radians(opd_data, wavelength)
        mu = cls.to_nm(circmean(ang_a), wavelength)
        return cls.difference(opd_data, mu, wavelength)

    @classmethod
    def wrap_around_zero(cls, opd_data, wavelength):
        a = cls.to_radians(opd_data, wavelength)
        return cls.to_nm(np.arctan2(np.sin(a), np.cos(a)), wavelength)

    @staticmethod
    def to_radians(opd_data, wavelength):
        '''
        Convert an optical path difference (opd) expressed in nm into
        phase data in radians

        Parameters
        ----------
        opd_data: astropy.units.Quantity in length units
            optical path difference

        wavelength: astropy.units.Quantity in length units
            wavelength used to convert opd into phase

        Returns
        -------
        opd: float
            opd converted in radians in (-pi, pi) domain
        '''
        dpi = 2 * np.pi
        return opd_data.to_value(u.nm) / wavelength.to_value(u.nm) * dpi

    @staticmethod
    def to_nm(angle_data, wavelength):
        '''
        Convert phase data in radians in opd in nm
        '''
        if isinstance(angle_data, u.Quantity):
            angle_data = angle_data.to_value(u.rad)
        dpi = 2 * np.pi
        return angle_data * wavelength.to(u.nm) / dpi

    @classmethod
    def difference(cls, opd_a, opd_b, wavelength):
        ang_a = cls.to_radians(opd_a, wavelength)
        ang_b = cls.to_radians(opd_b, wavelength)
        diff = ang_a - ang_b
        ret = (diff + np.pi) % (2 * np.pi) - np.pi
        return cls.to_nm(ret, wavelength)

    @classmethod
    def mean(cls, opd_a, wavelength):
        ang_a = cls.to_radians(opd_a, wavelength)
        mu = circmean(ang_a)
        return cls.to_nm(mu, wavelength)


wrap_around_zero = _CircularMath.wrap_around_zero
zero_mean = _CircularMath.zero_mean
to_radians = _CircularMath.to_radians
to_nm = _CircularMath.to_nm
difference = _CircularMath.difference
mean = _CircularMath.mean
