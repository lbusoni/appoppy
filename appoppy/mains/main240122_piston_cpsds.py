import numpy as np
import matplotlib.pyplot as plt
import astropy.units as u
from arte.atmo.von_karman_covariance_calculator import VonKarmanSpatioTemporalCovariance
from arte.types.aperture import CircularOpticalAperture
from arte.types.guide_source import GuideSource
from morfeo.utils.constants import ELT
from arte.atmo.cn2_profile import EsoEltProfiles


def covariance_of_differential_piston_anisoplanatism(
        source1, source2, aperture1, aperture2, cn2_profile):
    '''
    Compute the covariance of the differential piston between two circular apertures measured along two different directions. 

    Parameters
    ----------
    source1: arte.types.guide_source.GuideSource
        First source used to measure differential piston between aperture1 and aperture2.

    source2: arte.types.guide_source.GuideSource
        Second source used to measure differential piston between aperture1 and aperture2.

    aperture1: arte.types.aperture.CircularOpticalAperture
        First circular aperture.

    aperture2: arte.types.aperture.CircularOpticalAperture
        Second circular aperture.

    cn2_profile: arte.atmo.cn2_profile.Cn2Profile
        Cn2 profile of atmospheric turbulence.

    '''
    j = 1
    spat_freqs = np.logspace(-4, 4, 100)
    vk = VonKarmanSpatioTemporalCovariance(source1=source1, source2=source1, aperture1=aperture1,
                                           aperture2=aperture1, cn2_profile=cn2_profile,
                                           spat_freqs=spat_freqs)
    var_turb = vk.getZernikeCovariance(j, j)
    vk.setAperture2(aperture2)
    cov_on1_on2 = vk.getZernikeCovariance(j, j)
    vk.setAperture2(aperture1)
    vk.setSource2(source2)
    cov_on1_off1 = vk.getZernikeCovariance(j, j)
    vk.setAperture1(aperture2)
    cov_on2_off1 = vk.getZernikeCovariance(j, j)
    vk.setAperture1(aperture1)
    vk.setAperture2(aperture2)
    cov_on1_off2 = vk.getZernikeCovariance(j, j)
    res = 2 * (2 * var_turb - 2 * cov_on1_off1 - 2 * cov_on1_on2 + cov_on1_off2 + cov_on2_off1)
    wl_in_m = cn2_profile.wavelength()
    rad_to_nm = wl_in_m.to(u.nm) / (2 * np.pi * u.rad)
    return res * rad_to_nm**2


def spectrum_of_differential_piston_anisoplanatism(
        source1, source2, aperture1, aperture2, cn2_profile, temp_freqs):
    '''
    Compute the cross-spectrum of the differential piston between two circular apertures measured along two different directions. 

    Parameters
    ----------
    source1: arte.types.guide_source.GuideSource
        First source used to measure differential piston between aperture1 and aperture2.

    source2: arte.types.guide_source.GuideSource
        Second source used to measure differential piston between aperture1 and aperture2.

    aperture1: arte.types.aperture.CircularOpticalAperture
        First circular aperture.

    aperture2: arte.types.aperture.CircularOpticalAperture
        Second circular aperture.

    cn2_profile: arte.atmo.cn2_profile.Cn2Profile
        Cn2 profile of atmospheric turbulence.

    temp_freqs: numpy.ndarray
        Array of temporal frequency points to evaluate the cross-spectrum.
    '''
    j = 1
    spat_freqs = np.logspace(-4, 4, 100)
    vk = VonKarmanSpatioTemporalCovariance(source1=source1, source2=source1, aperture1=aperture1,
                                           aperture2=aperture1, cn2_profile=cn2_profile,
                                           spat_freqs=spat_freqs)
    psd_turb = vk.getGeneralZernikeCPSD(j, j, temp_freqs)
    vk.setAperture2(aperture2)
    cpsd_on1_on2 = vk.getGeneralZernikeCPSD(j, j, temp_freqs)
    vk.setAperture2(aperture1)
    vk.setSource2(source2)
    cpsd_on1_off1 = vk.getGeneralZernikeCPSD(j, j, temp_freqs)
    vk.setAperture1(aperture2)
    cpsd_on2_off1 = vk.getGeneralZernikeCPSD(j, j, temp_freqs)
    vk.setAperture1(aperture1)
    vk.setAperture2(aperture2)
    cpsd_on1_off2 = vk.getGeneralZernikeCPSD(j, j, temp_freqs)
    res = 2 * (2 * psd_turb - 2 * cpsd_on1_off1 - 2 * cpsd_on1_on2 + cpsd_on1_off2 + cpsd_on2_off1)
    wl_in_m = cn2_profile.wavelength()
    rad_to_nm = wl_in_m.to(u.nm) / (2 * np.pi * u.rad)
    return res * rad_to_nm**2


def segment(theta_in_deg):
    '''
    Each segment of the ELT pupil is approximated with a circular pupil inscribed in the segment itself.
    The cartesian coordinates of the pupil center are evaluated as: 
        (rho*cos(theta), rho*sin(theta))
    with:
        rho = R/2 + r/2
    where R is the radius of the ELT pupil and r is the radius of the central obstruction.
    The radius of the circular pupil is computed as:
        rho * sin(30º) = rho / 2.

    Parameters
    ----------
    theta_in_deg: astropy.quantity.Quantity
        Angle of segment center in degrees.
    '''
    R = ELT.pupil_diameter / 2
    r = ELT.central_obstruction_diameter / 2
    rho = R / 2 + r / 2
    cartesian_crds = [rho * np.cos(theta_in_deg), rho * np.sin(theta_in_deg), 0 * u.m]
    radius = rho / 2
    return [radius, cartesian_crds]


def main_spectrum_of_differential_piston_anisoplanatism(
        source1_crds = (0, 0),
        source2_crds = (55, 0)):
    '''
    Parameters
    ----------
    source1_crds: tuple
        Coordinates in (arcsec, degrees) of the first source used to measure differential piston between two segments.

    source2_crds: tuple
        Coordinates in (arcsec, degrees) of the second source used to measure differential piston between two segments.
    '''
    thetas = [0, 60, -60] * u.deg
    apertures = [CircularOpticalAperture(
        segment(th)[0], segment(th)[1]) for th in thetas]
    cn2_profile = EsoEltProfiles.Median()
    cn2_profile.set_zenith_angle(30 * u.deg)
    temp_freqs = np.logspace(-3, 3, 1000)
    cpsd_1 = spectrum_of_differential_piston_anisoplanatism(
        GuideSource(source1_crds, np.inf), GuideSource(source2_crds, np.inf),
        apertures[0], apertures[1], cn2_profile, temp_freqs)
    cpsd_2 = spectrum_of_differential_piston_anisoplanatism(
        GuideSource(source1_crds, np.inf), GuideSource(source2_crds, np.inf),
        apertures[0], apertures[2], cn2_profile, temp_freqs)
    cpsd_3 = spectrum_of_differential_piston_anisoplanatism(
        GuideSource(source1_crds, np.inf), GuideSource(source2_crds, np.inf),
        apertures[1], apertures[2], cn2_profile, temp_freqs)
    plt.figure()
    plt.loglog(temp_freqs, abs(cpsd_1), label='Segments at (0, 60) deg')
    plt.loglog(temp_freqs, abs(cpsd_2), label='Segments at (0, -60) deg')
    plt.loglog(temp_freqs, abs(cpsd_3), label='Segments at (60, -60) deg')
    plt.grid()
    plt.xlabel('Temporal frequency [Hz]')
    plt.ylabel('Spectrum of differential piston anisoplanatism [nm$^{2}$ / Hz]')
    plt.legend()
    plt.title('Off axis star in (%s", %s deg)' % (source2_crds[0], source2_crds[1]))
    print('Petal error for segments at (0, 60) deg: %s' % (
        np.sqrt(np.trapz(cpsd_1, temp_freqs*u.Hz))).real)
    print('Petal error for segments at (0, -60) deg: %s' % (
        np.sqrt(np.trapz(cpsd_2, temp_freqs*u.Hz))).real)
    print('Petal error for segments at (60, -60) deg: %s' % (
        np.sqrt(np.trapz(cpsd_3, temp_freqs*u.Hz))).real)