import numpy as np
import astropy.units as u
from appoppy.long_exposure_simulation import LongExposureSimulation

ROOT_DIR = '/Users/giuliacarla/Documents/INAF/Lavoro/Progetti/MORFEO/Petalometer/CiaoCiaoWFS/analysis/data/from_appoppy/'
ROOT_DIR = '/Users/lbusoni/Library/CloudStorage/GoogleDrive-lorenzo.busoni@inaf.it/Il mio Drive/adopt/Varie/CiaoCiaoWFS/analysis/data/from_appoppy/'

TN_LWE = '20230517_160708.0_coo0.0_0.0'
TN_P10 = '20210512_081313.0_coo'
TN_P50 = '20210511_144618.0_coo'


def main230607_long_exposure_MORFEO_residuals_and_small_petals():
    le = LongExposureSimulation(
        tracking_number='20210511_144618.0_coo0.0_0.0', rot_angle=60,
        petals=np.array([200, 30, -100, 370, 500, 0]) * u.nm,
        wavelength=24e-6 * u.m)
    le.run()
    return le


def main230607_long_exposure_MORFEO_residuals_and_large_petals():
    le = LongExposureSimulation(
        tracking_number='20210511_144618.0_coo0.0_0.0', rot_angle=60,
        petals=np.array([1200, -1000, 3000, 370, 1500, 0]) * u.nm,
        wavelength=24e-6 * u.m)
    le.run()
    return le


def main230609_long_exposure_MORFEO_residuals_and_petals_for_plot():
    le = LongExposureSimulation(
        tracking_number='20210511_144618.0_coo0.0_0.0', rot_angle=20,
        petals=np.array([500, 400, -200, 300, -100, 0]) * u.nm,
        wavelength=24e-6 * u.m)
    le.run()
    return le


def main230623_long_exposure_MORFEO_residuals_and_LWE():
    le = LongExposureSimulation(
        tracking_number=TN_LWE, rot_angle=60,
        wavelength=24e-6 * u.m)
    le.run()
    return le


def main230626_long_exposure_MORFEO_residuals_and_LWE_1600():
    le = LongExposureSimulation(
        tracking_number=TN_LWE, rot_angle=60,
        wavelength=1.60e-6 * u.m)
    le.run()
    return le


def main230626_long_exposure_MORFEO_residuals_and_LWE_1500():
    le = LongExposureSimulation(
        tracking_number=TN_LWE, rot_angle=60,
        wavelength=1.50e-6 * u.m)
    le.run()
    return le


def main230627_long_exposure_MORFEO_residuals_and_one_petals():
    le = LongExposureSimulation(
        tracking_number='20210511_144618.0_coo0.0_0.0', rot_angle=60,
        petals=np.array([800, 0, 0, 0, 0, 0]) * u.nm,
        wavelength=2.2e-6 * u.m)
    le.run()
    return le


def main231213_long_exposure_MORFEO_residuals_no_petals(
        wavelength=2.2e-6 * u.m):
    le = LongExposureSimulation(
        tracking_number='20231209_202232.0_coo55.0_0.0', rot_angle=60,
        wavelength=wavelength)
    le.run()
    return le


def main231213_long_exposure_MORFEO_residuals_one_100nm_piston(
        wavelength=2.2e-6 * u.m):
    le = LongExposureSimulation(
        tracking_number='20231209_202232.0_coo55.0_0.0', rot_angle=60,
        petals=np.array([0, 0, 0, 0, 100, 0]) * u.nm,
        wavelength=wavelength)
    le.run()
    return le
