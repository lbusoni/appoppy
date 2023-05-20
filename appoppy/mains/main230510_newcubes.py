from appoppy.long_exposure import LongExposurePetalometer
import os
from pathlib import Path


def main_save():
    tracking_number = '20221026_123454.0_coo0.0_0.0'
    lep = LongExposurePetalometer(tracking_number,
                                  rot_angle=10,
                                  start_from_step=100,
                                  n_iter=20)
    lep.run()
    lep.save(os.path.join(
        str(Path.home()), 'appoppy', tracking_number, 'lep.fits'))
    return lep


def main_load(tracking_number='20221026_123454.0_coo0.0_0.0'):
    filename = os.path.join(
        str(Path.home()), 'appoppy', tracking_number, 'lep.fits')
    lep = LongExposurePetalometer.load(filename)
    return lep


def main_no_opd():
    tracking_number = 'none'
    lep = LongExposurePetalometer(tracking_number=tracking_number,
                                  rot_angle=10,
                                  start_from_step=100,
                                  n_iter=20)
    lep.run()
    lep.save(os.path.join(
        str(Path.home()), 'appoppy', tracking_number, 'lep.fits'))
