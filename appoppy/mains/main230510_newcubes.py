from appoppy.long_exposure import LongExposurePetalometer
import os
from pathlib import Path


def main_save():
    tracking_number = '20221026_123454.0'
    lep = LongExposurePetalometer(tracking_number, rot_angle=10)
    lep.run()
    lep.save(os.path.join(
        str(Path.home()), 'appoppy', tracking_number, 'lep.fits'))
    return lep


def main_load():
    tracking_number = '20221026_123454.0'
    filename = os.path.join(
        str(Path.home()), 'appoppy', tracking_number, 'lep.fits')
    lep = LongExposurePetalometer.load(filename)
    return lep
