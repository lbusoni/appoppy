import poppy
from astropy.io import fits
from poppy.poppy_core import PlaneType
import os
from appoppy.package_data import data_root_dir


def restore_elt_pupil_mask():
    fname = os.path.join(data_root_dir(),
                         'pupilstop',
                         'EELT480pp0.0813spider.fits')
    mask = fits.getdata(fname)
    maskb = (mask == False)
    # it shoud be read from params.txt MAIN PIXEL_PITCH
    pixel_pitch = 0.08215
    hdr = fits.Header()
    hdr['PIXELSCL'] = pixel_pitch
    hdu = fits.PrimaryHDU(data=maskb.astype(int), header=hdr)
    return hdu


def ELTAperture():
    def _invert_int_mask(mask):
        return -mask + 1
    hdumask = restore_elt_pupil_mask()
    hdumask.data = _invert_int_mask(hdumask.data)
    return poppy.FITSOpticalElement(
        transmission=fits.HDUList([hdumask]),
        planetype=PlaneType.pupil)
