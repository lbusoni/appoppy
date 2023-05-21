import poppy
from poppy.optics import ScalarOpticalPathDifference
from poppy.poppy_core import PlaneType
import numpy as np
from astropy import units as u
from appoppy.petaled_m4 import PetaledM4
from appoppy.elt_aperture import ELTAperture
from appoppy.elt_for_petalometry import EltForPetalometry
from matplotlib import pyplot as plt
from appoppy.gif_animator import Gif2DMapsAnimator
import os


def mainZernike(zernike_coeff_in_nm=[0, 0], r0=np.inf, title=''):
    model = EltForPetalometry(
        r0=r0, zern_coeff=np.array(zernike_coeff_in_nm) * 1e-9)
    model.compute_psf()
    plt.figure(1)
    model.display_psf(title=title)
    plt.figure(2)
    model.display_pupil_opd(title=title)
    return model


def main_long_exposure_zernike(indexes, ampli, niter=50):

    def _shuffle(indexes, ampli):
        cc = np.zeros(indexes.max())
        cc[indexes - 1] = np.random.randn(indexes.size) * ampli
        return cc
    model = EltForPetalometry(zern_coeff=_shuffle(indexes, ampli))
    model.compute_psf()
    sh = (niter,
          model.psf()[0].shape[0],
          model.psf()[0].shape[1])
    psf = np.ma.zeros(sh)
    shopd = (niter,
             model.pupil_opd().shape[0],
             model.pupil_opd().shape[1])
    opd = np.ma.zeros(shopd)
    for i in range(niter):
        print("step %d" % i)
        cc = _shuffle(indexes, ampli)
        model.set_input_wavefront_zernike(cc)
        psf[i] = model.psf()[0].data
        opd[i] = model.pupil_opd()
    return psf, opd, model


def main_long_exposure_atmo(r0=0.2):
    model = EltForPetalometry(
        r0=r0, kolm_seed=0)
    model.compute_psf()
    niter = 100
    sh = (niter,
          model.psf()[0].shape[0],
          model.psf()[0].shape[1])
    psf = np.ma.zeros(sh)
    shopd = (niter,
             model.pupil_opd().shape[0],
             model.pupil_opd().shape[1])
    opd = np.ma.zeros(shopd)
    for i in range(niter):
        print("step %d" % i)
        model.set_kolm_seed(i)
        psf[i] = model.psf()[0].data
        opd[i] = model.pupil_opd()
    return psf, opd, model


def make_gifs(basedir, psf, opd, model, step=1, vminmax=None):
    Gif2DMapsAnimator(
        os.path.join(basedir, 'opd'),
        opd,
        deltat=1e-3,
        pixelsize=model.pixelsize.to_value(u.m),
        exist_ok=True,
        remove_jpg=True).make_gif(step=step)

    pxscalepsf = float(model.psf()[0].header['PIXELSCL'])
    Gif2DMapsAnimator(
        os.path.join(basedir, 'psf'),
        psf,
        deltat=1e-3,
        pixelsize=pxscalepsf,
        exist_ok=True,
        remove_jpg=True,
        vminmax=vminmax).make_gif(step=step)

    def _cumaverage(cube):
        cs = cube.cumsum(axis=0)
        res = np.array([cs[i] / (i + 1) for i in np.arange(cube.shape[0])])
        return res
    cumret = _cumaverage(psf)

    Gif2DMapsAnimator(
        os.path.join(basedir, 'cumpsf'),
        cumret,
        deltat=1e-3,
        pixelsize=pxscalepsf,
        exist_ok=True,
        remove_jpg=True,
        vminmax=vminmax).make_gif(step=step, colorbar=False)
