from appoppy.petalometer import Petalometer
import numpy as np
from astropy import units as u
import matplotlib
import matplotlib.pyplot as plt
from appoppy.elt_for_petalometry import EltForPetalometry
from astropy.io import fits
from astropy.io.fits.header import Header
import imageio
import os
from appoppy.ao_residuals import AOResidual
from pathlib import Path


def main_plot_pupil():
    pet = Petalometer(use_simulated_residual_wfe=False,
                      r0=99999,
                      petals=np.zeros(6) * u.nm,
                      rotation_angle=20)
    pet._i4.display_pupil_intensity()
    pet._model1.display_pupil_intensity()
    pet._model2.display_pupil_intensity()

    pet = Petalometer(use_simulated_residual_wfe=False,
                      r0=99999,
                      petals=np.zeros(6) * u.nm,
                      rotation_angle=20,
                      zernike=np.array([0, 0, 10]) * u.um)
    plt.clf()
    plt.imshow(pet._model2.pupil_phase() * pet._model2.pupil_intensity(),
               origin='lower',
               cmap='twilight')
    plt.clf()
    plt.imshow(pet._model1.pupil_phase() * pet._model2.pupil_intensity(),
               origin='lower',
               cmap='twilight')
    pet._i4.display_pupil_intensity()


def opd_turbolenza_kolmo():
    m2 = EltForPetalometry(
        use_simulated_residual_wfe=False,
        r0=0.26, kolm_seed=np.random.randint(2147483647))
    osys = m2._osys
    kopd = osys.planes[0].get_opd(osys.input_wavefront(2.2e-6 * u.m))
    kopdm = np.ma.MaskedArray(kopd, mask=m2.pupil_mask())
    print('std %g' % kopdm.std())


def opd_turbolenza_residui_MCAO(start_from=0):
    m2 = EltForPetalometry(
        use_simulated_residual_wfe=True,
        tracking_number='20210518_223459.0',
        residual_wavefront_start_from=start_from)
    osys = m2._osys
    kopd = osys.planes[0].get_opd(osys.input_wavefront(2.2e-6 * u.m))
    kopdm = np.ma.MaskedArray(kopd, mask=m2.pupil_mask())
    print('std %g' % kopdm.std())
    return m2
    # perchè sembra diversa dal cubo di Guido? no è ok da 1.8um a 330nm


def phase_shift():
    pet = Petalometer(use_simulated_residual_wfe=False,
                      r0=99999,
                      petals=np.array([0, 200, -400, 600, -800, 1000]) * u.nm,
                      rotation_angle=20,
                      zernike=np.array([0, 10000, -5000, 3000, 400, 500, 600, -200, 100]) * u.nm)
    pet._i4._wf_0.display()
    pet._i4._wf_1.display()
    pet._i4._wf_2.display()
    pet._i4._wf_3.display()


def no_turbolence():
    pet = Petalometer(use_simulated_residual_wfe=False,
                      r0=99999,
                      petals=np.array([0, 100, -200, 300, -400, 500]) * u.nm,
                      rotation_angle=20)

    pet._model2.display_pupil_opd()
    np.round(pet.all_jumps)
    np.round(pet.estimated_petals - pet.estimated_petals[0])
    np.round(pet.error_jumps)
    np.round(pet.error_petals)


#os.path.join('/Users', 'lbusoni', 'Downloads', 'anim')
class SeriesOfInterferogram():

    def __init__(self, rot_angle=10, jpeg_root_folder=None):
        self._niter = 1000
        self._rot_angle = rot_angle
        if jpeg_root_folder is None:
            home = str(Path.home())
            jpeg_root_folder = os.path.join(home, 'appoppy_anim')
        self._jpg_root = jpeg_root_folder
        self._res_map_cumave = None
        self._res_map = None
        self._meas_petals = None
        self._meas_petals_no_global_pist = None
        self._meas_petals_cumave = None
        self._aores = AOResidual()

    def run(self):
        self._pet = Petalometer(use_simulated_residual_wfe=True,
                                tracking_number='20210518_223459.0',
                                petals=np.array([0, 0, 0, 0, 0, 0]) * u.nm,
                                rotation_angle=self._rot_angle)
        self._res_map = np.ma.zeros((self._niter, 480, 480))
        self._meas_petals = np.zeros((self._niter, 6))
        self._pet.set_step_idx(0)
        for i in range(self._niter):
            print("step %d" % self._pet._step_idx)
            self._pet.sense_wavefront_jumps()
            self._meas_petals[self._pet._step_idx] = self._pet.error_petals
            self._res_map[self._pet._step_idx] = self._pet._i4.interferogram()
            self._pet.advance_step_idx()

    @property
    def phase_screen(self):
        return self._aores.phase_screen

    @property
    def phase_screen_cumave(self):
        return self._aores.phase_screen_cumave

    @property
    def phase_screen_ave(self):
        return self._aores.phase_screen_ave

    @property
    def res_map(self):
        return self._res_map

    @property
    def petals(self):
        if self._meas_petals_no_global_pist is None:
            self._meas_petals_no_global_pist = self._meas_petals - np.broadcast_to(
                self._meas_petals.mean(axis=1), (6, self._niter)).T
        return self._meas_petals_no_global_pist

    @property
    def res_map_cumave(self):
        if self._res_map_cumave is None:
            self._res_map_cumave = self._cumaverage(self.res_map)
        return self._res_map_cumave

    @property
    def res_map_ave(self):
        return self.res_map.mean(axis=0)

    @property
    def petals_cumave(self):
        if self._meas_petals_cumave is None:
            self._meas_petals_cumave = self._cumaverage(self.petals)
        return self._meas_petals_cumave

    def _cumaverage(self, cube):
        cs = cube.cumsum(axis=0)
        res = np.array([cs[i] / (i + 1) for i in np.arange(cube.shape[0])])
        return res

    def display_map(self, resmap, title='', vmin=None, vmax=None, cmap='twilight'):
        plt.clf()
        norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
        plt.imshow(resmap,
                   origin='lower',
                   norm=norm,
                   extent=[-19.5, 19.5, -19.5, 19.5],
                   cmap=cmap)
        plt.colorbar()
        plt.title(title)
        plt.show()

    def display_mean(self):
        self.display_map(self.res_map_ave)

    def display_std(self):
        self.display_map(self.res_map.std(axis=0))

    def _animate(self, cube_map, basename='', step=1, **kwargs):
        # for i, z in enumerate(self._zenith_angles):
        plt.close('all')
        for i in range(0, self._niter, step):
            m = cube_map[i]
            self.display_map(m, title='t=%4d ms' % (i * 2), **kwargs)
            plt.savefig(os.path.join(self._jpg_root,
                                     '%s%04d.jpeg' % (basename, i)))
            plt.close('all')

    def animate_interferograms(self):
        self._animate(self.res_map, step=20, vmin=-1100, vmax=1100)

    def animate_interferograms_cumave(self):
        self._animate(self.res_map_cumave, basename='cs_',
                      step=20, vmin=-400, vmax=400)

    def animate_phase_screens_cumave(self):
        self._animate(self.phase_screen_cumave, basename='psc_',
                      step=10, vmin=-400, vmax=400, cmap='cividis')

    def animate_phase_screens(self):
        self._animate(self.phase_screen, basename='ps_',
                      step=10, vmin=-2000, vmax=2000, cmap='cividis')

    def make_gif(self, step=10, basename=''):
        outputfname = os.path.join(self._jpg_root, '%smap.gif' % basename)
        with imageio.get_writer(outputfname, mode='I') as writer:
            for i in range(0, self._niter, step):
                filename = os.path.join(self._jpg_root,
                                        '%s%04d.jpeg' % (basename, i))
                image = imageio.imread(filename)
                writer.append_data(image)

    def save(self, filename):
        hdr = Header()
        hdr['NITER'] = self._niter
        hdr['ROTANG'] = self._rot_angle
        fits.writeto(filename, self.res_map.data, header=hdr, overwrite=True)
        fits.append(filename, self.res_map.mask.astype(int))
        fits.append(filename, self._meas_petals)

    @staticmethod
    def load(filename):
        soi = SeriesOfInterferogram()
        hdr = fits.getheader(filename)
        soi._niter = hdr['NITER']
        soi._rot_angle = hdr['ROTANG']
        res_map_d = fits.getdata(filename, 0)
        res_map_m = fits.getdata(filename, 1).astype(bool)
        soi._res_map = np.ma.masked_array(data=res_map_d, mask=res_map_m)
        soi._meas_petals = fits.getdata(filename, 2)
        return soi

    def plot_petals(self):
        t = np.arange(0, self._niter * 0.002, 0.002)

        dd = self.petals
        plt.plot(t, dd)
        plt.ylabel("petals [nm]")
        plt.xlabel("time [s]")
        print("mean %s" % str(dd.mean(axis=0)))
        print("std  %s" % str(dd.std(axis=0)))

    def _compute_jumps_from_res_map_ave(self, rotation_angle):
        return Petalometer.compute_jumps(self.res_map_ave, rotation_angle)

    def petals_from_res_map_ave(self):
        jj = self._compute_jumps_from_res_map_ave(self._rot_angle)
        pp = -1 * np.cumsum(jj[::2])
        pp = pp - pp.mean()
        #pp=np.array([168., 275., 339., 177., 156.,  93.])*u.nm
        pet = Petalometer(
            use_simulated_residual_wfe=False,
            r0=999999,
            petals=pp,
            rotation_angle=self._rot_angle)
        opd = pet._model2.pupil_opd()
        return opd, pp, jj

    def phase_screen_petal_corrected(self):
        return self.phase_screen_ave - self.petals_from_res_map_ave()


class GifAnimator():
    # TODO: move it on arte
    '''
    Create jpeg and animated gif from a 3D array

    The first axis of the array is the time dimension


    Parameters
    ----------

    root_folder: string
        the root folder where jpeg and gif will be saved

    cube_map: ndarray
        maps to be animated. Axis 0 is time

    deltat: float
        time interval between maps in millisec
    '''

    def __init__(self, root_folder, cube_map, deltat):
        self._jpg_root = root_folder
        self._cube_map = cube_map
        self._n_iter = cube_map.shape[0]
        self._dt = deltat
        Path(self._jpg_root).mkdir(parents=True, exist_ok=False)

    def _animate(self, step=1, **kwargs):
        plt.close('all')
        for i in range(0, self._n_iter, step):
            self.display_map(i, title='t=%4d ms' % (i * self._dt), **kwargs)
            plt.savefig(self._file_name(i))
            plt.close('all')

    def display_map(self, idx, title='', vmin=None, vmax=None, cmap='twilight'):
        plt.clf()
        norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
        plt.imshow(self._cube_map[idx],
                   origin='lower',
                   norm=norm,
                   extent=[-19.5, 19.5, -19.5, 19.5],
                   cmap=cmap)
        plt.colorbar()
        plt.title(title)
        plt.show()

    def _file_name(self, idx):
        return os.path.join(self._jpg_root, '%04d.jpeg' % idx)

    def make_gif(self, step=10, **kwargs):
        self._animate(step, **kwargs)
        outputfname = os.path.join(self._jpg_root, 'map.gif')
        with imageio.get_writer(outputfname, mode='I') as writer:
            for i in range(0, self._n_iter, step):
                image = imageio.imread(self._file_name(i))
                writer.append_data(image)


def petal_estimate_55():
    soi = SeriesOfInterferogram.load(
        '/Users/lbusoni/Downloads/anim/soi55.fits')
    eopd, epet, ejump = soi.petals_from_res_map_ave()

    mpet = soi._aores.petals_average * u.nm

    return soi, eopd, epet, ejump, mpet
