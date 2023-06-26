import numpy as np
import os
from pathlib import Path
import matplotlib
from matplotlib import pyplot as plt
import imageio
import glob


class Gif2DMapsAnimator():
    # TODO: move it on arte
    '''
    Create jpeg and animated gif from a 3D array

    The first axis of the array is the time dimension


    Parameters
    ----------

    root_folder: string
        the root folder where jpeg and gif will be saved.
        It is created, if it doesn't exist

    cube_map: ndarray
        maps to be animated. Axis 0 is time

    deltat: float
        time interval between maps in s. Default=0.001s

    pixelsize: float
        map pixel size in meter. Default=1m

    vminmax: tuple of shape (2,) or None
        min max of colorbar normalization.
        Set it to None for min/max range of cube_map.
        Default to None

    exist_ok: bool
        set it to True to allow overriding files in root_folder.
        Default=True

    remove_jpg: bool
        set it to True to remove all jpg files in the root_folder 
        after the gif has been created.
        Default=False
    '''

    def __init__(self,
                 root_folder,
                 cube_map,
                 deltat=1,
                 pixelsize=1,
                 vminmax=None,
                 exist_ok=True,
                 remove_jpg=False
                 ):
        self._jpg_root = root_folder
        self._cube_map = cube_map
        self._n_iter = cube_map.shape[0]
        self._dt = deltat
        self._pxs = pixelsize
        self._extent = [-cube_map.shape[1] * self._pxs / 2,
                        cube_map.shape[1] * self._pxs / 2,
                        -cube_map.shape[2] * self._pxs / 2,
                        cube_map.shape[2] * self._pxs / 2]
        if vminmax is None:
            self._vmin = None  # 3 * np.std(self._cube_map, axis=(1, 2)))
            self._vmax = None  # -self._vmax
        else:
            self._vmin = vminmax[0]
            self._vmax = vminmax[1]
        Path(self._jpg_root).mkdir(parents=True, exist_ok=exist_ok)
        self._should_remove_jpg = remove_jpg

    def _animate(self, step=1, **kwargs):
        plt.close('all')
        for i in range(0, self._n_iter, step):
            self.display_map(i, title='t=%4d' % 
                             (i * self._dt * 1000), **kwargs)
            plt.savefig(self._file_name(i))
            plt.close('all')

    def display_map(self, idx, title='', cmap='twilight', colorbar=True):
        plt.clf()
        norm = matplotlib.colors.Normalize(vmin=self._vmin, vmax=self._vmax)
        plt.imshow(self._cube_map[idx],
                   origin='lower',
                   norm=norm,
                   extent=self._extent,
                   cmap=cmap)
        if colorbar:
            plt.colorbar()
        plt.title(title)
        plt.show()

    def _file_name(self, idx):
        return os.path.join(self._jpg_root, '%04d.jpeg' % idx)

    def _remove_jpg_if_required(self):
        if self._should_remove_jpg:
            filelist = glob.glob(os.path.join(self._jpg_root, "*.jpeg"))
            for f in filelist:
                os.remove(f)

    def make_gif(self, step=10, **kwargs):
        self._animate(step, **kwargs)
        outputfname = os.path.join(self._jpg_root, 'map.gif')
        with imageio.get_writer(outputfname, mode='I') as writer:
            for i in range(0, self._n_iter, step):
                image = imageio.imread(self._file_name(i))
                writer.append_data(image)
        self._remove_jpg_if_required()
