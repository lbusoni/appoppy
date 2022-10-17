from appoppy.petalometer import Petalometer
import numpy as np
from astropy import units as u
import matplotlib
import matplotlib.pyplot as plt
from appoppy.elt_for_petalometry import EltForPetalometry


def main_plot_pupil():
    pet = Petalometer(r0=99999, petals=np.zeros(6)*u.nm, rotation_angle=20)
    pet._i4.display_pupil_intensity()
    pet._model1.display_pupil_intensity()
    pet._model2.display_pupil_intensity()
    
    
    
    pet = Petalometer(r0=99999, petals=np.zeros(6)*u.nm, 
                      rotation_angle=20,
                      zernike=np.array([0, 0, 10])*u.um)
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
    m2 = EltForPetalometry(r0=0.26, npix=240, kolm_seed=np.random.randint(2147483647)) 
    osys=m2._osys
    kopd= osys.planes[0].get_opd(osys.input_wavefront(2.2e-6 * u.m))
    kopdm= np.ma.MaskedArray(kopd, mask=m2.pupil_mask())
    print('std %g' % kopdm.std())
    
    
def opd_turbolenza_residui_MCAO(start_from=0):
    m2 = EltForPetalometry(r0=np.inf, npix=480, residual_wavefront_start_from=start_from) 
    osys=m2._osys
    kopd= osys.planes[0].get_opd(osys.input_wavefront(2.2e-6 * u.m))
    kopdm= np.ma.MaskedArray(kopd, mask=m2.pupil_mask())
    print('std %g' % kopdm.std())
    return m2
    # perchè sembra diversa dal cubo di Guido? no è ok da 1.8um a 330nm
    

def phase_shift():
    pet = Petalometer(r0=99999,
                      petals=np.array([0, 200, -400, 600, -800, 1000])*u.nm,
                      rotation_angle=20,
                      zernike=np.array([0,10000,-5000,3000,400,500,600,-200,100])*u.nm)
    pet._i4._wf_0.display()
    pet._i4._wf_1.display()
    pet._i4._wf_2.display()
    pet._i4._wf_3.display()
    


def no_turbolence():    
    pet = Petalometer(r0=99999,
                      petals=np.array([0, 100, -200, 300, -400, 500])*u.nm,
                      rotation_angle=20)
    
    pet._model2.display_pupil_opd()
    np.round(pet.all_jumps)
    np.round(pet.estimated_petals - pet.estimated_petals[0])
    np.round(pet.error_jumps)
    np.round(pet.error_petals)
    
    
    
class SeriesOfInterferogram():
    
    def __init__(self, rot_angle=10):
        self._pet = Petalometer(r0=np.inf,
                              petals=np.array([0, 0, 0,0, 0, 0])*u.nm,
                              rotation_angle=rot_angle)
        self._niter=10
        
    def run(self): 
        self._res_map =np.ma.zeros((self._niter, 480, 480))
        self._res_petals = np.zeros((self._niter, 6))
        self._pet.set_step_idx(0) 
        for i in range(self._niter):
            print("step %d" % self._pet._step_idx)
            self._pet.sense_wavefront_jumps()
            self._res_petals[self._pet._step_idx] = self._pet.error_petals
            self._res_map[self._pet._step_idx]=self._pet._i4.interferogram()
            self._pet.advance_step_idx()
    
    
    def display_map(self, resmap):
        plt.clf()
        norm = matplotlib.colors.Normalize(vmin=-1100, vmax=1100)
        plt.imshow(resmap,
                   origin='lower',
                   norm=norm,
                   extent=[-19.5, 19.5, -19.5, 19.5],
                   cmap='twilight')
        plt.colorbar()
        plt.show()
        
    
    def animate(self):
        import matplotlib.pyplot as plt
        # for i, z in enumerate(self._zenith_angles):
        for i, m in enumerate(self._res_map):
            self.display_map(m)
            plt.savefig('/Users/lbusoni/Downloads/anim/%04d.jpeg' % i)
            plt.close('all')

    
    
    