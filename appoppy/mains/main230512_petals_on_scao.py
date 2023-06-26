import numpy as np
import astropy.units as u
import matplotlib.pyplot as plt
from appoppy.petalometer import Petalometer
from appoppy.long_exposure import LongExposurePetalometer
import os


class PetalsOnHIRESResiduals():
    '''
    Petals estimation in case SCAO (HIRES) residuals are present (pet1) or not
    (pet2).
    Tracking numbers available:
        - '20221026_123454.0_coo0.0_0.0': seeing = 0.4";
        - '20221026_125500.0_coo0.0_0.0': seeing = 0.8";
        - '20221026_130556.0_coo0.0_0.0': seeing = 1.2".
    '''    
    
    TN_BAD_SEEING = '20221026_130556.0_coo0.0_0.0'
    TN_MEDIUM_SEEING = '20221026_125500.0_coo0.0_0.0'
    TN_GOOD_SEEING = '20221026_123454.0_coo0.0_0.0'
    DATA_ROOT_DIR = '/Users/giuliacarla/Documents/INAF/Lavoro/Progetti/MORFEO/Petalometer/CiaoCiaoWFS/analysis/data/from_appoppy'
    
    def __init__(self, seeing='good', rotation_angle=15):
        self._set_tracking_number(seeing)
        self._rot_angle = rotation_angle
        
    def _set_tracking_number(self, seeing):
        if seeing == 'bad':
            self._tracking_number = self.TN_BAD_SEEING
        elif seeing == 'medium':
            self._tracking_number = self.TN_MEDIUM_SEEING
        elif seeing == 'good':
            self._tracking_number = self.TN_GOOD_SEEING

    def main230515_petals_estimation(
            self, petals=[200, 30, -100, 370, 500, 0] * u.nm):
        '''
        Estimate input petals when AO residuals are present.
        '''
        pet1 = Petalometer(r0=np.inf,
                          tracking_number=self._tracking_number,
                          petals=petals,
                          rotation_angle=self._rot_angle)
        pet2 = Petalometer(r0=np.inf,
                          tracking_number=None,
                          petals=petals,
                          rotation_angle=self._rot_angle)
        print('\n\nEstimated petals with SCAO residuals ON: %s'
              % pet1.estimated_petals)
        print('\nErrors with SCAO residuals ON: %s' % pet1.error_petals)
        print('\nErrors with SCAO residuals ON (without first term subtraction): %s'
              % (pet1.estimated_petals - pet1.petals))
        print('\nJumps on odd segments with SCAO residuals ON: %s'
              % pet1.all_jumps[1::2])
        print('\nStd of errors with SCAO residuals ON: %s' % pet1.error_petals.std())
        print('\nErrors with SCAO residuals OFF: %s' % pet2.error_petals)
        plt.clf()
        pet1._i4.display_interferogram()
        plt.figure()
        pet2._i4.display_interferogram()
        return pet1, pet2
    
    def main230516_petals_correction(self):
        '''
        Estimate petals from petalometer and correct them within the AO loop.
        Petals are measured from temporal average of AO residuals (i.e. from 
        long exposure of petalometer detector). 
        '''
        le = LongExposurePetalometer.load(
            os.path.join(self.DATA_ROOT_DIR, self._tracking_number,
                         'long_exp.fits'))
        correction_from_petalometer = le.phase_correction_from_petalometer()
        ao_res_phase_screen = le.phase_screen()
        pet_res_phase_screen = []
        std_pet_res_phase_screen = []
        for ps in ao_res_phase_screen:
            corrected_ps = ps - correction_from_petalometer
            pet_res_phase_screen.append(corrected_ps)
            std_pet_res_phase_screen.append(corrected_ps.std())
        return np.array(pet_res_phase_screen), np.array(std_pet_res_phase_screen)
    
    def main230516_LWE_petals_estimation(self, wind_speed):
        '''
        Estimate petals introduced by LWE without and with AO residuals.
        '''
        pet_ao_off = Petalometer(lwe_speed=wind_speed,
                                 rotation_angle=self._rot_angle)
        petals_ao_off = pet_ao_off.estimated_petals
        
        pet_ao_on = Petalometer(tracking_number=self._tracking_number,
                                lwe_speed=wind_speed,
                                rotation_angle=self._rot_angle)
        petals_ao_on = pet_ao_on.estimated_petals
        plt.clf()
        pet_ao_off._i4.display_interferogram()
        plt.figure()
        pet_ao_on._i4.display_interferogram()
        print('\nPetals difference SCAO residuals ON/OFF: %s'
              % (petals_ao_on - petals_ao_off))
        return petals_ao_off, petals_ao_on
    

def main230517_LWE_deltaOPD_estimation(wind_speed, rotation_angle):
    '''
    Estimate delta OPD across spiders introduced by LWE. Comparison with
    Fig.11 in "D. Martins, R. Holzlöhner, C. Verinaud, C. Kleinklaus,
    “Transient Wavefront Error from Cooled Air Downwind of Telescope
    Spiders”, SPIE Montreal (2022)".
    '''
    pet = Petalometer(lwe_speed=wind_speed, rotation_angle=rotation_angle)
    r = rotation_angle
    angs = (90, 90 - r, 30, 30 - r, -30, -30 - r, -
            90, -90 - r, -150, -150 - r, -210, -210 - r, -270)
    ifms = [pet._mask_ifgram(pet._i4.interferogram(), (angs[i + 1], angs[i]))
            for i in range(len(angs) - 1)]
    plt.figure()
    for i in range(6):
        plt.imshow(ifms[::2][i], origin='lower')
    plt.figure()
    plt.plot(ifms[0][:, 131])
    return pet, ifms


def main230518_petals_estimation_with_Kolmogorov_turbulence(r0=0.1):
    petals = [200, 30, -100, 370, 500, 0] * u.nm
    ang_small = 15
    ang_overlap = 60
    pet_small = Petalometer(r0=r0, petals=petals, rotation_angle=ang_small)
    plt.figure()
    pet_overlap = Petalometer(r0=r0, petals=petals, rotation_angle=ang_overlap)
    petals_small_angle = pet_small.estimated_petals
    petals_overlap_angle = pet_overlap.estimated_petals
    print('Petals estimation with rotation of %s: %s' % (ang_small,
                                                     petals_small_angle))
    print('Petals estimation with rotation of %s: %s' % (ang_overlap,
                                                     petals_overlap_angle))
    return pet_small, pet_overlap
