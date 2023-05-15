import numpy as np
import astropy.units as u
import matplotlib.pyplot as plt
from appoppy.petalometer import Petalometer


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

    def main(self, petals=[200, 30, -100, 370, 500, 0] * u.nm):
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
