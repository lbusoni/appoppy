from appoppy.gif_animator import Gif2DMapsAnimator
from appoppy.low_wind_effect import LowWindEffectWavefront


def main230601_animate_LWE(vminmax=[0, 5], wind_speed=0.5):
    root_folder = '/Users/giuliacarla/Documents/INAF/Lavoro/Progetti/MORFEO/Petalometer/CiaoCiaoWFS/analysis/animations/LWE_' + str(wind_speed)
    lwe = LowWindEffectWavefront(wind_speed=wind_speed)
    lwe_cube_maps = lwe.phase_screens()
    # lwe_cube_maps_rot = rotate(lwe_cube_maps, 30, axes=(1, 2), reshape=False)
    gf = Gif2DMapsAnimator(root_folder, lwe_cube_maps,
                           vminmax=vminmax, deltat=lwe.time_step,
                           pixelsize=lwe._pxscale.value)
    gf.make_gif(step=1)
