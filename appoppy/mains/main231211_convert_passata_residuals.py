

from appoppy.elt_aperture import PUPIL_MASK_480, PUPIL_MASK_480_PHASE_C_SPIDER23
from appoppy.maory_residual_wfe import PASSATASimulationConverter, restore_residual_wavefront


def convert_passata_residual_example():
    '''
    Let's assume we need to convert the residuals of simulation 20210511_144618.0
    along the direction rho=0, theta=0

    Data files CUBE_CL_coo0.0_0.0.fits and params.txt have already been 
    manually copied in appoppy/data/passata_simulations/20210511_144618.0  
    
    Create a PASSATASimulationConverter object.
    Define the directions rho e theta as string '0.0' as in the name of the fits file    
    
    From the file params.txt search for (it should be done automatically but  we are lazy guys) 
    PUPIL_STOP.PUPIL_MASK_TAG and MAIN.TIME_STEP  
    Define pupil_mask_tag = 'EELT480pp0.0813spider.fits'. It must be listed in appoppy.elt_aperture    
    Define time_step = 0.002 
    
    Call PASSATASimulationConverter.convert_from_fits_data()
    
    The folder appoppy/data/passata_simulations_converted/20120511_144618.0_coo0.0_0.0 
    is created with file CUBE_CL_converted.fits inside.
    
    The converted file can be read using restore_residual_wavefront
    
    IMPORTANT: DO NOT SAVE ON GITHUB THE DATA FILES. THEY ARE TOO BIG
    
    '''
    tracknum = '20210511_144618.0'
    rho = '0.0'
    theta = '0.0'
    pupil_mask_tag = PUPIL_MASK_480
    time_step = 0.002
    psc = PASSATASimulationConverter()
    psc.convert_from_fits_data(tracknum, rho, theta, pupil_mask_tag, time_step)

    wf, hdr = restore_residual_wavefront('20210511_144618.0_coo0.0_0.0')
    print(hdr)
    return wf


def convert_residual_for_morfeo_analysis():
    psc = PASSATASimulationConverter()
    psc.convert_from_fits_data(
        '20231209_202232.0', '55.0', '0.0', PUPIL_MASK_480_PHASE_C_SPIDER23, 0.002)
    psc.convert_from_fits_data(
        '20231209_202232.0', '55.0', '120.0', PUPIL_MASK_480_PHASE_C_SPIDER23, 0.002)
    psc.convert_from_fits_data(
        '20231209_202232.0', '55.0', '240.0', PUPIL_MASK_480_PHASE_C_SPIDER23, 0.002)

    psc.convert_from_fits_data(
        '20231209_202232.0', '55.0', '0.0DAO', PUPIL_MASK_480_PHASE_C_SPIDER23, 0.002)
    psc.convert_from_fits_data(
        '20231209_202232.0', '55.0', '120.0DAO', PUPIL_MASK_480_PHASE_C_SPIDER23, 0.002)
    psc.convert_from_fits_data(
        '20231209_202232.0', '55.0', '240.0DAO', PUPIL_MASK_480_PHASE_C_SPIDER23, 0.002)

    psc.convert_from_fits_data(
        '20231213_101833.0', '0.0', '0.0', PUPIL_MASK_480_PHASE_C_SPIDER23, 0.002)
    psc.convert_from_fits_data(
        '20231212_212912.0', '0.0', '0.0', PUPIL_MASK_480_PHASE_C_SPIDER23, 0.002)
    psc.convert_from_fits_data(
        '20231213_123051.0', '0.0', '0.0', PUPIL_MASK_480_PHASE_C_SPIDER23, 0.002)
    psc.convert_from_fits_data(
        '20231213_123200.0', '0.0', '0.0', PUPIL_MASK_480_PHASE_C_SPIDER23, 0.002)
    psc.convert_from_fits_data(
        '20231213_123403.0', '0.0', '0.0', PUPIL_MASK_480_PHASE_C_SPIDER23, 0.002)
