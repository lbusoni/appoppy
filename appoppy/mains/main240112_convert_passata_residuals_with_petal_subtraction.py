from appoppy.elt_aperture import PUPIL_MASK_480, PUPIL_MASK_480_PHASE_C_SPIDER23
from appoppy.maory_residual_wfe import PASSATASimulationConverter, restore_residual_wavefront


def convert_residual_for_morfeo_analysis():
    psc = PASSATASimulationConverter()
    psc.convert_from_fits_data_with_petal_subtraction(
        '20231209_202232.0', '55.0', '0.0', PUPIL_MASK_480_PHASE_C_SPIDER23, 0.002)
    psc.convert_from_fits_data_with_petal_subtraction(
        '20231209_202232.0', '55.0', '120.0', PUPIL_MASK_480_PHASE_C_SPIDER23, 0.002)
    psc.convert_from_fits_data_with_petal_subtraction(
        '20231209_202232.0', '55.0', '240.0', PUPIL_MASK_480_PHASE_C_SPIDER23, 0.002)

    psc.convert_from_fits_data_with_petal_subtraction(
        '20231209_202232.0', '55.0', '0.0DAO', PUPIL_MASK_480_PHASE_C_SPIDER23, 0.002)
    psc.convert_from_fits_data_with_petal_subtraction(
        '20231209_202232.0', '55.0', '120.0DAO', PUPIL_MASK_480_PHASE_C_SPIDER23, 0.002)
    psc.convert_from_fits_data_with_petal_subtraction(
        '20231209_202232.0', '55.0', '240.0DAO', PUPIL_MASK_480_PHASE_C_SPIDER23, 0.002)

    psc.convert_from_fits_data_with_petal_subtraction(
        '20231213_101833.0', '0.0', '0.0', PUPIL_MASK_480_PHASE_C_SPIDER23, 0.002)
    psc.convert_from_fits_data_with_petal_subtraction(
        '20231212_212912.0', '0.0', '0.0', PUPIL_MASK_480_PHASE_C_SPIDER23, 0.002)
    psc.convert_from_fits_data_with_petal_subtraction(
        '20231213_123051.0', '0.0', '0.0', PUPIL_MASK_480_PHASE_C_SPIDER23, 0.002)
    psc.convert_from_fits_data_with_petal_subtraction(
        '20231213_123200.0', '0.0', '0.0', PUPIL_MASK_480_PHASE_C_SPIDER23, 0.002)
    psc.convert_from_fits_data_with_petal_subtraction(
        '20231213_123403.0', '0.0', '0.0', PUPIL_MASK_480_PHASE_C_SPIDER23, 0.002)
