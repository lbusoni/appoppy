from appoppy.elt_aperture import PUPIL_MASK_480_PHASE_C_SPIDER23
from appoppy.maory_residual_wfe import PASSATASimulationConverter


class KnownPassataTracknum:
    TN_MCAO_1 = '20231209_202232.0_coo55.0_0.0'
    TN_MCAO_2 = '20231209_202232.0_coo55.0_120.0'
    TN_DAO_1 = '20231209_202232.0_coo55.0_0.0DAO'
    TN_SCAO_1000 = '20231212_212912.0_coo0.0_0.0'
    TN_SCAO_2000 = '20231213_101833.0_coo0.0_0.0'
    TN_REF_500 = '20231213_123051.0_coo0.0_0.0'
    TN_REF_100 = '20231213_123200.0_coo0.0_0.0'
    TN_REF_10 = '20231213_123403.0_coo0.0_0.0'

    TN_MCAO_1_PS = '20231209_202232.0_coo55.0_0.0_ps'
    TN_MCAO_2_PS = '20231209_202232.0_coo55.0_120.0_ps'
    TN_DAO_1_PS = '20231209_202232.0_coo55.0_0.0DAO_ps'
    TN_SCAO_1000_PS = '20231212_212912.0_coo0.0_0.0_ps'
    TN_SCAO_2000_PS = '20231213_101833.0_coo0.0_0.0_ps'
    TN_REF_500_PS = '20231213_123051.0_coo0.0_0.0_ps'
    TN_REF_100_PS = '20231213_123200.0_coo0.0_0.0_ps'
    TN_REF_10_PS = '20231213_123403.0_coo0.0_0.0_ps'


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
