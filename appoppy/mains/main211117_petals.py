from appoppy.simple_example import PyramidWFSExample, FocalPlaneWFSExample
import matplotlib.pyplot as plt
import astropy.units as u
import poppy


def main(petal_in_meter=100 * u.nm):
    model = PyramidWFSExample()
    model.set_m4_petals(petal_in_meter)
    model.display_intermediates = False
    model.run()
    plt.figure()
    model.display_pupil_intensity()
    return model


class FocalPlaneSpatialFilter(object):

    def __init__(self,
                 spatial_filter_in_arcsec=1,
                 name='Spatial Filter'):
        self.name = name
        self.spatial_filter_in_arcsec = spatial_filter_in_arcsec

    def add_to_system(self, osys, index):
        osys.add_pupil(poppy.FQPM_FFT_aligner(), index=index)
        osys.add_image(
            poppy.AnnularFieldStop(
                radius_outer=self.spatial_filter_in_arcsec,
                name='spatial filter'),
            index=index + 1)

        pupil_plane = poppy.FQPM_FFT_aligner(direction='backward',
                                             name="Spatial Filter pupil plane")
        pupil_plane.wavefront_display_hint = 'intensity'
        osys.add_pupil(
            pupil_plane,
            index=index + 2)


class SpatialFilterExample(FocalPlaneWFSExample):

    def __init__(self,
                 spatial_filter_in_lambda_d=0,
                 **kwargs):
        FocalPlaneWFSExample.__init__(self, **kwargs)
        sf = FocalPlaneSpatialFilter(
            name='My Spatial Filter',
            spatial_filter_in_arcsec=spatial_filter_in_lambda_d * self.lambda_over_d)
        sf.add_to_system(self._osys, self._wfs_plane_from)


def mainSpatialFilter(petal_in_meter=100 * u.nm, spatial_filter_in_lambda_d=1):
    model = SpatialFilterExample()
    model.set_m4_petals(petal_in_meter)
    model.display_intermediates = False
    model.run()
    plt.figure()
    model.display_pupil_intensity()
    return model

