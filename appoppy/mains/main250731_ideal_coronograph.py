import matplotlib.pyplot as plt
from networkx import radius
import poppy
from appoppy.ideal_coronograph import IdealCoronographMask
from astropy import units as u
import numpy as np
import matplotlib as mpl
from poppy.poppy_core import PlaneType
import logging
from poppy.optics import TiltOpticalPathDifference
from astropy.io import fits

logging.basicConfig(level=logging.DEBUG, format='%(levelname)s:%(name)s:%(message)s')
logging.getLogger('matplotlib').setLevel(logging.WARNING)
logging.getLogger('matplotlib.pyplot').setLevel(logging.WARNING)
logging.getLogger('asyncio').setLevel(logging.WARNING)


def _display_plane(wavefront_list, plane_number, what, scale='linear', max_log_contrast=1e4):
    wave = wavefront_list[plane_number]
    if what == 'intensity':
        image = wave.intensity
        cmap = 'cividis'
    elif what == 'phase':
        image = wave.phase
        cmap = 'twilight'
    else:
        raise Exception('Unknown property to display: %s')
    title = wave.location

    if scale == 'linear':
        norm = mpl.colors.Normalize()
    elif scale == 'log':
        #image = _pump_up_zero_for_log_display(image)
        vmax = np.max(image)
        vmin = np.maximum(np.min(image), np.max(image) / max_log_contrast)
        norm = mpl.colors.LogNorm(vmin=vmin, vmax=vmax)
    else:
        raise Exception('Unknown scale %s' % scale)

    if wave.planetype == PlaneType.pupil:
        pc = wave.pupil_coordinates(image.shape, wave.pixelscale)
        extent = [pc[0].min(), pc[0].max(), pc[1].min(), pc[1].max()]
    elif wave.planetype == PlaneType.image:
        pc = (wave.shape * u.pix * wave.pixelscale/2).to_value(u.arcsec)
        extent = [-pc[0], pc[0], -pc[1], pc[1]]
    plt.imshow(image, norm=norm, extent=extent, origin='lower', cmap=cmap)
    plt.title(title)
    plt.colorbar()


def show_intermediate_wavefronts(wf_list):    
    """
    Display the intermediate wavefronts in a grid.
    """
    n = len(wf_list)
    for i in range(n):
        plt.figure(figsize=(10, 4))
        plt.subplot(1, 2, 1)
        _display_plane(wf_list, i, 'intensity', scale='log', max_log_contrast=1e8)
        plt.subplot(1, 2, 2)
        _display_plane(wf_list, i, 'phase', scale='linear')
        plt.tight_layout()
        plt.show()

def show_system(osys):
    """
    Display the optical system with its components.
    """
    plt.figure(figsize=(10, 4))
    osys.display(what='amplitude')
    plt.title("Optical System Amplitude")
    plt.show()

    plt.figure(figsize=(10, 4))
    osys.display(what='intensity')
    plt.title("Optical System Intensity")
    plt.show()

    plt.figure(figsize=(10, 4))
    osys.display(what='opd')
    plt.title("Optical System OPD")
    plt.show()

def add_residual_wavefront_error(osys, radius=4, seed=42, rms=50 * u.nm):
    """
    Add a PowerSpectrumWFE to the optical system to simulate wavefront errors.
    """
    psd_parameters = [
        [2, 0 * u.m**2, 40 * u.m, 0.05, 1 * u.m**4]
    ]
    osys.add_pupil(
        poppy.PowerSpectrumWFE(
            psd_parameters=psd_parameters,
            seed=seed,
            rms=rms,
            radius=radius * u.m
        )
    )

def coronograph_system(source_offset_arcsec=0, source_flux=1.0, use_coronograph=True,
                       mask_transmission=1.0, outer_transmission=0.2775, radius_mask_lambda_over_d=0.25):
    logger = logging.getLogger("Coronograph System")
    radius = 4
    wavelength = 1e-6  # 1 micron
    oversample = 4
    diffraction_limit_arcsec = (wavelength / (2 * radius) * u.radian).to_value(u.arcsec)
    osys = poppy.OpticalSystem(name="Telescope + IdealCoronograph", oversample=oversample)
    osys.add_pupil(poppy.CircularAperture(radius=radius))  # radius in meters

    if source_offset_arcsec != 0:
        tilt = TiltOpticalPathDifference(
            tilt_angle=source_offset_arcsec * u.arcsec
        )
        osys.add_pupil(tilt)

    add_residual_wavefront_error(osys, radius=radius, seed=42, rms=100 * u.nm)

    # Add IdealCoronograph at the next image plane
    if use_coronograph:
        osys.add_image(IdealCoronographMask(name='Ideal Coronograph Mask',
                                            central_transmission= mask_transmission,
                                            outer_transmission=outer_transmission,
                                            radius=radius_mask_lambda_over_d*diffraction_limit_arcsec))

    # Pupil plane to check field
    osys.add_pupil(poppy.CircularAperture(radius=radius*oversample, name='Pupil plane without stop'))
    # Add pupil stop
    osys.add_pupil(poppy.CircularAperture(radius=radius, name='Exit Pupil'))

    # Add detector after coronagraph
    osys.add_detector(pixelscale=0.010, fov_arcsec=1.0, name='Detector after Coronograph')

    # Calculate PSFs at both detectors
    psf, wf_list = osys.calc_psf(wavelength, return_intermediates=True, display_intermediates=False)
    
    # Normalize total flux. 
    # Poppy already normalizes the PSF to 1 in the entrance pupil, so we just scale to the desired flux.
    for hdu in psf:
        hdu.data *= source_flux
 
    logger.info(f"Wavefront amplitude at exit pupil (max): {wf_list[-2].amplitude.max()}")
    # show_intermediate_wavefronts(wf_list)    
    
    plt.figure(figsize=(10, 4))
    poppy.display_psf(psf, title="PSF after IdealCoronograph", scale='log')
    plt.show()

    return osys, psf, wf_list



def main():
    '''
    Main function to run the Ideal Coronograph example.
    This function sets up the optical system, calculates the PSF,
    and displays the results.
    It also demonstrates the effect of a source offset on the PSF.
    The function returns the optical system, PSF, wavefront list,
    and a combined image of two PSFs with different source offsets.
    The first PSF has no source offset, and the second has a 0.2 arcsec offset.
    The combined image is return as a FITS HDUList.
    The function is intended to be run as a script.
    '''
    logger = logging.getLogger("IdealCoronographExample")
    use_coronograph = True  # Set to False to disable the coronagraph
    osys1, psf1, wf_list1 = coronograph_system(source_offset_arcsec=0, source_flux=1e3, use_coronograph=use_coronograph)
    osys2, psf2, wf_list2 = coronograph_system(source_offset_arcsec=0.05, source_flux=1, use_coronograph=use_coronograph)
    img = psf1[0].data + psf2[0].data
    image_fits = fits.HDUList(fits.PrimaryHDU(img))
    image_fits[0].header = psf1[0].header
 
    plt.figure(figsize=(10, 4))
    poppy.display_psf(image_fits, 
                      title="Binary system with IdealCoronograph", 
                      scale='log',
                      vmax=image_fits[0].data.max())
    plt.show()
    
    
    return osys1, psf1, wf_list1, osys2, psf2, wf_list2, image_fits


def main_coronograph_optimizer():
    logger = logging.getLogger("CoronographOptimizer")
    radius_mask_lambda_over_d = 0.25  # Radius of the mask in units of lambda/D
    osys_no, psf_no, wf_list_no = coronograph_system(source_offset_arcsec=0, source_flux=1,
                                             use_coronograph=False)
    osys_in, psf_in, wf_list_in = coronograph_system(source_offset_arcsec=0, source_flux=1,
                                             use_coronograph=True,
                                             mask_transmission=1.0,
                                             outer_transmission=0,
                                             radius_mask_lambda_over_d=radius_mask_lambda_over_d)
    osys_out, psf_out, wf_list_out = coronograph_system(source_offset_arcsec=0, source_flux=1,
                                             use_coronograph=True,
                                             mask_transmission=0.0,
                                             outer_transmission=1.0,
                                             radius_mask_lambda_over_d=radius_mask_lambda_over_d)

    # In the output pupil the total field is the sum of the contribution from
    # the inner mask and from the outer part. The phases are opposite. 
    # We want the amplitudes of the 2 contribution to be equal in such a way that the total field is nulled
    # and the coronograph is optimized.
    # A simple way to do this is to compute the sqrt of the ratio of intensities
    optimal_outer_transmission = np.sqrt(wf_list_in[-2].total_intensity / wf_list_out[-2].total_intensity)

    osys_opt, psf_opt, wf_list_opt = coronograph_system(source_offset_arcsec=0, source_flux=1,
                                             use_coronograph=True,
                                             mask_transmission=1.0,
                                             outer_transmission=optimal_outer_transmission,
                                             radius_mask_lambda_over_d=radius_mask_lambda_over_d)

    def _display():    
        plt.figure()
        cc=wf_list_in[-3].shape[0]//2
        plt.plot(wf_list_in[-3].amplitude[cc,:], color='blue', label='from inner mask (before exit pupil stop)')
        plt.plot(wf_list_out[-3].amplitude[cc,:], color='orange', label='from outer part (before exit pupil stop)')
        plt.plot(wf_list_out[-3].amplitude[cc,:]*optimal_outer_transmission, '--', color='orange', label='outer part attenuated')
        #plt.plot(wf_list_no[-2].amplitude[cc,:], color='green', label='total field no coro (after exit pupil stop)')
        plt.plot(wf_list_opt[-2].amplitude[cc,:], color='red', label='total field opt coro (after exit pupil stop)')
        plt.legend()
        plt.title("Amplitude in the output pupil")
        plt.ylabel("Amplitude")
        plt.show()
        
        plt.figure()
        poppy.display_psf(psf_opt, title="PSF with Ideal Coronograph", scale='log', vmin=1e-16)
        plt.show()

        radius_opt, profile_opt, ee_opt = poppy.radial_profile(psf_opt, ee=True, ext=0)
        radius_no, profile_no, ee_no = poppy.radial_profile(psf_no, ee=True, ext=0)
        plt.figure(figsize=(8, 8))
        plt.suptitle("Radial Profile & Encircled Energy")

        # Radial profile (scala logaritmica)
        plt.subplot(2, 1, 1)
        plt.semilogy(radius_opt, profile_opt, label='Coronografo', color='red')
        plt.semilogy(radius_no, profile_no, label='Senza coronografo', color='blue')
        plt.ylabel("Radial Profile [log scale]")
        plt.xlabel("Radius [arcsec]")
        plt.legend()
        plt.grid(True, which='both', linestyle='--', alpha=0.5)

        # Encircled Energy (scala lineare)
        plt.subplot(2, 1, 2)
        plt.plot(radius_opt, ee_opt, label='Coronografo', color='red')
        plt.plot(radius_no, ee_no, label='Senza coronografo', color='blue')
        plt.ylabel("Encircled Energy [linear scale]")
        plt.xlabel("Radius [arcsec]")
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.5)

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])


    print(f"Optimal outer transmission: {optimal_outer_transmission:.4f}")
    print(f"Flux in PSF without coronagraph: {psf_no[0].data.sum():.4e}")
    print(f"Flux in PSF from inner mask: {psf_in[0].data.sum():.4e}")
    print(f"Flux in PSF from outer part: {psf_out[0].data.sum():.4e}")
    print(f"Flux in PSF with Ideal Coronograph: {psf_opt[0].data.sum():.4e}")
    _display()
    return osys_no, psf_no, wf_list_no, osys_in, psf_in, wf_list_in, osys_out, psf_out, wf_list_out, osys_opt, psf_opt, wf_list_opt


if __name__ == "__main__":
    main()