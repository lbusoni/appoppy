import numpy as np
from scipy.ndimage import center_of_mass
from arte.types.mask import CircularMask


def mask_from_median(image, cut):
    imask = image < np.median(image) / cut
    mask = np.zeros(image.shape)
    mask[imask] = 1
    return mask


def sector_mask(shape, angle_range, centre=None, radius=None):
    """
    Return a boolean mask for a circular sector. 
    The start/stop angles in
    `angle_range` should be given in counterclockwise order, from East

    angle_range=(0, 90) goes from East to North
    angle_range=(-150, 150) includes S, E. N and not W

    angle_range=(150, -150) raises ValueError
    """

    x, y = np.ogrid[:shape[0],:shape[1]]
    if centre is None:
        centre = (shape[0] / 2, shape[1] / 2)
    if radius is None:
        radius = np.min(centre)
    cx, cy = centre
    tmin, tmax = np.deg2rad(angle_range)

    # ensure stop angle > start angle
    if tmax < tmin:
        # tmax += 2 * np.pi
        raise ValueError(
            "angle_range must be given in increasing order. Got %s" % 
            str(angle_range))

    # convert cartesian --> polar coordinates
    r2 = (x - cx) * (x - cx) + (y - cy) * (y - cy)
    theta = np.arctan2(x - cx, y - cy) - tmin

    # wrap angles between 0 and 2*pi
    theta %= (2 * np.pi)

    # circular mask
    circmask = r2 <= radius * radius

    # angular mask
    anglemask = theta <= (tmax - tmin)

    return circmask * anglemask


def mask_phase_screen(phase_screen, angle_range):
    smask1 = sector_mask(phase_screen.shape,
                        (angle_range[0], angle_range[1]))
    mask = np.ma.mask_or(phase_screen.mask, ~smask1)
    return np.ma.masked_array(
        phase_screen, mask=np.broadcast_to(
            mask, phase_screen.shape))


def mask_from_threshold(ima, threshold):
    '''
    threshold: float
        Percentage of maximum value.
    '''
    cut = threshold * ima.max()
    one_bit_ima = np.zeros(ima.shape)
    one_bit_ima[ima >= cut] = 1
    
    def get_n_pixels():
        n_pixels_above_cut = ima[one_bit_ima == 1].shape[0]
        r = np.sqrt(n_pixels_above_cut / np.pi)
        y, x = center_of_mass(one_bit_ima)
        mask = CircularMask(frameShape=ima.shape,
                            maskRadius=r-0.5,
                            maskCenter=[y, x])
        changed_pixels = ima[np.logical_and(mask.mask()==False, one_bit_ima==0)].shape[0]
        one_bit_ima[np.logical_and(mask.mask()==False, one_bit_ima==0)] = 1 
        return changed_pixels, mask

    while True:
        n_pix, mask = get_n_pixels() 
        if n_pix == 0:
            break
        
    return mask