import numpy as np


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
    smask1 = sector_mask(phase_screen[0].shape,
                        (angle_range[0], angle_range[1]))
    mask = np.ma.mask_or(phase_screen[0].mask, ~smask1)
    return np.ma.masked_array(
        phase_screen, mask=np.broadcast_to(
            mask, phase_screen.shape))
