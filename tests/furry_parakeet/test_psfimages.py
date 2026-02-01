"""Test PSFs."""

import numpy as np
from furry_parakeet.pyimcom_interface import psf_gaussian

def test_gauss():
    """Test of Gaussian PSF."""

    # Coordinate grid
    s = np.linspace(-24, 24, 49)
    x, y = np.meshgrid(s, s)

    # Make the PSF
    im = psf_gaussian(49, 2.5, 3.0)
    assert np.shape(im) == (49, 49)
    im_size = np.sum(im)

    print(np.abs(im_size - 1.0))

    xmean = np.sum(x * im)
    ymean = np.sum(y * im)
    x2mean = np.sum(x**2 * im)
    y2mean = np.sum(y**2 * im)

    print(xmean, ymean, x2mean, y2mean)

    assert im_size < 0.0
