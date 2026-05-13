"""Tests for interpolation routines."""

import numpy as np
from furry_parakeet import pyimcom_croutines, pyimcom_interface


def _f(x, y):
    return np.cos(np.pi * y / 25.0) + np.sin(2.0 * np.pi * x / 30.0)


def test_interp():
    """Compares interpolations both ways."""

    # rows & columns
    r = 25
    c = 30

    g = np.ones((r, c))
    u, v = np.meshgrid(np.arange(c), np.arange(r))
    image_in = _f(u, v)
    x = 14.0 + (u + np.sqrt(3) * v) / 2.0 * 0.25
    y = 12.0 + (np.sqrt(3) * u - v) / 2.0 * 0.25
    xy = np.zeros((r, c, 2))
    xy[:, :, 0] = y
    xy[:, :, 1] = x
    image_out = np.zeros((r, c))
    pyimcom_croutines.bilinear_interpolation(image_in, g, r, c, xy.reshape((-1, 2)), r * c, image_out)
    err = np.amax(np.abs(image_out - _f(x, y)))
    print(err)
    assert err < 0.015

    image_in32 = image_in.astype(np.float32)
    xy32 = xy.astype(np.float32)
    image_out32 = np.zeros((r, c), dtype=np.float32)
    pyimcom_croutines.bilinear_interpolation32(
        image_in32, g.astype(np.float32), r, c, xy32.reshape((-1, 2)), r * c, image_out32
    )
    err = np.amax(np.abs(image_out - image_out32))
    print(err)
    assert err < 5.0e-5

    image_out32[:, :] = 0.0
    pyimcom_interface.bilinear_interpolation(image_in32, g, xy, image_out32)
    err = np.amax(np.abs(image_out - image_out32))
    print(err)
    assert err < 5.0e-5

    image_orig = np.zeros((r, c))
    dots = np.zeros((r, c))
    dots[2, 8] = 1.0
    print("y=", xy[2, 8, 0])
    print("x=", xy[2, 8, 1])
    pyimcom_croutines.bilinear_transpose(dots, r, c, xy.reshape((-1, 2)), r * c, image_orig)
    for i in range(r):
        for j in range(c):
            if np.abs(image_orig[i, j]) > 1.0e-6:
                print(f"{i:2d} {j:2d} {image_orig[i, j]:12.7f}")
                assert i == 13 or i == 14
                assert j == 15 or j == 16
    assert 0.99 < np.sum(image_orig) < 1.01

    image_orig32 = np.zeros((r, c), dtype=np.float32)
    dots32 = np.zeros((r, c), dtype=np.float32)
    dots32[2, 8] = 1.0
    pyimcom_croutines.bilinear_transpose32(dots32, r, c, xy32.reshape((-1, 2)), r * c, image_orig32)
    err = np.amax(np.abs(image_orig - image_orig32))
    print(err)
    assert err < 1.0e-4

    image_orig32[:, :] = 0.0
    pyimcom_interface.bilinear_transpose(dots32, xy, image_orig32)
    err = np.amax(np.abs(image_orig - image_orig32))
    print(err)
    assert err < 1.0e-4
