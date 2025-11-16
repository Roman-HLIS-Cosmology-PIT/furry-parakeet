import numpy as np
from furry_parakeet import pyimcom_croutines


def test_spots():
    """Test function for interpolating spots."""

    N = 128
    theta = np.pi / 180.0 * 17.0
    x_, y_ = np.meshgrid(np.arange(N), np.arange(N))
    u = np.cos(theta) * (x_ - (N / 2.0)) + np.sin(theta) * (y_ - (N / 2.0))

    v = np.arange(265**2)
    k1 = (60 * v) % 265
    k2 = (221 * v) % 265

    print(k1[:6], k2[:6])

    xout = N * (0.1 + 0.8 * k1 / 265.0)
    yout = N * (0.1 + 0.8 * k2 / 265.0)
    u_true = np.cos(theta) * (xout - (N / 2.0)) + np.sin(theta) * (yout - (N / 2.0))

    # Lorentzian
    for q in range(1, 9):
        infunc = 1.0 / (1.0 + (u / q) ** 2)
        infunc = infunc[None, :]
        out1 = np.zeros((1, len(v)))
        out2 = np.zeros_like(out1)
        pyimcom_croutines.iD5512C(infunc, xout, yout, out1)
        pyimcom_croutines.iG4460C(infunc, xout, yout, out2)
        diff = np.amax(np.abs(out1 - out2))
        print(diff)
        assert diff < 0.005 / 5.0 ** (q - 1) + 3e-7
    print("")

    # Trig
    for q in range(1, 13):
        infunc = np.sin(2.0 * np.pi * u * (q / 24.0))
        infunc = infunc[None, :]
        out1 = np.zeros((1, len(v)))
        out2 = np.zeros_like(out1)
        pyimcom_croutines.iD5512C(infunc, xout, yout, out1)
        pyimcom_croutines.iG4460C(infunc, xout, yout, out2)
        diff = np.amax(np.abs(out1 - out2))
        diff2 = np.amax(np.abs(out1 - np.sin(2.0 * np.pi * u_true * (q / 24.0))))
        print(diff, diff2)
        assert np.log10(diff) < min(-1.4, -6.5 + 0.75 * q)


def test_sym():
    """Test for symmetrical interpolation routines."""

    N = 128
    theta = np.pi / 180.0 * 17.0
    x_, y_ = np.meshgrid(np.arange(N), np.arange(N))
    u = np.cos(theta) * (x_ - (N / 2.0)) + np.sin(theta) * (y_ - (N / 2.0))

    v = np.arange(13**2)
    k1 = (5 * v) % 13
    k2 = (12 * v) % 13
    xout = N / 2.0 + 3.6 * (k1[None, :] - k1[:, None]).ravel()
    yout = N / 2.0 + 3.6 * (k2[None, :] - k2[:, None]).ravel()

    for q in range(1, 13):
        infunc = np.sin(2.0 * np.pi * u * (q / 24.0))
        infunc = infunc[None, :]
        out1 = np.zeros((1, len(xout)))
        out2 = np.zeros_like(out1)
        pyimcom_croutines.iD5512C_sym(infunc, xout, yout, out1)
        pyimcom_croutines.iG4460C_sym(infunc, xout, yout, out2)
        diff = np.amax(np.abs(out1 - out2))
        print(diff)
        assert np.log10(diff) < min(-1.4, -6.5 + 0.75 * q)


def test_grid():
    """Test for symmetrical interpolation routines."""

    N = 128
    theta = np.pi / 180.0 * 17.0
    x_, y_ = np.meshgrid(np.arange(N), np.arange(N))
    u = np.cos(theta) * (x_ - (N / 2.0)) + np.sin(theta) * (y_ - (N / 2.0))

    xout = np.linspace(N / 4.0, 3 * N / 4.0, 100)[None, :]
    yout = np.linspace(N / 4.0, 3 * N / 4.0, 100)[None, :]

    for q in range(1, 13):
        infunc = np.sin(2.0 * np.pi * u * (q / 24.0))
        out1 = np.zeros((1, np.size(xout) * np.size(yout)))
        out2 = np.zeros_like(out1)
        pyimcom_croutines.gridD5512C(infunc, xout, yout, out1)
        pyimcom_croutines.gridG4460C(infunc, xout, yout, out2)
        diff = np.amax(np.abs(out1 - out2))
        print(diff)
        assert np.log10(diff) < min(-1.3, -6.5 + 0.75 * q)
