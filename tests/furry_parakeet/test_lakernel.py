"""Tests for lakernel.py"""

import furry_parakeet.pyimcom_lakernel as lk
import numpy as np
from furry_parakeet import pyimcom_croutines


def test_get_coadd_matrix_discrete():
    """Test function for get_coadd_matrix_discrete."""

    # number of outputs to print
    # npr = 4

    # number of layers to test multi-ouptut
    nt = 3

    sigma = 1.75
    # u = numpy.array([0.2, 0.1])

    # test grid: interpolate an m1xm1 image from n1xn1
    m1 = 50
    n1 = 80
    n = n1 * n1
    m = m1 * m1

    x = np.zeros((n,))
    y = np.zeros((n,))
    for i in range(n1):
        y[n1 * i : n1 * i + n1] = i
        x[i::n1] = i
    xout = np.zeros((m,))
    yout = np.zeros((m,))
    for i in range(m1):
        yout[m1 * i : m1 * i + m1] = 5 + 0.25 * i
        xout[i::m1] = 5 + 0.25 * i

    # make sample image
    # thisImage = numpy.exp(2 * numpy.pi * 1j * (u[0] * x + u[1] * y))
    # desiredOutput = numpy.exp(2 * numpy.pi * 1j * (u[0] * xout + u[1] * yout))

    A = np.zeros((n, n))
    mBhalf = np.zeros((m, n))
    mBhalfPoly = np.zeros((nt, m, n))
    C = np.ones((nt,))
    for i in range(n):
        for j in range(n):
            A[i, j] = np.exp(-1.0 / sigma**2 * ((x[i] - x[j]) ** 2 + (y[i] - y[j]) ** 2))
        for a in range(m):
            mBhalf[a, i] = np.exp(-1.0 / sigma**2 * ((x[i] - xout[a]) ** 2 + (y[i] - yout[a]) ** 2))
            for k in range(nt):
                mBhalfPoly[k, a, i] = np.exp(
                    -1.0 / sigma**2 / (0.5 + 0.5 * 1.05**k) * ((x[i] - xout[a]) ** 2 + (y[i] - yout[a]) ** 2)
                )
    for k in range(nt):
        C[k] = (1 + 1.05**k) ** 2 / 4.0 / 1.05**k

    # rescale everything
    A *= 0.7
    mBhalf *= 0.7
    mBhalfPoly *= 0.7
    C *= 0.7

    # fits.PrimaryHDU(A).writeto("A.fits", overwrite=True)
    # fits.PrimaryHDU(mBhalfPoly).writeto("mBhalf.fits", overwrite=True)
    print("C=", C)

    kappa_array = np.logspace(-6, -2, 3)
    print("kappa_array=", kappa_array)

    print("n", n, "m", m, "nt", nt, "nv", np.size(kappa_array))

    (kappa_, Sigma_, UC_, T_) = lk.get_coadd_matrix_discrete(
        A, mBhalfPoly, C, kappa_array, smax=0.5, ucmin=1.0e-6
    )

    # print information
    # fits.PrimaryHDU(T_).writeto("T.fits", overwrite=True)
    # fits.PrimaryHDU(UC_).writeto("UC.fits", overwrite=True)
    # fits.PrimaryHDU(Sigma_).writeto("Sigma.fits", overwrite=True)
    # fits.PrimaryHDU(kappa_).writeto("kappa_ind.fits", overwrite=True)

    print(np.shape(kappa_))
    print(np.shape(Sigma_))
    print(np.shape(UC_))
    print(np.shape(T_))
    print(np.amin(kappa_), np.amax(kappa_))
    print(np.amin(Sigma_), np.amax(Sigma_))
    print(np.amin(UC_), np.amax(UC_))
    print(np.amin(T_), np.amax(T_))

    assert np.shape(kappa_) == (3, 2500)
    assert np.shape(Sigma_) == (3, 2500)
    assert np.shape(UC_) == (3, 2500)
    assert np.shape(T_) == (3, 2500, 6400)
    assert np.amin(kappa_) > 0.005
    assert np.amax(kappa_) < 0.012
    assert np.amin(Sigma_) > 0.49
    assert np.amax(Sigma_) < 0.6
    assert np.amin(UC_) > 5.0e-4
    assert np.amax(UC_) < 0.002
    T__ = np.sum(T_, axis=-1)

    print(np.amin(T__), np.amax(T__))
    assert np.amax(np.abs(T__ - 1.0)) < 0.06

    # test for what happens if A isn't positive definite
    (kappa2_, Sigma2_, UC2_, T2_) = lk.get_coadd_matrix_discrete(
        A - 0.005 * np.identity(n), mBhalfPoly, C, kappa_array, smax=0.5, ucmin=1.0e-6
    )
    print(np.amin(kappa2_), np.amax(kappa2_))
    assert np.amin(kappa2_) > 0.005
    assert np.amax(kappa2_) < 0.012
    T__ = np.sum(T2_, axis=-1)
    print(np.amin(T__), np.amax(T__))
    assert np.amax(np.abs(T__ - 1.0)) < 0.06


def test_kernel():
    """Test case for the kernel."""

    # This is nothing fancy.  The test interpolates an image containing a single sine wave, with Gaussian PSF.

    # Test parameters
    sigma = 4.0  # The 1 sigma width of PSF (Gaussian)
    u = np.array([0.2, 0.1])  # Shape (2,). Fourier wave vector of sine wave, (x,y) ordering.

    # number of outputs to print
    npr = 4

    # number of layers to test multi-ouptut
    nt = 3

    # test grid: interpolate an m1xm1 image from n1xn1
    m1 = 25
    n1 = 33
    n = n1 * n1
    m = m1 * m1

    x = np.zeros((n,))
    y = np.zeros((n,))
    for i in range(n1):
        y[n1 * i : n1 * i + n1] = i
        x[i::n1] = i
    xout = np.zeros((m,))
    yout = np.zeros((m,))
    for i in range(m1):
        yout[m1 * i : m1 * i + m1] = 5 + 0.25 * i
        xout[i::m1] = 5 + 0.25 * i

    # make sample image
    thisImage = np.exp(2 * np.pi * 1j * (u[0] * x + u[1] * y))
    desiredOutput = np.exp(2 * np.pi * 1j * (u[0] * xout + u[1] * yout))

    A = np.zeros((n, n))
    mBhalf = np.zeros((m, n))
    mBhalfPoly = np.zeros((nt, m, n))
    C = 1.0
    for i in range(n):
        for j in range(n):
            A[i, j] = np.exp(-1.0 / sigma**2 * ((x[i] - x[j]) ** 2 + (y[i] - y[j]) ** 2))
        for a in range(m):
            mBhalf[a, i] = np.exp(-1.0 / sigma**2 * ((x[i] - xout[a]) ** 2 + (y[i] - yout[a]) ** 2))
            for k in range(nt):
                mBhalfPoly[k, a, i] = np.exp(
                    -1.0 / (1.05**k * sigma) ** 2 * ((x[i] - xout[a]) ** 2 + (y[i] - yout[a]) ** 2)
                )

    # rescale everything
    A *= 0.7
    mBhalf *= 0.7
    mBhalfPoly *= 0.7
    C *= 0.7

    # brute force version of kernel
    (kappa, Sigma, UC, T) = lk.BruteForceKernel(A, mBhalf, C, 1e-8)

    print("** brute force kernel **")
    print("kappa =", kappa[:npr])
    print("Sigma =", Sigma[:npr])
    print("UC =", UC[:npr])
    print("Image residual =")
    print(np.abs(T @ thisImage - desiredOutput).reshape((m1, m1))[:npr])

    # C version of kernel
    (kappa2, Sigma2, UC2, T2) = lk.CKernel(A, mBhalf, C, 1e-8)

    print("** C kernel **")
    print("kappa =", kappa2[:npr])
    print("Sigma =", Sigma2[:npr])
    print("UC =", UC2[:npr])
    print("Image residual =")
    print(np.abs(T2 @ thisImage - desiredOutput).reshape((m1, m1))[:npr])

    t_err = np.abs(T2 @ thisImage - desiredOutput).reshape((m1, m1))
    assert np.amax(t_err) < 1.0e-3

    (kappa3, Sigma3, UC3, T3) = lk.CKernelMulti(
        A, mBhalfPoly, C * 1.05 ** (2 * np.array(range(nt))), 1e-8 * np.ones((nt,))
    )
    print("Sigma3 =", Sigma3[:, :npr])
    print("output =", (T2 @ thisImage)[:npr], (T3 @ thisImage)[:, :npr])
    t_err = np.abs((T3 @ thisImage)[0, :] - desiredOutput).reshape((m1, m1))
    print(np.shape(t_err), np.amax(t_err), np.amax(t_err[:npr]))
    assert np.amax(t_err) < 1.0e-3

    # Test for single array
    (kappa4, Sigma4, UC4, T4) = lk.CKernelMulti(A, mBhalfPoly[0, :, :], C, 1e-8)
    err = np.abs(T4 - T3[0, :, :])
    print(np.amax(err))
    assert np.amax(err) < 1.0e-4


def test_interp():
    """Test interpolation functions on a sine wave."""

    u = np.array([0.2, 0.1])  # Shape (2,). Fourier wave vector of sine wave, (x,y) ordering.

    ny = 1024
    nx = 1024
    indata = np.zeros((3, ny, nx))
    indata[0, :, :] = 1.0
    for ix in range(nx):
        indata[1, :, ix] = u[0] * ix + u[1] * np.linspace(0, ny - 1, ny)
    indata[2, :, :] = np.cos(2 * np.pi * indata[1, :, :])

    no = 32768
    xout = np.linspace(8, 9, no)
    yout = np.linspace(10, 10.5, no)

    fout = np.zeros((3, no))

    pyimcom_croutines.iD5512C(indata, xout, yout, fout)
    # pyimcom_croutines.iD5512C(indata[2,:,:].reshape((1,ny,nx)), xout, yout, fout[2,:].reshape((1,no)))

    pred = u[0] * xout + u[1] * yout

    # print(fout)
    # print(pred)
    # print(numpy.cos(2*numpy.pi*pred))
    print("errors:")
    print(np.amax(np.abs(fout[0, :] - 1)))
    print(np.amax(np.abs(fout[1, :] - pred)))
    print(np.amax(np.abs(fout[2, :] - np.cos(2 * np.pi * pred))))

    # See if we got the right answer for the interpolated image and sine wave
    assert np.amax(np.abs(fout[0, :] - 1)) < 1.0e-7
    assert np.amax(np.abs(fout[1, :] - pred)) < 1.0e-7
    assert np.amax(np.abs(fout[2, :] - np.cos(2 * np.pi * pred))) < 3.0e-4
