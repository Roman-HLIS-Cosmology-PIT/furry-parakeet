"""
Interface functions (mostly to be called from pyimcom, but could be from somewhere else).

Functions
---------
psf_gaussian
    Makes a Gaussian spot.
psf_simple_airy
    Makes an possibly obstructed Airy spot, convolved with tophat and Gaussian.
psf_cplx_airy
    Complex Airy function with some features printed on.
rotation
    Rotation matrix in 2D.
shearmatrix
    Makes a shear matrix with unit determinant.
get_coadd_matrix
    Convert PSF overlap object to transfer matrices.

Classes
-------
PSF_Overlap
    Correlation matrices of all the PSFs.

"""

import time

import numpy as np
import scipy
import scipy.fft
import scipy.signal
import scipy.special

from . import pyimcom_croutines, pyimcom_lakernel

##################################################
### Simple PSF models (for testing or outputs) ###
##################################################


def psf_gaussian(n, sigmax, sigmay):
    """
    Makes a Gaussian spot.

    Parameters
    ----------
    n : int
        Size of the spot.
    sigmax : float
        Standard deviation along the x-axis.
    sigmay : float
        Standard deviation along the y-axis.

    Returns
    -------
    np.array
        2D image, shape =(`n`, `n`), normalized to unity.

    """

    xa = np.linspace((1 - n) / 2, (n - 1) / 2, n)
    x = np.zeros((n, n))
    y = np.zeros((n, n))
    x[:, :] = xa[None, :]
    y[:, :] = xa[:, None]

    I_ = np.exp(-0.5 * (x**2 / sigmax**2 + y**2 / sigmay**2)) / (2.0 * np.pi * sigmax * sigmay)

    return I_


def psf_simple_airy(n, ldp, obsc=0.0, tophat_conv=0.0, sigma=0.0):
    """
    Makes an possibly obstructed Airy spot, convolved with tophat and Gaussian.

    Parameters
    ----------
    n : int
        Size of the spot.
    ldp : float
        Diffraction width, lambda / D, in units of the pixel.
    obsc : float, optional
        Fractional linear obscuration, default = unobstructed.
    tophat_conv : float, optional
        Full width of square tophat, in units of the pixel.
    sigma : float, optional
        Standard deviation per axis of the Gaussian contribution, in units of the pixel.

    Returns
    -------
    np.array
        2D image, shape =(`n`, `n`), normalized to unity if analytically extended.

    Notes
    -----
    The output is centered on ``(n-1)/2,(n-1)/2`` (so on a pixel if
    `n` is odd and a corner if `n` is even).

    """

    # figure out pad size -- want to get to at least a tophat width and 6 sigmas
    kp = 1 + int(np.ceil(tophat_conv + 6 * sigma))
    npad = n + 2 * kp

    xa = np.linspace((1 - npad) / 2, (npad - 1) / 2, npad)
    x = np.zeros((npad, npad))
    y = np.zeros((npad, npad))
    x[:, :] = xa[None, :]
    y[:, :] = xa[:, None]
    r = np.sqrt(x**2 + y**2) / ldp  # r in units of ldp

    # make Airy spot
    I_ = (
        (
            scipy.special.jv(0, np.pi * r)
            + scipy.special.jv(2, np.pi * r)
            - obsc**2 * (scipy.special.jv(0, np.pi * r * obsc) + scipy.special.jv(2, np.pi * r * obsc))
        )
        ** 2
        / (4.0 * ldp**2 * (1 - obsc**2))
        * np.pi
    )

    # now convolve
    It = np.fft.fft2(I_)
    uxa = np.linspace(0, npad - 1, npad) / npad
    uxa[-(npad // 2) :] -= 1
    ux = np.zeros((npad, npad))
    uy = np.zeros((npad, npad))
    ux[:, :] = uxa[None, :]
    uy[:, :] = uxa[:, None]
    It *= (
        np.exp(-2 * np.pi**2 * sigma**2 * (ux**2 + uy**2))
        * np.sinc(ux * tophat_conv)
        * np.sinc(uy * tophat_conv)
    )
    I_ = np.real(np.fft.ifft2(It))

    return I_[kp:-kp, kp:-kp]


def psf_cplx_airy(n, ldp, tophat_conv=0.0, sigma=0.0, features=0):
    """
    Complex Airy function with some features printed on.

    Everything is band limited.

    Parameters
    ----------
    n : int
        Size of the spot.
    ldp : float
        Diffraction width, lambda / D, in units of the pixel.
    obsc : float, optional
        Fractional linear obscuration, default = unobstructed.
    tophat_conv : float, optional
        Full width of square tophat, in units of the pixel.
    sigma : float, optional
        Standard deviation per axis of the Gaussian contribution, in units of the pixel.
    features : int, optional
        If specified, determines what features get added.

    Returns
    -------
    np.array
        2D image, shape =(`n`, `n`), normalized to unity if analytically extended.

    Notes
    -----
    The output is centered on ``(n-1)/2,(n-1)/2`` (so on a pixel if
    `n` is odd and a corner if `n` is even).

    """

    # figure out pad size -- want to get to at least a tophat width and 6 sigmas
    kp = 1 + int(np.ceil(tophat_conv + 6 * sigma))
    npad = n + 2 * kp

    xa = np.linspace((1 - npad) / 2, (npad - 1) / 2, npad)
    x = np.zeros((npad, npad))
    y = np.zeros((npad, npad))
    x[:, :] = xa[None, :]
    y[:, :] = xa[:, None]
    r = np.sqrt(x**2 + y**2) / ldp  # r in units of ldp
    phi = np.arctan2(y, x)

    # make modified Airy spot
    L1 = 0.8
    L2 = 0.01
    f = L1 * L2 * 4.0 / np.pi
    II = scipy.special.jv(0, np.pi * r) + scipy.special.jv(2, np.pi * r)
    for t in range(6):
        II -= (
            f
            * np.sinc(L1 * r * np.cos(phi + t * np.pi / 6.0))
            * np.sinc(L2 * r * np.sin(phi + t * np.pi / 6.0))
        )
    I_ = II**2 / (4.0 * ldp**2 * (1 - 6 * f)) * np.pi

    if features % 2 == 1:
        rp = np.sqrt((x - 1 * ldp) ** 2 + (y + 2 * ldp) ** 2) / 2.0 / ldp
        II = scipy.special.jv(0, np.pi * rp) + scipy.special.jv(2, np.pi * rp)
        I_ = 0.8 * I_ + 0.2 * II**2 / (4.0 * (2.0 * ldp) ** 2) * np.pi

    if (features // 2) % 2 == 1:
        Icopy = np.copy(I_)
        I_ *= 0.85
        I_[:-8, :] += 0.15 * Icopy[8:, :]

    if (features // 4) % 2 == 1:
        Icopy = np.copy(I_)
        I_ *= 0.8
        I_[:-4, :-4] += 0.1 * Icopy[4:, 4:]
        I_[4:, :-4] += 0.1 * Icopy[:-4, 4:]

    # now convolve
    It = np.fft.fft2(I_)
    uxa = np.linspace(0, npad - 1, npad) / npad
    uxa[-(npad // 2) :] -= 1
    ux = np.zeros((npad, npad))
    uy = np.zeros((npad, npad))
    ux[:, :] = uxa[None, :]
    uy[:, :] = uxa[:, None]
    It *= (
        np.exp(-2 * np.pi**2 * sigma**2 * (ux**2 + uy**2))
        * np.sinc(ux * tophat_conv)
        * np.sinc(uy * tophat_conv)
    )
    I_ = np.real(np.fft.ifft2(It))

    return I_[kp:-kp, kp:-kp]


def rotmatrix(theta):
    """
    Rotation matrix in 2D.

    Parameters
    ----------
    theta : float
        Rotation angle in radians.

    Returns
    -------
    np.array of float
        Shape (2, 2) rotation matrix.

    Notes
    -----
    The convention is that ``X(stacking frame) = (distortion_matrix) @ X(native frame)``.

    In ds9 the native frame is angle `theta` *clockwise* from stacking frame
    (i.e., `theta` is position angle of the native frame).

    """

    return np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])


def shearmatrix(g1, g2):
    """
    Makes a shear matrix with unit determinant.

    Parameters
    ----------
    g1 : float
       The "stretch along x-axis" component of shear.
    g2 : float
       The "stretch along y=x" component of shear.

    Returns
    -------
    np.array of float
        Shape (2, 2) rotation matrix.

    """

    return np.array([[1 - g1, -g2], [-g2, 1 + g1]]) / np.sqrt(1.0 - g1**2 - g2**2)


########################################################
### Now the main functions that we use for real PSFs ###
########################################################


class PSF_Overlap:
    """
    Correlation matrices of all the PSFs.

    Parameters
    ----------
    psf_in_list : list of np.array
        The list of input PSFs (length ``n_in``).
    psf_out_list : list of np.array
        The list of output PSFs (length ``n_out``).
    dsample : float
        Sample spacing of the correlation matrices in terms of input pixels.
    nsample : int
        Number of output samples on each axis. Must be odd.
    s_in : float
        Input reference pixel scale.
    distort_matrices: list of np.array, optional
        The 2x2 distortion matrices associated with the input PSFs (length ``n_in``),
        in the form ``X(stacking frame) = (distortion_matrix[j]) @ X(native frame[j])``.
        Default is no distortion.
    amp_penalty : float, optional
        Experimental feature to change the weighting of Fourier modes
       (do not use or set to None unless you are trying to do a test with it).

    Attributes
    ----------
    n_in : int
        Number of input PSFs.
    n_out : int
        Number of output PSFs.
    n_tot : int
        Total number of PSFs, equal to `n_in` + `n_out`.
    psfarray : np.array
        3D array of PSFs, distorted if needed, shape (`n_tot`, ``ns2``, ``ns2``)
        (the padding size ``ns2`` is determined internally).
    overlaparray : np.array
        4D overlap array, shape (`n_tot`, `n_tot`, `nsample`, `nsample`)
    C : np.array
        Square norms of the output PSFs, shape (``n_out``).

    """

    def __init__(
        self, psf_in_list, psf_out_list, dsample, nsample, s_in, distort_matrices=None, amp_penalty=None
    ):
        # checking
        if nsample % 2 == 0:
            raise Exception(f"error in PSF_Overlap, nsample={nsample:d} must be odd")

        self.s_in = s_in

        tm0 = time.perf_counter()

        # number of PSFs
        self.n_in = len(psf_in_list)
        self.n_out = len(psf_out_list)
        self.n_tot = self.n_in + self.n_out
        # save metadata
        self.dsample = dsample
        self.nsample = nsample
        self.nc = nsample // 2
        # ... and make an array for the PSF overlaps
        self.overlaparray = np.zeros((self.n_tot, self.n_tot, nsample, nsample))

        # make an ns2 x ns2 grid onto which we will interpolate the PSF
        kpad = 5
        ns2 = nsample + 2 * kpad
        self.psf_array = np.zeros((self.n_tot, ns2, ns2))
        xoa = np.linspace((1 - ns2) / 2, (ns2 - 1) / 2, ns2) * dsample
        xo = np.zeros((ns2, ns2))
        yo = np.zeros((ns2, ns2))
        xo[:, :] = xoa[None, :]
        yo[:, :] = xoa[:, None]
        p = 0  # pad size
        #
        # now the interpolation
        for ipsf in range(self.n_in):
            # get center
            ny = np.shape(psf_in_list[ipsf])[0]
            nx = np.shape(psf_in_list[ipsf])[1]
            xctr = (nx - 1) / 2.0
            yctr = (ny - 1) / 2.0

            # detM = 1.0
            if distort_matrices is not None:
                if distort_matrices[ipsf] is not None:
                    M = np.linalg.inv(distort_matrices[ipsf])
                    # detM = numpy.linalg.det(M)
                    xco = M[0, 0] * xo + M[0, 1] * yo
                    yco = M[1, 0] * xo + M[1, 1] * yo
                else:
                    xco = np.copy(xo)
                    yco = np.copy(yo)
            else:
                xco = np.copy(xo)
                yco = np.copy(yo)
            out_array = np.zeros((1, ns2 * ns2))
            pyimcom_croutines.iD5512C(
                np.pad(psf_in_list[ipsf], p).reshape((1, ny + 2 * p, nx + 2 * p)),
                xco.flatten() + xctr + p / 2,
                yco.flatten() + yctr + p / 2,
                out_array,
            )
            self.psf_array[ipsf, :, :] = out_array.reshape((ns2, ns2))

        tm1 = time.perf_counter()

        # ... and the output PSFs
        for ipsf in range(self.n_out):
            # get center
            ny = np.shape(psf_out_list[ipsf])[0]
            nx = np.shape(psf_out_list[ipsf])[1]
            xctr = (nx - 1) / 2.0
            yctr = (ny - 1) / 2.0
            out_array = np.zeros((1, ns2 * ns2))
            pyimcom_croutines.iD5512C(
                np.pad(psf_out_list[ipsf], p).reshape((1, ny + 2 * p, nx + 2 * p)),
                xo.flatten() + xctr + p / 2,
                yo.flatten() + yctr + p / 2,
                out_array,
            )
            self.psf_array[self.n_in + ipsf, :, :] = out_array.reshape((ns2, ns2))

        tm2 = time.perf_counter()

        # FFT based method for overlaps
        nfft = nfft0 = 2 ** (1 + int(np.ceil(np.log2(ns2 - 0.5))))
        if 7 * nfft0 // 8 > 2 * ns2:
            nfft = 7 * nfft0 // 8
        if 3 * nfft0 // 4 > 2 * ns2:
            nfft = 3 * nfft0 // 4
        if 5 * nfft0 // 8 > 2 * ns2:
            nfft = 5 * nfft0 // 8
        b = nsample // 2
        #
        # this replaces the following simpler but more time-consuming code:
        # psf_array_pad = numpy.zeros((self.n_tot,nfft,nfft))
        # psf_array_pad[:,:ns2,:ns2] = self.psf_array
        # psf_array_pad_fft = numpy.fft.fft2(psf_array_pad)
        #
        psf_array_pad_fft = np.zeros((self.n_tot, nfft, nfft), dtype=np.complex128)
        if self.n_tot >= 2:
            # do these in pairs
            arC_ft_rev = np.zeros((nfft, nfft), dtype=np.complex128)
            for j in range(self.n_tot // 2):
                arC = np.zeros((nfft, nfft), dtype=np.complex128)
                arC[:ns2, :ns2].real = 0.5 * self.psf_array[2 * j, :, :]
                arC[:ns2, :ns2].imag = 0.5 * self.psf_array[2 * j + 1, :, :]
                arC_ft = scipy.fft.fft2(arC)
                arC_ft_rev[0, 0] = arC_ft[0, 0]
                arC_ft_rev[0, 1:] = arC_ft[0, :0:-1]
                arC_ft_rev[1:, 0] = arC_ft[:0:-1, 0]
                arC_ft_rev[1:, 1:] = arC_ft[:0:-1, :0:-1]
                psf_array_pad_fft[2 * j, :, :].real = arC_ft.real + arC_ft_rev.real
                psf_array_pad_fft[2 * j, :, :].imag = arC_ft.imag - arC_ft_rev.imag
                psf_array_pad_fft[2 * j + 1, :, :].real = arC_ft.imag + arC_ft_rev.imag
                psf_array_pad_fft[2 * j + 1, :, :].imag = arC_ft_rev.real - arC_ft.real
            del arC
            del arC_ft
            del arC_ft_rev
        if self.n_tot % 2 == 1:
            # take care of odd case at the end
            psf_array_pad = np.zeros((nfft, nfft))
            psf_array_pad[:ns2, :ns2] = self.psf_array[-1, :, :]
            psf_array_pad_fft[-1, :, :] = scipy.fft.fft2(psf_array_pad)
            del psf_array_pad

        if amp_penalty is not None:
            u = np.linspace(0, 1.0 - 1 / nfft, nfft)
            u = np.where(u > 0.5, u - 1, u)
            ut = np.sqrt(u[:, None] ** 2 + u[None, :] ** 2)
            for ip in range(self.n_tot):
                psf_array_pad_fft[ip, :, :] *= 1.0 + amp_penalty["amp"] * np.exp(
                    -2 * np.pi**2 * ut**2 * amp_penalty["sig"] ** 2
                )

        tm3 = time.perf_counter()

        # ... continue FFT
        for ipsf in range(self.n_tot):
            for jpsf in range(0, ipsf + 1, 2):
                if ipsf == jpsf:
                    ift = np.fft.ifftshift(
                        scipy.fft.ifft2(
                            psf_array_pad_fft[ipsf, :, :].real ** 2 + psf_array_pad_fft[ipsf, :, :].imag ** 2
                        )
                    )
                    self.overlaparray[ipsf, ipsf, :, :] = ift[
                        nfft // 2 - b : nfft // 2 + b + 1, nfft // 2 - b : nfft // 2 + b + 1
                    ].real
                else:
                    ift = np.fft.ifftshift(
                        scipy.fft.ifft2(
                            psf_array_pad_fft[ipsf, :, :]
                            * np.conjugate(
                                psf_array_pad_fft[jpsf, :, :] - 1j * psf_array_pad_fft[jpsf + 1, :, :]
                            )
                        )
                    )
                    self.overlaparray[ipsf, jpsf, :, :] = ift[
                        nfft // 2 - b : nfft // 2 + b + 1, nfft // 2 - b : nfft // 2 + b + 1
                    ].real
                    self.overlaparray[ipsf, jpsf + 1, :, :] = ift[
                        nfft // 2 - b : nfft // 2 + b + 1, nfft // 2 - b : nfft // 2 + b + 1
                    ].imag

        tm4 = time.perf_counter()

        # the 'above diagonal' can be obtained by flipping
        for ipsf in range(1, self.n_tot):
            for jpsf in range(ipsf):
                self.overlaparray[jpsf, ipsf, :, :] = self.overlaparray[ipsf, jpsf, ::-1, ::-1]

        tm5 = time.perf_counter()
        print(f"psf timing: {tm1-tm0:7.4f} {tm2-tm0:7.4f} {tm3-tm0:7.4f} {tm4-tm0:7.4f} {tm5-tm0:7.4f}")

        # extract C, the overlaps of the output PSFs
        self.C = np.diag(self.overlaparray[-self.n_out :, -self.n_out :, b, b])

    # ### <-- end __init__ here ###


def get_coadd_matrix(
    psfobj,
    psf_oversamp_factor,
    targetleak,
    ctrpos,
    distort_matrices,
    in_stamp_dscale,
    in_stamp_shape,
    out_stamp_dscale,
    out_stamp_shape,
    in_mask,
    tbdy_radius,
    smax=1.0,
    flat_penalty=0.0,
    use_kappa_arr=None,
    choose_outputs="CKMSTU",
):
    """
    Convert PSF overlap object to transfer matrices.

    Parameters
    ----------
    psfobj : furry_parakeet.pyimcom_interface.PSF_Overlap
        The PSF overlap class object.
    psf_oversamp_factor : float
        The PSF oversampling factor relative to native pixel scale.
    targetleak : np.array of float
        The target leakage U/C, length ``n_out`` (same as number of output PSFs in `psfobj`).
    ctrpos : list of (float, float)
        Length ``n_in`` list of postage stamp centroids in the stacking frame, in (x,y) ordering.
    distort_matrices : list of np.array
        List (length ``n_in``) of shape (2, 2) matrices. ``None`` can be used if there is no distortion.
    in_stamp_dscale : float
        Input postage stamp scale.
    in_stamp_shape : (int, int)
        Input postage stamp shape, length 2 tuple in the form (``ny_in``, ``nx_in``).
    out_stamp_dscale : float
        Output postage stamp scale.
    out_stamp_shape : (int, int)
        Output postage stamp shape, length 2 tuple in the form (``ny_out``, ``nx_out``).
    in_mask : np.array of bool
        Mask, True represents a good pixel. The shape is (``n_in``, ``ny_in``, ``nx_in``).
        Set to None to accept all pixels.
    tbdy_radius : float
        Radius of boundary to clip pixels in input images.
    smax : float, optional
        Maximum allowed value of noise factor Sigma (default = 1 if you want to avoid amplifying noise).
    flat_penalty : float, optional
        Amount by which to penalize having different contributions to the output from different
        input images (not recommended).
    use_kappa_arr : np.array of float, optional
        If supplied, array of kappa must be in ascending order (length ``nv`` in Cao et al. 2024) for
        Cholesky decompositions.
    choose_outputs : str, optional
        Which outputs to report (see notes for options and defualts), each option is coded by a character
        in the string.

    Returns
    -------
    dict
        Dictionary containing some or all of the keys 'A', 'mBhalf', 'C', 'kappa', 'fullmask',
        'Sigma', 'T', and 'UC' (see notes for description).

    Notes
    -----
    The option characters for `choose_outputs`, the corresponding keys in the output dictionaries,
    and desciptions are:

    *  ``A`` : ``'A'``, np.array, IMCOM system matrices (input-input overlap)

       Shape (n_all, n_all), where n_all is the total number of accepted pixels

    *  ``B`` : ``'mBhalf'``, np.array, IMCOM transfer matrices (input-output overlap), -0.5*B in
       Rowe et al. 2011.

       Shape (n_out, nx_out * ny_out, n_all)

    *  ``C`` : ``'C'``, np.array, IMCOM target normalizations (output-output overlap)

       Shape (n_out)

    *  ``K`` : ``'kappa'``, np.array, Lagrange multiplier

       Shape (n_out, ny_out, nx_out)

    *  ``M`` : ``'fullmask'``, np.array, coaddition input pixel mask

       Shape (n_in, ny_in, nx_in)

    *  ``S`` : ``'Sigma'``, np.array, IMCOM noise amplification map

       Shape (n_out, ny_out, nx_out)

    *  ``T`` = coaddition matrix, np.array, IMCOM transfer matrix

       Shape (n_out, ny_out, nx_out, n_in, ny_in, nx_in)

    *  ``U`` : ``'UC'``, np.array PSF leakage map (U_alpha/C)

       Shape (n_out, ny_out, nx_out)

    A and B are large and you probably only want them for test cases. The default choice is ``'CKMSTU'``.

    """

    # number of input and output images
    n_in = psfobj.n_in
    n_out = psfobj.n_out

    # get information
    (ny_in, nx_in) = in_stamp_shape
    (ny_out, nx_out) = out_stamp_shape

    tm0 = time.perf_counter()

    # positions of the pixels
    xpos = np.zeros((n_in, ny_in, nx_in))
    ypos = np.zeros((n_in, ny_in, nx_in))
    for i in range(n_in):
        xo = np.zeros((ny_in, nx_in))
        yo = np.zeros((ny_in, nx_in))
        xo[:, :] = np.linspace((1 - nx_in) / 2, (nx_in - 1) / 2, nx_in)[None, :]
        yo[:, :] = np.linspace((1 - ny_in) / 2, (ny_in - 1) / 2, ny_in)[:, None]
        xpos[i, :, :] = (
            in_stamp_dscale * (distort_matrices[i][0, 0] * xo + distort_matrices[i][0, 1] * yo) + ctrpos[i][0]
        )
        ypos[i, :, :] = (
            in_stamp_dscale * (distort_matrices[i][1, 0] * xo + distort_matrices[i][1, 1] * yo) + ctrpos[i][1]
        )

    tm1 = time.perf_counter()

    # mask information and table
    if in_mask is None:  # noqa: SIM108
        full_mask = np.full((n_in, ny_in, nx_in), True)
    else:
        full_mask = np.copy(in_mask)
    # now do the radius clipping
    Lx = out_stamp_dscale * (nx_out - 1) / 2.0
    Ly = out_stamp_dscale * (ny_out - 1) / 2.0
    full_mask = np.where(
        np.maximum(np.abs(xpos) - Lx, 0.0) ** 2 + np.maximum(np.abs(ypos) - Ly, 0.0) ** 2 < tbdy_radius**2,
        full_mask,
        False,
    )
    masklayers = []
    #
    # ... and we will make a list version of the mask.
    # ngood[i] pixels to be used from i th input image
    # ... starting from position nstart[i]
    #     (formally: nstart[n_in] == n_all)
    # n_all total
    #
    ngood = np.zeros((n_in,), np.int32)
    nstart = np.zeros((n_in + 1,), np.int32)
    for i in range(n_in):
        masklayers += [np.where(full_mask[i, :, :])]
        ngood[i] = len(masklayers[i][0])
        nstart[i + 1 :] += ngood[i]
    n_all = np.sum(ngood)

    # and build list of positions
    my_x = []
    my_y = []
    for i in range(n_in):
        my_x += [xpos[i, masklayers[i][0], masklayers[i][1]].flatten()]
        my_y += [ypos[i, masklayers[i][0], masklayers[i][1]].flatten()]

    tm2 = time.perf_counter()

    # build optimization matrices
    A = np.zeros((n_all, n_all))
    mBhalf = np.zeros((n_out, nx_out * ny_out, n_all))
    C = np.zeros((n_out,))
    #
    # the A-matrix first
    s = psfobj.dsample / psf_oversamp_factor * psfobj.s_in
    for i in range(n_in):
        for j in range(n_in):
            if j >= i:
                ddx = np.zeros((ngood[i], ngood[j]))
                ddy = np.zeros((ngood[i], ngood[j]))
                ddx[:, :] = my_x[i][:, None] - my_x[j][None, :]
                ddy[:, :] = my_y[i][:, None] - my_y[j][None, :]
                out1 = np.zeros((1, ngood[i] * ngood[j]))
                #
                # bounding box calculation, in interpolation (psfobj) pixels
                # my_xmax = (numpy.amax(my_x[i]) - numpy.amin(my_x[j])) / s
                # my_xmin = (numpy.amin(my_x[i]) - numpy.amax(my_x[j])) / s
                # my_ymax = (numpy.amax(my_y[i]) - numpy.amin(my_y[j])) / s
                # my_ymin = (numpy.amin(my_y[i]) - numpy.amax(my_y[j])) / s
                if j == i:
                    pyimcom_croutines.iD5512C_sym(
                        psfobj.overlaparray[i, j, :, :].reshape((1, psfobj.nsample, psfobj.nsample)),
                        ddx.flatten() / s + psfobj.nc,
                        ddy.flatten() / s + psfobj.nc,
                        out1,
                    )
                else:
                    pyimcom_croutines.iD5512C(
                        psfobj.overlaparray[i, j, :, :].reshape((1, psfobj.nsample, psfobj.nsample)),
                        ddx.flatten() / s + psfobj.nc,
                        ddy.flatten() / s + psfobj.nc,
                        out1,
                    )
                A[nstart[i] : nstart[i + 1], nstart[j] : nstart[j + 1]] = out1.reshape((ngood[i], ngood[j]))

            else:
                # we have already computed this component of A
                A[nstart[i] : nstart[i + 1], nstart[j] : nstart[j + 1]] = A[
                    nstart[j] : nstart[j + 1], nstart[i] : nstart[i + 1]
                ].T
    #
    # flat penalty
    if flat_penalty > 0.0:
        for i in range(n_in):
            for j in range(n_in):
                A[nstart[i] : nstart[i + 1], nstart[j] : nstart[j + 1]] -= flat_penalty / n_in / 2.0
                A[nstart[j] : nstart[j + 1], nstart[i] : nstart[i + 1]] -= flat_penalty / n_in / 2.0
            A[nstart[i] : nstart[i + 1], nstart[i] : nstart[i + 1]] += flat_penalty

    # force exact symmetry
    A = (A + A.T) / 2.0
    # end A matrix

    tm3 = time.perf_counter()

    # now the mBhalf matrix
    # first get output pixel positions
    # xout = numpy.zeros((ny_out,nx_out))
    # yout = numpy.zeros((ny_out,nx_out))
    # xout[:,:] = numpy.linspace((1-nx_out)/2, (nx_out-1)/2, nx_out)[None,:] * out_stamp_dscale
    # yout[:,:] = numpy.linspace((1-ny_out)/2, (ny_out-1)/2, ny_out)[:,None] * out_stamp_dscale
    # xout = xout.flatten()
    # yout = yout.flatten()
    # for i in range(n_in):
    #  for j in range(n_out):
    #    ddx = numpy.zeros((ngood[i],ny_out*nx_out))
    #    ddy = numpy.zeros((ngood[i],ny_out*nx_out))
    #    ddx[:,:] = my_x[i][:,None] - xout[None,:]
    #    ddy[:,:] = my_y[i][:,None] - yout[None,:]
    #    out1 = numpy.zeros((1,ngood[i]*ny_out*nx_out))
    #    pyimcom_croutines.iD5512C(psfobj.overlaparray[i,psfobj.n_in+j,:,:].reshape(
    #      (1,psfobj.nsample,psfobj.nsample)),
    #      ddx.flatten()/s+psfobj.nc, ddy.flatten()/s+psfobj.nc, out1)
    #    mBhalf[j,:,nstart[i]:nstart[i+1]] = out1.reshape((ngood[i],ny_out*nx_out)).T

    for i in range(n_in):
        ddx = np.zeros((ngood[i], nx_out))
        ddx[:, :] = (
            my_x[i][:, None]
            - np.linspace((1 - nx_out) / 2, (nx_out - 1) / 2, nx_out)[None, :] * out_stamp_dscale
        )
        ddy = np.zeros((ngood[i], ny_out))
        ddy[:, :] = (
            my_y[i][:, None]
            - np.linspace((1 - ny_out) / 2, (ny_out - 1) / 2, ny_out)[None, :] * out_stamp_dscale
        )
        out1 = np.zeros((ngood[i], ny_out * nx_out))
        for j in range(n_out):
            pyimcom_croutines.gridD5512C(
                psfobj.overlaparray[i, psfobj.n_in + j, :, :], ddx / s + psfobj.nc, ddy / s + psfobj.nc, out1
            )
            mBhalf[j, :, nstart[i] : nstart[i + 1]] = out1.T

    tm4 = time.perf_counter()

    # and C
    C = np.zeros((n_out,))
    for j in range(n_out):
        C[j] = psfobj.overlaparray[n_in + j, n_in + j, psfobj.nc, psfobj.nc]

    # generate matrices
    tm5 = time.perf_counter()
    if use_kappa_arr is None:
        (kappa_, Sigma_, UC_, T_) = pyimcom_lakernel.CKernelMulti(
            A, mBhalf, C, np.array(targetleak), smax=smax
        )
    else:
        (kappa_, Sigma_, UC_, T_) = pyimcom_lakernel.get_coadd_matrix_discrete(
            A, mBhalf, C, use_kappa_arr, np.array(targetleak), smax=smax
        )
    tm6 = time.perf_counter()

    # this code was just for testing, only works for n_out = 1
    # (kappa_, Sigma_, UC_, T_) = pyimcom_lakernel.BruteForceKernel(A,mBhalf[0,:,:],C[0],targetleak[0])
    # T_ = T_.reshape(1,ny_out*nx_out,nstart[n_in])

    # start building the output dictionary
    returndata = {}
    if "A" in choose_outputs.upper():
        returndata["A"] = A
    else:
        del A
    if "B" in choose_outputs.upper():
        returndata["mBhalf"] = mBhalf
    else:
        del mBhalf
    if "C" in choose_outputs.upper():
        returndata["C"] = C
    if "M" in choose_outputs.upper():
        returndata["full_mask"] = full_mask

    # post processing
    kappa = kappa_.reshape((n_out, ny_out, nx_out))
    del kappa_
    Sigma = Sigma_.reshape((n_out, ny_out, nx_out))
    del Sigma_
    UC = UC_.reshape((n_out, ny_out, nx_out))
    del UC_
    # ... and save
    if "K" in choose_outputs.upper():
        returndata["kappa"] = kappa
    if "S" in choose_outputs.upper():
        returndata["Sigma"] = Sigma
    if "U" in choose_outputs.upper():
        returndata["UC"] = UC

    # for T: (nt,m,n) --> (n_out,ny_out,nx_out,n_in,ny_in,nx_in)
    T = np.zeros((n_out, ny_out, nx_out, n_in, ny_in, nx_in), dtype=np.float32)
    for i in range(n_in):
        for k in range(ngood[i]):
            T[:, :, :, i, masklayers[i][0][k], masklayers[i][1][k]] = T_[:, :, nstart[i] + k].reshape(
                (n_out, ny_out, nx_out)
            )
    if "T" in choose_outputs.upper():
        returndata["T"] = T

    tm7 = time.perf_counter()
    print(
        "times: {:7.4f} {:7.4f} {:7.4f} {:7.4f} {:7.4f} {:7.4f} {:7.4f}".format(
            tm1 - tm0, tm2 - tm0, tm3 - tm0, tm4 - tm0, tm5 - tm0, tm6 - tm0, tm7 - tm0
        )
    )

    return returndata


def test_psf_inject(
    psf_in_list,
    psf_out_list,
    psf_oversamp_factor,
    ctrpos,
    distort_matrices,
    T,
    in_mask,
    in_stamp_dscale,
    out_stamp_dscale,
    srcpos,
):
    """
    Create input postage stamps of point sources of unit flux and coadd them to test the PSF matrices.

    Parameters
    ----------
    psf_in_list : list of np.array
        List of input PSFs (length ``n_in``, each is a 2D array).
    psf_out_list : list of np.array
        List of output PSFs (length ``n_out``, each is a 2D array).
    psf_oversamp_factor : float
        PSF oversampling factor relative to native pixel scale (usually >1).
    ctrpos : list of (float, float)
        List (length n_in) of postage stamp centroids in stacking frame, (x,y) ordering.
    distort_matrices : list of np.array
        List (length n_in) of shape (2, 2) matrices.
    T : np.array
        Coaddition matrix, shape = (n_out, ny_out, nx_out, n_in, ny_in, nx_in).
    in_mask : np.array of bool
        Mask, True represents a good pixel. The shape is (``n_in``, ``ny_in``, ``nx_in``).
        Set to None to accept all pixels.
    in_stamp_dscale : float
        Input postage stamp scale.
    out_stamp_dscale : float
        Output postage stamp scale.
    srcpos : (float, float)
        Position of the point source to inject, (x,y) ordering.

    Returns
    -------
    in_array : np.array
        Input postage stamps, shape = (n_in, ny_in, nx_in)
    out_array : np.array
        output postage stamp, shape = (n_out, ny_out, nx_out)
    diff_array : np.array
        output postage stamp error, shape = (n_out, ny_out, nx_out)

    """

    # basic info
    (n_out, ny_out, nx_out, n_in, ny_in, nx_in) = np.shape(T)

    # center pixel of input stamps
    xctr = (nx_in - 1) / 2.0
    yctr = (ny_in - 1) / 2.0

    # make input stamp array
    in_array = np.zeros((n_in, ny_in, nx_in))

    p = 5  # pad length

    # make the input stamps
    for ipsf in range(n_in):
        (ny, nx) = np.shape(psf_in_list[ipsf])

        # get position of source in stamp coordinates
        xpsf = srcpos[0] - ctrpos[ipsf][0]
        ypsf = srcpos[1] - ctrpos[ipsf][1]
        M = np.linalg.inv(in_stamp_dscale * distort_matrices[ipsf])
        xpos = M[0, 0] * xpsf + M[0, 1] * ypsf + xctr
        ypos = M[1, 0] * xpsf + M[1, 1] * ypsf + yctr

        # now pixel positions relative to the PSF
        inX = np.zeros((ny_in, nx_in))
        inY = np.zeros((ny_in, nx_in))
        inX[:, :] = np.linspace(-xpos, nx_in - 1 - xpos, nx_in)[None, :]
        inY[:, :] = np.linspace(-ypos, ny_in - 1 - ypos, ny_in)[:, None]
        interp_array = np.zeros((1, ny_in * nx_in))
        pyimcom_croutines.iD5512C(
            np.pad(psf_in_list[ipsf], p).reshape((1, ny + 2 * p, nx + 2 * p)),
            psf_oversamp_factor * inX.flatten() + (nx - 1) / 2.0 + p,
            psf_oversamp_factor * inY.flatten() + (ny - 1) / 2.0 + p,
            interp_array,
        )
        in_array[ipsf, :, :] = interp_array.reshape((ny_in, nx_in)) * psf_oversamp_factor**2

    if in_mask is not None:
        in_array = np.where(in_mask, in_array, 0.0)

    # --- end construction of the input postage stamps ---

    out_array = (T.reshape(n_out * ny_out * nx_out, n_in * ny_in * nx_in) @ in_array.flatten()).reshape(
        n_out, ny_out, nx_out
    )

    # --- and now the 'target' output array ---
    target_out_array = np.zeros((n_out, ny_out, nx_out))
    xctr = (nx_out - 1) / 2.0
    yctr = (ny_out - 1) / 2.0
    for ipsf in range(n_out):
        (ny, nx) = np.shape(psf_out_list[ipsf])

        # get position of source in stamp coordinates
        xpos = srcpos[0] / out_stamp_dscale + xctr
        ypos = srcpos[1] / out_stamp_dscale + yctr

        # now pixel positions relative to the PSF
        inX = np.zeros((ny_out, nx_out))
        inY = np.zeros((ny_out, nx_out))
        inX[:, :] = (
            np.linspace(-xpos, nx_out - 1 - xpos, nx_out)[None, :] * out_stamp_dscale / in_stamp_dscale
        )
        inY[:, :] = (
            np.linspace(-ypos, ny_out - 1 - ypos, ny_out)[:, None] * out_stamp_dscale / in_stamp_dscale
        )
        interp_array = np.zeros((1, ny_out * nx_out))
        pyimcom_croutines.iD5512C(
            np.pad(psf_out_list[ipsf], p).reshape((1, ny + 2 * p, nx + 2 * p)),
            psf_oversamp_factor * inX.flatten() + (nx - 1) / 2.0 + p,
            psf_oversamp_factor * inY.flatten() + (ny - 1) / 2.0 + p,
            interp_array,
        )
        target_out_array[ipsf, :, :] = interp_array.reshape((ny_out, nx_out)) * psf_oversamp_factor**2

    diff_array = out_array - target_out_array
    return (in_array, out_array, diff_array)
