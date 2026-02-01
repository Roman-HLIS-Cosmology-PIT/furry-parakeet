import re
import time

import numpy as np
from furry_parakeet import pyimcom_interface
from numpy.random import RandomState


def test_evensize():
    """Test exception handling for even sizes in PSF_Overlap."""

    try:
        ovl = pyimcom_interface.PSF_Overlap([], [], 1.0 / 6.0, 64, 0.11)
        assert ovl.nsample < 0  # will fail, the above should raise an exception
    except Exception as e:
        print(str(e))
        assert str(e) == "error in PSF_Overlap, nsample=64 must be odd"


def test_ikernel():
    """General test for furry-parakeet."""

    output = ""

    # Config file for this test
    content = [
        "# configuration file for the test script (you can copy/modify to experiment with it)",
        "# wavelength [um]",
        "LAMBDA: 1.29",
        "# number of dithers",
        "N: 6",
        "# input and output postage stamp sizes",
        "INSIZE: 64 64",
        "OUTSIZE: 50 50",
        "# target sqrt{U/C} (1e-3 is a reasonable choice)",
        "UCTARGET: 1e-3",
        "# how much Gaussian smoothing (in input pixels) to put in the output image",
        "EXTRASMOOTH: 0.63",
        "# bad pixel fraction (this script randomly kills pixels)",
        "BADFRAC: .025",
        "# Optional rolls (in degrees for each exposure) and camera distortions",
        "# (shear is g1,g2 for each exposure; magnify is for each exposure)",
        "ROLL: 45 45 45 0 195 0",
        "SHEAR: 0 0 0 0 0 0 0 0 0 0 -.01 .01",
        "MAGNIFY: .02 -.02 0 0 0 0",
        "# turn this on if you want horrific messy non-identical input PSF",
        "MESSYPSF",
        "# output file prefix",
        "OUT: ~/this_test_",
        "# random number generator seed",
        "RNGSEED: 3000",
    ]

    # basic parameters
    n_out = 1  # number of output images
    nps = 8  # PSF oversampling factor
    s_in = 0.11  # input pixel scale in arcsec
    s_out = 0.025  # output pixel scale in arcsec
    cd = 0.3  # charge diffusion, rms per axis in pixels
    n1 = 512  # PSF postage stamp size
    extbdy = 1.0  # boundary extension in arcsec
    uctarget = 2e-4  # target image leakage (in RMS units)
    flat_penalty = 1e-8  # penalty for having different dependences on different inputs

    # default parameters
    seed = 1000  # default seed
    sigout = np.sqrt(1.0 / 12.0 + cd**2)  # output smoothing
    roll = None  # no rolls
    shear = None  # no camera shear
    magnify = None  # no camera magnification
    badfrac = 0.0  # fraction of pixels to randomly kill
    messyPSF = False  # put messy asymmetries in the PSF

    for line in content:
        # -- REQUIRED KEYWORDS --

        m = re.search(r"^LAMBDA:\s*(\S+)", line)
        if m:
            lam = float(m.group(1))

        m = re.search(r"^N:\s*(\d+)", line)
        if m:
            n_in = int(m.group(1))

        m = re.search(r"^OUT:\s*(\S+)", line)
        if m:
            outstem = m.group(1)

        m = re.search(r"^OUTSIZE:\s*(\d+)\s+(\d+)", line)  # ny, then nx
        if m:
            ny_out = int(m.group(1))
            nx_out = int(m.group(2))

        m = re.search(r"^INSIZE:\s*(\d+)\s+(\d+)", line)  # ny, then nx
        if m:
            ny_in = int(m.group(1))
            nx_in = int(m.group(2))

        # -- OPTIONS --
        m = re.search(r"^RNGSEED:\s*(\d+)", line)  # RNG seed
        if m:
            seed = int(m.group(1))

        m = re.search(r"^EXTRASMOOTH:\s*(\S+)", line)  # extra smoothing, pix rms per axis
        if m:
            sigout = np.sqrt(sigout**2 + float(m.group(1)) ** 2)

        m = re.search(r"^UCTARGET:\s*(\S+)", line)
        if m:
            uctarget = float(m.group(1))

        m = re.search(r"^BADFRAC:\s*(\S+)", line)
        if m:
            badfrac = float(m.group(1))

        m = re.search(r"^ROLL:", line)  # input is in degrees
        if m:
            roll = line.split()[1:]
            for j in range(len(roll)):
                roll[j] = float(roll[j]) * np.pi / 180.0  # convert to float

        m = re.search(r"^SHEAR:", line)  # shears g1,g2 for each exposure
        if m:
            shear = line.split()[1:]
            for j in range(len(shear)):
                shear[j] = float(shear[j])

        m = re.search(
            r"^MAGNIFY:", line
        )  # plate scale divided by 1+magnify[], so >0 --> smaller plate scale --> magnified image
        if m:
            magnify = line.split()[1:]
            for j in range(len(magnify)):
                magnify[j] = float(magnify[j])

        m = re.search(r"^MESSYPSF", line)
        if m:
            messyPSF = True

    rng = RandomState(seed)  # legacy random number generator so it doesn't update and cause a test to fail

    output += f"lambda = {lam} micron\n"
    output += f"n_in = {n_in}\n"
    output += f"output --> {outstem}\n"

    ld = lam / 2.37e6 * 206265.0 / s_in

    ImIn = []
    mlist = []
    posoffset = []
    if roll is None:
        roll = np.zeros((n_in,)).tolist()
    if shear is None:
        shear = np.zeros((2 * n_in,)).tolist()
    if magnify is None:
        magnify = np.zeros((n_in,)).tolist()
    for k in range(n_in):
        fk = 0
        if messyPSF:
            fk = k
        ImIn += [pyimcom_interface.psf_cplx_airy(n1, nps * ld, tophat_conv=nps, sigma=nps * cd, features=fk)]
        mlist += [
            pyimcom_interface.rotmatrix(roll[k])
            @ pyimcom_interface.shearmatrix(shear[2 * k], shear[2 * k + 1])
            / (1.0 + magnify[k])
        ]

        # positional offsets
        f = np.zeros((2,))
        f[0] = rng.random()
        f[1] = rng.random()
        Mf = s_in * mlist[k] @ f
        posoffset += [(Mf[0], Mf[1])]
    ImOut = [pyimcom_interface.psf_simple_airy(n1, nps * ld, tophat_conv=0.0, sigma=nps * sigout)]

    output += f"translation: {posoffset}\n"
    output += f"roll: {np.array(roll) * 180.0 / np.pi}\n"
    output += f"magnify: {np.array(magnify)}\n"
    output += f"shear g1: {np.array(shear)[::2]}\n"
    output += f"shear g2: {np.array(shear)[1::2]}\n"

    t1a = time.perf_counter()
    P = pyimcom_interface.PSF_Overlap(ImIn, ImOut, 0.5, 2 * n1 - 1, s_in, distort_matrices=mlist)
    P2 = pyimcom_interface.PSF_Overlap(
        ImIn, ImOut, 0.5, 2 * n1 - 1, s_in, distort_matrices=mlist, amp_penalty={"amp": 0.5, "sig": 0.5}
    )
    t1b = time.perf_counter()
    output += f"timing psf overlap: {t1b - t1a}\n"

    assert np.shape(P.psf_array) == (7, 1033, 1033)

    err = P.psf_array[:, 520, 540] - np.array(
        [0.00025673, 0.00046702, 0.00032949, 0.00020641, 0.00038896, 0.00020381, 0.00064662]
    )

    maxerr = np.amax(np.abs(err))
    assert maxerr < 1e-6

    # mask
    inmask = None
    if badfrac > 0:
        inmask = np.where(rng.random(size=(n_in, ny_in, nx_in)) > badfrac, True, False)
        output += f"number good {np.count_nonzero(inmask)} of {n_in * ny_in * nx_in}\n"
        output += f"{np.shape(inmask)}\n"
    else:
        # shouldn't get here
        assert False  # noqa: B011

    assert np.count_nonzero(inmask) == 23946
    assert np.shape(inmask) == (6, 64, 64)

    # and now make a coadd matrix
    t1c = time.perf_counter()
    ims = pyimcom_interface.get_coadd_matrix(
        P,
        float(nps),
        [uctarget**2],
        posoffset,
        mlist,
        s_in,
        (ny_in, nx_in),
        s_out,
        (ny_out, nx_out),
        inmask,
        extbdy,
        smax=1.0 / n_in,
        flat_penalty=flat_penalty,
        choose_outputs="ABCKMSTU",
    )
    ims_alt = pyimcom_interface.get_coadd_matrix(
        P,
        float(nps),
        [uctarget**2],
        posoffset,
        mlist,
        s_in,
        (ny_in, nx_in),
        s_out,
        (ny_out, nx_out),
        None,
        extbdy,
        smax=1.0 / n_in,
        flat_penalty=flat_penalty,
        choose_outputs="T",
    )
    print(np.shape(ims_alt["T"]))
    T_err = np.sum(ims_alt["T"], axis=(3, 4, 5)) - 1.0
    print(np.percentile(T_err, 25), np.percentile(T_err, 75))
    print(np.amax(np.abs(T_err)))
    assert np.amax(np.abs(T_err)) < 0.04
    t1d = time.perf_counter()
    output += f"timing coadd matrix: {t1d - t1c}\n"

    # another alternative, with amp_penalty (hence using P2)
    ims_alt2 = pyimcom_interface.get_coadd_matrix(
        P2,
        float(nps),
        [uctarget**2],
        posoffset,
        mlist,
        s_in,
        (ny_in, nx_in),
        s_out,
        (ny_out, nx_out),
        None,
        extbdy,
        smax=1.0 / n_in,
        flat_penalty=flat_penalty,
        choose_outputs="T",
    )
    diff_in_t = np.abs(ims_alt["T"] - ims_alt2["T"])
    print(np.median(diff_in_t), np.amax(diff_in_t))
    assert np.amax(diff_in_t) < 0.001

    output += f"number of output x input pixels used: {np.shape(ims["mBhalf"])}\n"
    output += f"C = {ims["C"]}\n"
    output += f"shape of T: {np.shape(ims["T"])}\n"

    assert np.abs(ims["C"][0] - 0.00517163) < 1e-6
    assert np.shape(ims["T"]) == (1, 50, 50, 6, 64, 64)

    err = ims["T"][0, 24, 30, 2, 29, 30:35] - np.array(
        [-7.6104574e-05, 0.0000000e00, 4.5013532e-02, 7.2479500e-03, -2.4549062e-03]
    )
    assert np.amax(np.abs(err)) < 1e-6

    # these outputs are turned off for the test
    # hdu = fits.PrimaryHDU(numpy.where(ims["full_mask"], 1, 0).astype(numpy.uint16))
    # hdu.writeto(outstem + "mask.fits", overwrite=True)
    # hdu = fits.PrimaryHDU(ims["A"])
    # hdu.writeto(outstem + "A.fits", overwrite=True)
    # hdu = fits.PrimaryHDU(ims["mBhalf"])
    # hdu.writeto(outstem + "mBhalf.fits", overwrite=True)
    # hdu = fits.PrimaryHDU(ims["C"])
    # hdu.writeto(outstem + "C.fits", overwrite=True)
    # hdu = fits.PrimaryHDU(ims["kappa"])
    # hdu.writeto(outstem + "kappa.fits", overwrite=True)
    # hdu = fits.PrimaryHDU(numpy.sqrt(ims["Sigma"]))
    # hdu.writeto(outstem + "sqSigma.fits", overwrite=True)
    # hdu = fits.PrimaryHDU(numpy.sqrt(ims["UC"]))
    # hdu.writeto(outstem + "sqUC.fits", overwrite=True)
    # hdu = fits.PrimaryHDU(
    #     ims["T"].reshape(
    #         (
    #             n_out,
    #             ny_out * nx_out,
    #             n_in * ny_in * nx_in,
    #         )
    #    )
    # )
    # hdu.writeto(outstem + "T.fits", overwrite=True)
    # hdu = fits.PrimaryHDU(
    #     numpy.transpose(numpy.sum(ims["T"], axis=(4, 5)), axes=(1, 0, 3, 2)).reshape(
    #         (ny_out, nx_out * n_in * n_out)
    #     )
    # )
    # hdu.writeto(outstem + "Tsum.fits", overwrite=True)
    # hdu = fits.PrimaryHDU(numpy.where(ims["full_mask"], 1, 0).astype(numpy.uint16))
    # hdu.writeto(outstem + "mask.fits", overwrite=True)

    # make test image
    # put test source in lower-right quadrant as displayed in ds9
    #
    # [   |   ]
    # [   |   ]
    # [   |   ]   (60% of the way from left to right,
    # [---+---]    20% of the way from bottom to top)
    # [   |   ]
    # [   |X  ]
    # [   |   ]
    #
    test_srcpos = (0.1 * (nx_out - 1) * s_out, -0.3 * (ny_out - 1) * s_out)
    (intest, outtest, outerr) = pyimcom_interface.test_psf_inject(
        ImIn, ImOut, nps, posoffset, mlist, ims["T"], inmask, s_in, s_out, test_srcpos
    )
    output += f"input image sums = {np.sum(intest, axis=(1, 2))}\n"

    # hdu = fits.PrimaryHDU(intest)
    # hdu.writeto(outstem + "sample_ptsrc_in.fits", overwrite=True)
    # hdu = fits.PrimaryHDU(numpy.transpose(intest, axes=(1, 0, 2)).reshape(ny_in, n_in * nx_in))
    # hdu.writeto(outstem + "sample_ptsrc_in_flat.fits", overwrite=True)
    # hdu = fits.PrimaryHDU(outtest)
    # hdu.writeto(outstem + "sample_ptsrc_out.fits", overwrite=True)
    # hdu = fits.PrimaryHDU(outerr)
    # hdu.writeto(outstem + "sample_ptsrc_out_err.fits", overwrite=True)

    output += f"in,out,err = {np.amax(intest)}, {np.amax(outtest)}, {np.amax(outerr)}\n"
    assert np.abs(np.amax(outtest) - 0.1843259689448174) < 5e-5
    assert np.amax(outerr) < 1e-5

    amp = np.zeros((n_out,))
    for ipsf in range(n_out):
        print(
            f"error {ipsf:2d} " f"{np.sqrt(np.sum(outerr[ipsf,:,:]**2)/np.sum(outtest[ipsf,:,:]**2)):11.5E}"
        )
        amp[ipsf] = np.sqrt(np.sum(outtest[ipsf, :, :] ** 2))

    output += f"{amp}\n"
    assert np.abs(amp[0] - 1.26544701) < 1e-4
