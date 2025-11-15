C routines
##########

This document describes the functions in ``furry_parakeet.pyimcom_croutines``.

Linear algebra kernel
=====================

**lakernel1** (lam, Q, mPhalf, C, targetleak, kCmin, kCmax, nbias, smax, kappa, Sigma, UC, T)

  **Parameters** :

  -  lam : np.ndarray
       system matrix eigenvalues, shape=(n,)
  -  Q : np.ndarray
       system matrix eigenvectors, shape=(n,n)
  -  mPhalf : np.ndarray
       premultiplied target overlap matrix (-P/2), shape=(m,n)
  -  C : float
       target normalization
  -  targetleak : float
       allowable leakage of target PSF
  -  kCmin, kCmax, nbis : float
       range of kappa/C to test, number of bisections
  -  smax : float
       maximum allowed noise amplification Sigma
  -  kappa : np.ndarray
       (*Writes to this array.*) Lagrange multiplier per output pixel, shape=(m,)
  -  Sigma : np.ndarray
       (*Writes to this array.*) output noise amplification, shape=(m,)
  -  UC : np.ndarray
       (*Writes to this array.*) fractional squared error in PSF, shape=(m,)
  -  T : np.ndarray
       (*Writes to this array.*) coaddition matrix, shape=(m,n)
        
       **Warning**: This needs to be multiplied by Q^T after the return.

  Returns:
    None


10x10 and 8x8 interpolation routines
====================================

These are 2D interpolation routines. The "D5512" routines are based on the nearest 10x10 points, and have the highest accuracy. The "G4460" routines are based on the nearest 8x8 points, and are sufficient in some cases.
        
**iD5512C** (infunc, xpos, ypos, fhatout)

**iG4460C** (infunc, xpos, ypos, fhatout)

  **Parameters** :

  -   infunc : np.ndarray
        input function on some grid, shape =(nlayer,ngy,ngx)
  -   xpos : np.ndarray
        input x values, shape=(nout,)
  -   ypos : np.ndarray
        input y values, shape=(nout,)
  -   fhatout : np.ndarray
        (*Writes to this array.*) location to put the output values, shape=(nlayer,nout)

  Returns:
    None
        
The following routines assume that the output is symmetric as a matrix of size sqrt(nout) x sqrt(nout). They are particularly useful for interpolating even functions of a separation r_i-r_j and arranging the results into a matrix form (A_ij).

**iD5512C_sym** (infunc, xpos, ypos, fhatout)

**iG4460C_sym** (infunc, xpos, ypos, fhatout)

  **Parameters** :

  -   infunc : np.ndarray
        input function on some grid, shape =(nlayer,ngy,ngx)
  -   xpos : np.ndarray
        input x values, shape=(nout,)
  -   ypos : np.ndarray
        input y values, shape=(nout,)
  -   fhatout : np.ndarray
        (*Writes to this array.*) location to put the output values, shape=(nlayer,nout)

  Returns:
    None
        
The following routines interpolate onto a rectilinear grid.

**gridD5512C** (infunc, xpos, ypos, fhatout)

**gridG4460C** (infunc, xpos, ypos, fhatout)

  **Parameters** :

  -   infunc : np.ndarray
        input function on some grid, shape =(ngy,ngx)
  -   xpos : np.ndarray
        input x values, shape=(npi,nxo)
  -   ypos : np.ndarray
        input y values, shape=(npi,nyo)
  -   fhatout : np.ndarray
        (*Writes to this array.*) location to put the output values, shape=(npi,nyo*nxo)

  Returns:
    None
        
Destriping interpolation routines
=================================

These are much faster routines, but don't interpolate varying functions as accurately. They are intended to be used in the destriping module, where the "structure" present is mostly sky + noise.

The "forward" bilinear interpolation is:
         
**bilinear_interpolation** (image, g_eff, rows, cols, coords, num_coords, interpolated_image)

  **Parameters** :

  -   image : np.ndarray
         the input image data (image to-be-interpolated; image "B"), shape=(rows,cols)
  -   g_eff : np.ndarray
         the input image pixel response matrix (image "B" g_eff), shape=(rows,cols)
  -   rows, cols : int
         dimensions of the images
  -   coords : np.ndarray
         the array of coordinates (x,y) to be interpolated onto (image "A" coords); ``coords[0,:]`` is the y-coordinate and ``coords[1,:]`` is the x-coordinate (last axis is flattened), shape=(2,num_coords)
  -   num_coords : int
         number of provided coordinate pairs
  -   interpolated_image : np.ndarray
         (*Writes to this array.*) output array for interpolated values. I_B interpolated onto I_A grid, shape=(rows,cols)

  Returns:
    None

The "transpose" bilinear interpolation (2D cloud-in-cell) is:

**bilinear_transpose** (image, rows, cols, coords, num_coords, original_image)

  **Parameters** :

  -   image : np.ndarray
         the gradient image data (gradient image to-be-transpose-interpolated; image "gradient_interpolated"), shape=(rows,cols)
  -   g_eff : np.ndarray
         the input image pixel response matrix (image "B" g_eff), shape=(rows,cols)
  -   rows, cols : int
         dimensions of the images
  -   coords : np.ndarray
         the array of coordinates (x,y) to be interpolated onto (image "A" coords); ``coords[0,:]`` is the y-coordinate and ``coords[1,:]`` is the x-coordinate (last axis is flattened), shape=(2,num_coords)
  -   num_coords : int
         number of provided coordinate pairs
  -   original_image : np.ndarray
         (*Writes to this array.*) the output of transpose interpolation (gradient image interpolated onto image "B" grid), shape=(rows,cols)

  Returns:
    None

          
Routines for kappa interpolation
================================

These are linear algebra utilities involved in the Cholesky method with multiple kappa nodes.

**build_reduced_T_wrap** (Nflat, Dflat, Eflat, kappa, ucmin, smax, out_kappa, out_Sigma, out_UC, out_w)

  **Parameters** :
          
  -   Nflat : np.ndarray
          input noise array, shape=(m,nv,nv).flatten=(m*nv*nv,)
  -   Dflat : np.ndarray
          input 1st order signal D/C, shape=(m,nv).flatten=(m*nv,)
  -   Eflat : np.ndarray
          input 2nd order signal E/C, shape=(m,nv,nv).flatten=(m*nv*nv,)
  -   kappa : np.ndarray
          list of eigenvalues, must be sorted ascending, shape=(nv,)
  -   ucmin : float
          min U/C
  -   smax : float
          max Sigma (noise)
  -   out_kappa : np.ndarray
          (*Writes to this array.*) output "kappa" parameter, shape=(m,)
  -   out_Sigma : np.ndarray
          (*Writes to this array.*) output "Sigma", shape=(m,)
  -   out_UC : np.ndarray
          (*Writes to this array.*) output "U/C", shape=(m,)
  -   out_w : np.ndarray
          (*Writes to this array.*) output weights for each eigenvalue and each output pixel, shape=(m,nv).flatten=(m*nv,)
