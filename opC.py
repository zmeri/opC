# -*- coding: utf-8 -*-
"""
This code calculates the k and n spectra from an ATR-FTIR input spectrum.

This code is based on Python 2.7.6, Numpy 1.10.4 and SciPy 0.17.0.

Author: Zach Baird

References:
* J. E. Bertie, S. L. Zhang, and C. D. Keefe, “Measurement and use of absolute 
infrared absorption intensities of neat liquids,” Vibrational Spectroscopy, 
vol. 8, pp. 215–229, 1995.
* J. E. Bertie and Z. Lan, “An accurate modified Kramers–Kronig transformation 
from reflectance to phase shift on attenuated total reflection,” The Journal 
of Chemical Physics, vol. 105, no. 19, p. 8502, 1996.
* J. E. Bertie, “John Bertie’s Download Site.” [Online]. Available: 
http://www.ualberta.ca/~jbertie/JBDownload.HTM. [Accessed: 21-Jun-2016].
"""
import numpy as np
import math
from scipy.optimize import differential_evolution
from scipy import interpolate

def KKtransTH(v, r, vve=None, rre=None):    
    """ This function performs the Kramer-Kronig transformation to convert the 
    reflection spectrum to the phase shift spectrum (theta).
    v: an array containing the wavenumbers of the spectrum
    r: an array containing the reflection values of the spectrum
    vve: an array containing the wavenumbers of the extension that is added to 
    the spectrum
    rre: an array containing the reflection values of the extension that is added
    to the spectrum (if None, then the Rs spectrum is linearly extrapolated down
    to 0 at a wavenumber of 0)
    """
    if rre is None:    
        dnu = v[1] - v[0]
        npts = math.floor(np.amin(v)/dnu)
        vve = np.amin(v) - (npts-np.arange(npts))*dnu
        rre = r[0]/np.amin(v)*vve
        vve = np.expand_dims(vve, axis=1)
        rre = np.expand_dims(rre, axis=1)

    num_ext = vve.shape[0]
    vv = np.vstack((vve, v))
    rr = np.vstack((rre, r))
    theta = np.zeros_like(v)
    dvv = np.zeros_like(vv)
    dif = np.diff(vv[1::2, 0], axis=0)
    dvv[1::2, 0] = np.append(dif, [dif[-1]], axis=0)
    dif = np.diff(vv[::2, 0], axis=0)
    dvv[::2, 0] = np.append(dif, [dif[-1]], axis=0)
    vv2 = pow(vv,2)
    numer = vv * 0.5 * np.log(abs(rr)) * dvv
    
    v_size = v.shape[0]
    
    for h in range(v_size):
        if (h+num_ext)%2 == 0:
            jnk1 = numer[1::2, :] / (vv2[1::2, :] - vv2[h+num_ext,0])
            theta[h,0] = np.sum(jnk1)
        else:
            jnk1 = numer[::2, :] / (vv2[::2, :] - vv2[h+num_ext,0])
            theta[h,0] = np.sum(jnk1)
        
    theta = theta * -2.0 / math.pi
    
    return theta
    
def KKtrans(v, k, vve=None, kke=None, n_S=1.5):
    """ This function performs the Kramer-Kronig transformation to convert the 
    imaginary spectrum (k) to the real spectrum (n).
    v: an array containing the wavenumbers of the spectrum
    k: an array containing the k values of the spectrum
    vve: an array containing the wavenumbers of the extension that is added to 
    the spectrum
    kke: an array containing the k values of the extension that is added
    to the spectrum (if None, then the k spectrum is linearly extrapolated down
    to 0 at a wavenumber of 0)
    n_S: the refractive index of the sample at a high wavenumber (default=1.5)
    """
    if kke is None:    
        dnu = v[1] - v[0]
        npts = math.floor(np.amin(v)/dnu)
        vve = np.amin(v) - (npts-np.arange(npts))*dnu
        kke = k[0]/np.amin(v)*vve
        vve = np.expand_dims(vve, axis=1)
        kke = np.expand_dims(kke, axis=1)
    
    num_ext = vve.shape[0] 
    vv = np.vstack((vve, v))
    kk = np.vstack((kke, k))
    nn = np.zeros_like(v)
    dvv = np.zeros_like(vv)
    dif = np.diff(vv[1::2, 0], axis=0)
    dvv[1::2, 0] = np.append(dif, [dif[-1]], axis=0)
    dif = np.diff(vv[::2, 0], axis=0)
    dvv[::2, 0] = np.append(dif, [dif[-1]], axis=0)
    vv2 = pow(vv,2)
    kk[np.where(kk < 0)] = 0.0
    numer = vv * kk * dvv
    
    v_size = v.shape[0]
    
    for h in range(v_size):
        if (h+num_ext)%2 == 0:
            jnk1 = numer[1::2, 0] / (vv2[1::2, 0] - vv2[h+num_ext,0])
            nn[h,0] = np.sum(jnk1)
        else:
            jnk1 = numer[::2, 0] / (vv2[::2, 0] - vv2[h+num_ext,0])
            nn[h,0] = np.sum(jnk1)
    
    nn = nn * 2.0 / math.pi
    nn = nn + n_S
    
    return nn
    
def calib(params, nu, a, Rs, ex, nRef=1):
    # A function used for solving for the effective number of reflections.
    aa = np.copy(a)
    eR = params[0]
    Ab_std = -np.log10((pow(Rs,nRef*eR)+pow(Rs,2*nRef*eR))/2.0)
    aa = np.multiply(ex, aa) + np.multiply((1.-ex), Ab_std)
    return np.sum(100*(Ab_std - aa)**2)

def calibrate(nu_Z, n_Z, nu, Ab, nu_std, k_std, n_std, n_S=1.5, nRef=1):
    """ calibrate is used to determine the effective number of reflections of 
    a spectrometer.
    nu_Z: an array containing the wavenumbers of the refractive index data for the ATR crystal material
    n_Z: an array containing the refractive index of the ATR crystal material
    nu: an array containing the wavenumbers of the measured spectrum
    Ab: an array containing the absorption values of the ATR infrared spectrum
    nu_std: an array containing the wavenumbers of the standard spectrum
    k_std: an array containing the standard k values against which calibration is performed
    n_std: an array containing the standard n values against which calibration is performed
    returns the effective number of reflections
    n_S: the refractive index of the sample at a high wavenumber (default = 1.5)
    (water=1.33041, toluene=1.477012, chlorobenzene=1.503378, benzene=1.480162)
    nRef: the effective number of reflections of the spectrometer """
    
    # Setup
    angI = 45.0 # the angle of incidence for the ATR setup (in degrees)
    angI = math.radians(angI)
    Ab[np.where(Ab < 0)] = 0.0
     
    Ab = np.expand_dims(Ab, axis=1)
    if len(nu.shape) < 2: nu = np.expand_dims(nu, axis=1)
    
        # interpolation to find the refractive index of ZnSe at the spectral wavelengths
    tck = interpolate.splrep(nu_Z, n_Z, s=0)
    n_Z = interpolate.splev(nu, tck, der=0)

    spline2 = interpolate.splrep(nu_std, n_std, s=0)
    n_std = interpolate.splev(nu, spline2, der=0)
    spline3 = interpolate.splrep(nu_std, k_std, s=0)
    k_std = interpolate.splev(nu, spline3, der=0)
    
    n_crit = n_Z*math.sin(angI) # calculation of critical sample refractive index
    excld = np.ones_like(nu)
    excld[np.where(n_crit < n_std)] = 0. # determining which values do not satisfy total internal reflection condition and should be excluded
    
    a = n_std**2 - k_std**2 - n_Z**2 * math.sin(angI)**2
    b = 2*n_std*k_std
    A = np.sqrt((np.sqrt(pow(a,2)+pow(b,2))+a)/2.0)
    B = np.sqrt((np.sqrt(pow(a,2)+pow(b,2))-a)/2.0)
    Rs = (pow(A**2+B**2-n_Z**2 * math.cos(angI)**2,2)+4*B**2 * n_Z**2 * math.cos(angI)**2)/pow(np.add(A,n_Z*math.cos(angI))**2+B**2,2)
    
    # Solving for effective number of reflections -------------------------------
    bnds = ((0.4, 1.0),)
    result = differential_evolution(calib, bnds, args=(nu, Ab, Rs, excld, nRef), maxiter=200, tol=0.000000001, polish=False)
    p = result.x

    return p[0]

def op_constants(nu_Z, n_Z, nu, Ab, efRef, n_S=1.5, nRef=1, nu_ext=None, k_ext=None, tol=0.0001, maxiter=20):
    """ op_constants calculates the optical constants of a material from its
    ATR infrared absorbance spectrum.
    nu_Z: an array containing the wavenumbers of the refractive index data for the ATR crystal material
    n_Z: an array containing the refractive index of the ATR crystal material
    nu: an array containing the wavenumbers of the measured spectrum
    Ab: an array containing the absorption values of the ATR infrared spectrum
    nu_ext: an array containing the wavenumbers of the extension to the measured spectrum
    k_ext: an array containing the k values used to extend the measured spectrum down to 0 cm$^{-1}$
    efRef: the effective number of reflections for the spectrometer
    n_S: the refractive index of the sample at a high wavenumber (default = 1.5)
    nRef: the number of reflections on the ATR crystal-sample interface (default = 1)
    tol: the tolerance limit for determining if iteration has converged (default = 0.005)
    maxiter: the maximum number of iterations performed (default = 20)
    returns an array containing the k and n spectra of the material """

    # Setup
    angI = 45.0 # the angle of incidence for the ATR setup (in degrees)
    angI = math.radians(angI)
    Ab[np.where(Ab < 0)] = 0.0
    
    Ab = np.expand_dims(Ab, axis=1)
    if len(nu.shape) < 2: nu = np.expand_dims(nu, axis=1)
    
        # interpolation to find the refractive index of ZnSe at the spectral wavelengths
    tck = interpolate.splrep(nu_Z, n_Z, s=0)
    n_Z = interpolate.splev(nu, tck, der=0)

    if nu_ext is not None:
        nu_min = np.amin(nu)
        jnk2 = np.searchsorted(nu_ext, nu_min)
        k_ext = k_ext[:jnk2]
        nu_ext = nu_ext[:jnk2]
    
        nu_ext = np.expand_dims(nu_ext, axis=1)
        k_ext = np.expand_dims(k_ext, axis=1) 
            
        n_ext = KKtrans(nu_ext, k_ext, np.array([[0.0]]), np.array([[0.0]]))
        a = pow(n_ext,2) - pow(k_ext,2) - pow(2.38,2)*pow(math.sin(angI),2)
        b = 2*n_ext*k_ext
        A = np.sqrt((np.sqrt(pow(a,2)+pow(b,2))+a)/2.0)
        B = np.sqrt((np.sqrt(pow(a,2)+pow(b,2))-a)/2.0)
        Rs_ext = (pow(pow(A,2)+pow(B,2)-pow(2.38,2)*pow(math.cos(angI),2),2)+4*pow(B,2)*pow(2.38,2)*pow(math.cos(angI),2))/pow(pow(np.add(A,2.38*math.cos(angI)),2)+pow(B,2),2)   

    n_crit = n_Z*math.sin(angI) # calculation of critical sample refractive index
    
    # Calculating k and n spectra
    Rs = pow(np.sqrt(1+8*np.power(10,-Ab))/2.0-0.5,1/(nRef*efRef))
    Rs0 = np.copy(Rs)
    if nu_ext is not None:
        th_p = KKtransTH(nu,Rs, nu_ext, Rs_ext)
    else:
        th_p = KKtransTH(nu,Rs)
    dth = math.pi-2*np.arctan(np.sqrt(pow(n_Z,2)*pow(math.sin(angI),2)-pow(n_S,2))/(n_Z*math.cos(angI))) - th_p[-1,0]
    th = np.add(th_p,dth)
    ep1 = pow(n_Z,2)*(pow(math.sin(angI),2)+(pow(math.cos(angI),2)*(pow(1-Rs,2)-4*Rs*pow(np.sin(th),2)))/pow(1+Rs-2*np.sqrt(Rs)*np.cos(th),2))
    ep2 = pow(n_Z,2)*(4*pow(math.cos(angI),2)*np.sin(th)*(1-Rs)*np.sqrt(Rs))/pow(1+Rs-2*np.sqrt(Rs)*np.cos(th),2)
    k = np.sqrt(0.5*(np.sqrt(pow(ep1,2)+pow(ep2,2))-ep1))
    if nu_ext is not None:
        n = KKtrans(nu, k, nu_ext, k_ext, n_S=n_S)
    else:
        n = KKtrans(nu, k, n_S=n_S)
    
    # Iterating until solution converges --------------------------------------
    ctr = 0
    err = 2.0
    
    while (err > tol) and (ctr < maxiter):
        ctr += 1
        
        a = pow(n,2) - pow(k,2) - pow(n_Z,2)*pow(math.sin(angI),2)
        b = 2*n*k
        A = np.sqrt((np.sqrt(pow(a,2)+pow(b,2))+a)/2.0)
        B = np.sqrt((np.sqrt(pow(a,2)+pow(b,2))-a)/2.0)
        sgn = (-1.*n_Z**2 * math.cos(angI)**2 + A**2 + B**2)/((A + n_Z*math.cos(angI))**2 + B**2)
        th = np.arctan((2.*B*n_Z*math.cos(angI))/(-1.*n_Z**2 * math.cos(angI)**2 + A**2 + B**2))
        th[np.where(sgn < 0)] = th[np.where(sgn < 0)] + math.pi
        th[np.where(sgn == 0)] = math.pi/2.
        
        ep1 = pow(n_Z,2)*(pow(math.sin(angI),2)+(pow(math.cos(angI),2)*(pow(1-Rs0,2)-4*Rs0*pow(np.sin(th),2)))/pow(1+Rs0-2*np.sqrt(Rs0)*np.cos(th),2))
        ep2 = pow(n_Z,2)*(4*pow(math.cos(angI),2)*np.sin(th)*(1-Rs0)*np.sqrt(Rs0))/pow(1+Rs0-2*np.sqrt(Rs0)*np.cos(th),2)
        k = np.sqrt(0.5*(np.sqrt(pow(ep1,2)+pow(ep2,2))-ep1))
        if nu_ext is not None:
            n = KKtrans(nu, k, nu_ext, k_ext, n_S=n_S)
        else:
            n = KKtrans(nu, k, n_S=n_S)
        
        a = pow(n,2) - pow(k,2) - pow(n_Z,2)*pow(math.sin(angI),2)
        b = 2*n*k
        A = np.sqrt((np.sqrt(pow(a,2)+pow(b,2))+a)/2.0)
        B = np.sqrt((np.sqrt(pow(a,2)+pow(b,2))-a)/2.0)
        Rs = (pow(pow(A,2)+pow(B,2)-pow(n_Z,2)*pow(math.cos(angI),2),2)+4*pow(B,2)*pow(n_Z,2)*pow(math.cos(angI),2))/pow(pow(np.add(A,n_Z*math.cos(angI)),2)+pow(B,2),2)
        Abc = -np.log10((pow(Rs,nRef*efRef)+pow(Rs,2*nRef*efRef))/2.0)
    
        err_all = np.add(Ab,-Abc)
        err_all[np.where(n > n_crit)] = 0.
        err = np.sum(np.abs(err_all))

    # Output
    oData = np.hstack((k,n))    
    return oData
