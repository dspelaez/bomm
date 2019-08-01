#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
# vim:fdm=marker
#
# =============================================================
#  Copyright © 2018 Daniel Santiago <dpelaez@cicese.edu.mx>
#  Distributed under terms of the GNU/GPL license.
# =============================================================

"""
This module contains function to write a NetCDF4 file from the BOMM data.
"""

# --- import libs ---
import numpy as np
import scipy.signal as signal
import datetime as dt
import netCDF4 as nc
import yaml
import gsw
import sys
import os
#
import src.processing.wdm as wdm
import src.processing.motion_correction as motcor
#


# functions to compute importan variables
# fourier spectrum {{{
def welch(x, fs, nfft=512, overlap=128):
    """Computes the Fourier periodograms ignoring segments with NaNs."""

    # check how if all data is nan
    n = len(x)
    nans = len(np.where(np.isnan(x))[0])
    if n == nans:
        raise Exception("Array is full of NaNs.")

    # loop for each segment
    S = []
    for j in np.arange(0, n-nfft+overlap, overlap):
        arr = x[j:j+nfft]
        nans = len(np.where(np.isnan(arr))[0])
        if nans == 0 and len(arr) == nfft:
            f, S0 = signal.welch(arr, fs, window="hann", nperseg=nfft)
            S += [S0]

    return f, np.mean(S, axis=0)

# }}}

# wavenumber {{{
def wavenumber(f, d=100, mode="hunt"):
    """
    mode = "exact"
    --------------
    Calculo del numero de onda usando la relacion de dispersion de la teoria
    lineal resuelta con un metodo iterativo (punto fijo) sin tener en cuenta
    el efecto doppler de las corrientes en la frecuencia de las olas.
            2
          w    =   g * k * tanh(k*h)    --->    w  = 2*pi*f

    mode = "hunt"  (default)
    ------------------------
    Calculo del numero de onda usando la aproximacion empirica propuesta por Hunt 1979
              2        2             y
          (kh)    =   y   +  --------------------
                                   6
                                 ----      n
                             1 + \    d   y
                                 /     n
                                 ----
                                 n = 1
               2
          y = w  h  / g

          d0 = [0.666, 0.355, 0.161, 0.0632, 0.0218, 0.0065]
        """

    if d < 0:
        raise ValueError("Depth must be positive")

    if mode == "exact":
        #
        tol = 1e-9
        maxiter = 1000000
        g = 9.8
        w = 2.* np.pi * f
        k0 = (w**2.)/g
        for cnt in range(maxiter):
            k = (w**2)/(g*np.tanh(k0*d))
            k0 = k
            if all(abs(k - k0) >= tol):
                return k0
        return k

    elif mode == "hunt":
        #
        d0 = [0.666, 0.355, 0.161, 0.0632, 0.0218, 0.0065]
        g = 9.8
        w = 2.* np.pi * f
        y = (w**2)*d/g
        #
        poly = np.zeros_like(f)
        for n, dn in enumerate(d0):
            poly = poly + dn * y**(n+1)
        #
        k = np.sqrt(y**2 + y/(1 + poly))/d

        return k
        #
    else:
        raise ValueError("`mode` must br `hunt` o `exact`")
# }}}

# compute wave parameters {{{
def wave_parameters(frqs, dirs, E):
    """"Return basic bulk wave parameters from the directional wave spectrum."""

    # TODO: check for nans

    # integrate directional wave specrtum
    dirs = np.radians(dirs)
    S_int = np.trapz(E, x=dirs, axis=0)
    D_int = np.trapz(E, x=frqs, axis=1)

    # computes m,n oder moments of the spectrum
    # indices <<mn>> represents the exponents of f and S respectivaly
    m = lambda n, p: np.trapz((frqs**n)*(S_int**p), x=frqs)
    
    # compute basic parameters
    Hm0 = 4.0 * np.sqrt(m(0,1))
    Tp1 = m(0,4) / m(1,4)
    #
    # compute directional params
    m = lambda n, p: np.trapz((dirs**n)*(D_int**p), x=dirs)
    # pDir = np.mod(dirs[np.argmax(D_int)], 2*np.pi)
    pDir = m(1,4) / m(0,4)
    mDir = m(1,1) / m(0,1)

    return Hm0, Tp1, np.degrees(pDir), np.degrees(mDir)

# }}}

# compute stokes drift profile {{{
def stokes_drift(f, S, z=-np.logspace(-5,2,50)):
    """Compute stokes drift profile as Breivik et al 2016 eq5."""
    
    # angular frequency and spectrum in right units
    g = 9.8
    k = wavenumber(f, 100) 
    w = 2*np.pi * f
    Sw = S / (2*np.pi)
    
    fac = 2 / g
    if isinstance(z, float) or isinstance(z, int):
        dummy = w**3 * Sw * np.exp(2*k*z)
    else:
        dummy = w[None,:]**3 * Sw[None,:] * np.exp(2*k[None,:]*z[:,None])
    return np.trapz(fac*dummy, w)

# }}}

# remove outliers {{{
def remove_outliers(x):
    """Recursively remove outliers from a give signal"""

    # compute mean and standar deviation
    xmean, xstd = nanmean(x), np.nanstd(x)

    # first remove values lying 5 time std
    x_clean = x.copy()
    x_clean[abs(x - xmean) > 5*xstd] = np.nan

    return x_clean
# }}}

# eddy correlation method {{{
def eddy_correlation_flux(U, V, W, T):
    """Computes momentum and heat fluxes from corrected velocity components.

    Args:
        U, V, W, T (array): Array with sonic anemometer variables.
    
    Returns:
        tuple: x and y momentum flux and latent heat flux.

    References:
        * https://www.eol.ucar.edu/content/wind-direction-quick-reference
    """

    # check for NANs
    for v in (U, V, W, T):
        nans = np.isnan(v)
        if len(nans.nonzero()[0]) / len(v) < 0.1:
            v[nans] = nanmean(v)
        else:
            raise Exception("More than 10% of invalid data")

    # perform a decimate and remove outliers

    # align with max variability axis (average V = 0)
    theta = np.arctan2(nanmean(V), nanmean(U)) #<- from U to V counterclockwise
    U_stream =  U*np.cos(theta) + V*np.sin(theta)
    V_stream = -U*np.sin(theta) + V*np.cos(theta)

    # align with the flow to do mean W equals zero
    phi = np.arctan2(nanmean(W), nanmean(U_stream)) #<- from U to W counterclockwise
    U_proj =  U_stream*np.cos(phi) + W*np.sin(phi)
    V_proj =  V_stream.copy()
    W_proj = -U_stream*np.sin(phi) + W*np.cos(phi)

    def flux(x, y):
        window = np.ones(10) / 10
        xf = np.convolve(window, remove_outliers(x), "same")
        yf = np.convolve(window, remove_outliers(y), "same")
        return nanmean((xf-nanmean(xf)) * (yf-nanmean(yf)))

    # compute turbulent fluxes
    u, v, w, T = U_proj, V_proj, W_proj, T
    uw, vw, wT = flux(u, w), flux(v, w), flux(w, T)
    

    return uw, vw, wT
    # }}} 

# janssen dissipation {{{
def janssen_source_diss(frqs, dirs, E):
    """Implementation of the WAM Cycle 4 dissipation source term.
    
    Args:
        frqs: Frequency array
        dirs: Directions array
        E: Directional wave spectrum
    
    Returns:
        Numpy array with the dissipation source term

    References:
        * Mastenbroek, C., Burgers, G., & Janssen, P. A. E. M. (1993). The
          dynamical coupling of a wave model and a storm surge model through the
          atmospheric boundary layer. Journal of physical Oceanography, 23(8),
          1856-1866.

    """

    # function to compute spectral moments
    S = np.trapz(E, x=np.radians(dirs), axis=0)
    m = lambda n, p: np.trapz((frqs**n)*(S**p), x=frqs)
    integral = lambda X: np.trapz(np.trapz(X, frqs), np.radians(dirs))

    # compute phase speed
    w = 2 * np.pi * frqs
    k = sp.wavenumber(frqs, d=100)
    c = w / k

    # comupute averaged parameters
    fm = m(1,1) / m(0,1)
    wm = 2 * np.pi * fm
    km = sp.wavenumber(fm, d=100)
    m0 = m(0,1)

    # compute source term
    factor = -2.25 * wm * (m0*km**2)**2 * (k / km + k**2 / km**2)
    Sds = factor[None,:] * E
    tau_ds = rhow*g * integral(Sds/c[None,:]) 
    Eds = g * integral(Sds)

    return Sds, tau_ds, Eds
# --- }}}

# janssen wind input {{{
def janssen_source_input(frqs, dirs, E, ustar, udir, **kwargs):
    """Implementation of the WAM Cycle 5 wind input source term.

    Args:
        frqs (array): Frequency array
        dirs (array): Directions array
        E (array): Wave spectrum
        ustar (float): Air-side friction velocity
        udir (float): Wind direction

    Returns:
        Wind input source term.

    References:
        * Mastenbroek, C., Burgers, G., & Janssen, P. A. E. M. (1993). The
          dynamical coupling of a wave model and a storm surge model through the
          atmospheric boundary layer. Journal of physical Oceanography, 23(8),
          1856-1866.

    """

    # function to integrate quickly
    integral = lambda X: np.trapz(np.trapz(X, frqs), np.radians(dirs))

    # get kwargs
    rhoa = kwargs.get('rhoa', 1.25)
    rhow = kwargs.get('rhow', 1020)
    depth = kwargs.get('depth', 100)

    # parameters
    g = 9.8
    kappa = 0.41
    dens_rel = rhoa / rhow
    alpha_garret = 0.0144

    # funtion to recursevily estimate wind input
    def source_input_estimation(ze):

        # compute directional distribution
        dir2 = (dirs - udir) % 360
        mask = np.logical_and(dir2>90, dir2<270)
        cos = np.cos(np.radians(dirs-udir))
        cos[mask] = 1E-256

        # compute Miles parameter
        ustar_along_wind = abs(ustar * cos)
        wave_age = c[None,:] / ustar_along_wind[:,None]
        # wave_age =  c[None,:] * cos[:,None] / ustar
        mu = (g * ze / c[None,:]**2) * np.exp(kappa * wave_age)
        mu[mu >= 1] = 1
        
        # compute beta paremter
        # ---> Masterbroek eq13
        beta = (1.2 / kappa**2) * mu * np.log(mu)**4

        # compute source term
        factor = w[None,:] * dens_rel * beta * (ustar / c[None,:]) ** 2
        return factor * cos[:,None]**2 *  E

    # compute phase speed
    w = 2 * np.pi * frqs
    try:
        k = wavenumber(frqs, d=depth)
    except:
        k = w**2 / g
    c = w / k

    # compute roughness lenght iteratively
    tau = rhoa * ustar**2
    z0 = alpha_garret * ustar**2 / g

    # first estimation
    Sin = source_input_estimation(z0)
    tauw = rhow*g * integral(Sin / c[None,:])
    Ein = g * integral(Sin)
    ze = z0
    if tauw > tau:
        return Sin, tauw, ze, z0, Ein

    # iteration
    else:
        for counter in range(10):
            Sin = source_input_estimation(ze)
            tauw = rhow*g * integral(Sin / c[None,:])
            Ein = g * integral(Sin)
            ze = z0 / np.sqrt(1 - tauw / tau)
    
        # return variables
        return Sin, tauw, ze, z0, Ein
# --- }}}

# wind speed at neutral conditions and monin obukov {{{
def monin_obukhov(ustar, wT, T, z):
    """Compute Monin-Obukhov similarity lenght."""
    
    # compute similarity lenght as the ralationship between turbulence
    # production scaling and buoyancy flux (surface sensible heat).
    g = 9.8
    kappa = 0.4
    L = (T * ustar**3) / (kappa*g*wT)
    return z / L
    

def wind_speed_neutral(zL, U, ustar, height=6.5):
    """Compute U10N with stability function for mometum dut to Donelan 1990."""

    # find stable and unstable indices
    ix_stab = zL > 0
    ix_unst = zL < 0

    # allocate array
    Psi = np.zeros_like(zL) * np.nan

    # for unstable conditions: zL<0
    Psi[ix_unst] = (1 + 15.2 * np.abs(zL[ix_unst]))**-0.25
    Psi[ix_stab] = 1 + 4.8*zL[ix_stab]

    # force to be valid between -10 < zL < 10
    Psi[abs(zL) > 10] = np.nan

    # compute wind speed for neutral conditions
    kappa = 0.4
    UzN = U + (ustar / kappa) * Psi

    # compute wind at 10 meters above sea level
    U10N = UzN + (ustar / kappa) * np.log(10 / height)

    return U10N
# }}}

# air density {{{
def air_dens(Ta, rh, Pa):
    """Computes the density of moist air.

    Parameters
    ----------
    Ta : array_like
        air temperature [:math:`^\\circ` C]
    rh : array_like
        relative humidity [percent]
    Pa : array_like, optional
        air pressure [mb]

    Returns
    -------
    rhoa : array_like
        air density [kg m :sup:`-3`]

    See Also
    --------
    TODO: qsat

    Examples
    --------
    >>> from airsea import atmosphere as asea
    >>> asea.air_dens([5., 15., 23.1], 95.)
    array([ 1.27361105,  1.22578105,  1.18750778])
    >>> asea.air_dens([5., 15., 23.1], [95, 100, 50], 900)
    array([ 1.12331233,  1.08031123,  1.05203796])

    Modifications: Original from AIR_SEA TOOLBOX, Version 2.0
    04/07/1999: version 1.2 (contributed by AA)
    08/05/1999: version 2.0
    11/26/2010: Filipe Fernandes, Python translation.
    
    Acknowledgements
    ----------------
    This function was taken from pyoceans/python-airsea
    """

    # force to be numpy array
    Ta, rh, Pa = np.asarray(Ta), np.asarray(rh), np.asarray(Pa)

    # compute the specific humidity [kg/kg] at saturation at air temperature
    sflag = True
    ew = (6.1121 * (1.0007 + 3.46e-6 * Pa) *
          np.exp((17.502 * Ta) / (240.97 + Ta)))  # [mb]
    qsat = 0.62197 * (ew / (Pa - 0.378 * ew))  # [mb] -> [g/kg]
    qsat = (1.0 - 0.02 * sflag) * qsat  # flag for fresh (0) or salt (1) water

    # compute air density
    gas_const_R = 287.04  # NOTE: 287.1 in COARE
    eps_air = 0.62197
    o61 = 1 / eps_air - 1    # 0.61 -> Moisture correction for temperature
    Q = (0.01 * rh) * qsat   # Specific humidity of air [kg/kg]
    T = Ta + 273.16
    Tv = T * (1 + o61 * Q)  # Air virtual temperature.
    rhoa = (100 * Pa) / (gas_const_R * Tv)  # Air density [kg/m**3]

    return rhoa
# }}}


# useful functions
# get data from netcdf {{{
def get_netcdf_data(grp, date, number_of_minutes=30):
    """Return the data corresponding to the netCDF group for a specific date.
    
    Args: TODO
    Returns:
        Dictionary containig all variables.
    
    """

    # start and final indices
    fs = grp.sampling_frequency
    if isinstance(fs, str):
        fs = eval(grp.sampling_frequency)

    # assure the sampling frequency of at least 10 minutes
    fsmin = 1./600.
    if fs < fsmin:
        number_of_minutes = 30 # minimum sampling rate

    # number of samples in a day
    N = int(fs * 24 * 3600)
    hour, minute = date.hour, date.minute

    # check number of minutes
    if number_of_minutes > N:
        raise ValueError(f"Number of minutes must be less than a day. Max={N}.")

    # start and final index
    i = int(fs*hour*3600 + fs*minute*60)
    j = i + int(fs*number_of_minutes*60)

    dic = {}
    dic["time"] = grp["time"][i:j]
    for k in grp.variables.keys():
        if k not in ["time"]:
            dic[k] = grp[k][i:j]

    return dic
    # }}}

# simple_despike {{{
def simple_despike(x, isangle=False):
    """Remove some strange data from the time series."""
    return nanmean(x, isangle=isangle)
# }}}

# nanmean {{{
def nanmean(x, isangle=False):
    """Fancy nanmean without akward warning msg. If isangle, must be in degrees."""
    
    x = np.asarray(x)
    nans = np.isnan(x).nonzero()[0]
    if len(nans) == len(x):
        return np.nan
    else:
        if isangle:
            return (np.angle(np.nanmean(np.exp(1j*np.radians(x))))*180/np.pi)
        else:
            return np.mean(x[~np.isnan(x)])
# }}}

# convert to decimal degree {{{
def convert_to_decimal_degree(x):
    """Convert ungly number to latitude of longitude coordinates."""
    ddmm, ss = str(x).split(".")
    mm, dd = float(ddmm[-2:]) , float(ddmm[:-2])
    return np.sign(dd) * (np.abs(dd) + (mm + float(f".{ss}")) / 60.)
# }}}


# ad-hoc functions
# rbr_data_correction {{{
def rbr_data_correction(fname):
    """Remove some strange data from the time series in RBR data.
    
    This function is inteded to be run after the dataset was generated. The
    TEOS10 equations are used here to correct the RBR conductivity readings.
    """
    
    import warnings
    warnings.filterwarnings("ignore",category=RuntimeWarning)

    # open dataset as append mode
    dataset = nc.Dataset(fname, "a")

    # extract data into numpy arrays
    Cw, Tw, Sw, p = (dataset[v][:].filled() for v in ["Cw", "Tw", "Sw", "depth"])

    # remove salinity data when the gradient is greater than 0.05
    ix = np.append(False, np.abs(np.diff(Sw)) >= 0.1)
    Sw[ix] = np.nan

    # compute conductivity from teos
    Sw_clean = Sw.copy()
    Sw_clean[np.isnan(Sw)] = np.nanmean(Sw)
    Cw_from_teos = gsw.C_from_SP(Sw_clean, Tw, p)
    Sw_from_teos = gsw.SP_from_C(Cw_from_teos, Tw, p)

    # compute the clean water density
    rhow = gsw.rho(Sw_clean, Tw, p)
    
    # save data into the dataset
    dataset["rhow"][:] = rhow
    dataset["Cw"][:] = 0.1*Cw + 0.9*Cw_from_teos
    dataset["Sw"][:] = Sw
    #
    dens_rel = np.sqrt(dataset["rhoa"][:]/dataset["rhow"][:])
    dataset["wstar"][:] = dens_rel * dataset["ustar"][:]
    
    # close dataset
    dataset.close()

# }}}

# wstaff data correction {{{
def isvalid_wstaff(eta):
    """Check if surface elevation measured by the wavestaff is valid"""
    
    # conversiotn factor (from counts to meters)
    fac = 3.5/4095

    # we have, so far, three cases here: 1) when the wire timeseries is all
    # zeros. 2) when it is near 4095. 3) when we have a lot of spikes.
    #
    valid = True
    #
    if np.nanmean(eta) == 0.0:
        valid = False
    
    if (np.nanmean(eta) + np.nanstd(eta)) > 4095:
        valid = False

    return valid
# }}}



# oop class to handle processing
# main class {{{
class ProcessingData(object):

    """
    This class contains methods to process the level-1 netCDF4 files for the
    CICESE-BOMM (Oceanographic and Marine Meteorology Buoys).
    """

    _list_of_dictionaries = "ekx wnd gps mvi met pro rbr sig vec wav".split()
    __slots__ = "metadata list_of_variables r Acc Gyr Eul".split() +\
                 _list_of_dictionaries + ["number_of_minutes"] + \
                 ["U_unc", "U_rot", "U_cor", "X", "Y", "Z"] + \
                 ["valid_wires", "valid_wires_index"]

    # private methods {{{
    def __init__(self, metafile, number_of_minutes=30):
        """Function to initialize the class.

        Args:
            date (datetime): datetime object
        """

        # load metadata
        self.number_of_minutes = number_of_minutes
        with open(metafile, "r") as f:
            self.metadata = yaml.load(f)

    # run function
    def run(self, date):
        """Run all the processing scripts"""

        # global variables
        basepath = self.metadata["basepath"]
        bomm_name = self.metadata["name"]

        # date and filename
        nm = self.number_of_minutes
        filename = f"{basepath}/{bomm_name}/level1/{date.strftime('%Y%m%d')}.nc"

        # load data as dictionaries
        self.r = {}
        self.list_of_variables = {}
        with nc.Dataset(filename, "r") as data:
            self.ekx = get_netcdf_data(data["ekinox"],    date, nm)
            self.wnd = get_netcdf_data(data["sonic"],     date, nm)
            self.gps = get_netcdf_data(data["gps"],       date, nm)
            self.mvi = get_netcdf_data(data["marvi"],     date, nm)
            self.met = get_netcdf_data(data["maximet"],   date, nm)
            self.pro = get_netcdf_data(data["proceanus"], date, nm)
            self.rbr = get_netcdf_data(data["rbr"],       date, nm)
            self.sig = get_netcdf_data(data["signature"], date, nm)
            self.vec = get_netcdf_data(data["vector"],    date, nm)
            self.wav = get_netcdf_data(data["wstaff"],    date, nm)

        # save date in the results dictionary
        self.r["time"] = date

        # get data that dont depends on the motion correction
        _list_of_methods = [self.air_data, self.water_data, self.electronics_data]
        for function in _list_of_methods:
            try:
                function()
            except Exception as e:
                exc_type, exc_obj, exc_tb = sys.exc_info()
                string = f"{date}: {exc_type.__name__} ocurred in function " +\
                         f"self.{function.__name__}() " + \
                         f"at line {exc_tb.tb_lineno} ---> Error: {exc_obj}"
                print(string)
        
        # get data that depends on the motion correction
        _list_of_methods = [
                self.motion_matrices, self.wave_data,
                self.wind_data, self.current_data
                ]
        for function in _list_of_methods:
            try:
                function()
            except Exception as e:
                exc_type, exc_obj, exc_tb = sys.exc_info()
                string = f"{date}: {exc_type.__name__} ocurred in function " +\
                         f"self.{function.__name__}() " + \
                         f"at line {exc_tb.tb_lineno} ---> Error: {exc_obj}"
                print(string)
    # }}}

    # check for nans in input variables {{{
    def check_nans(self, dic, limit=0.3):
        """Check if some of our dictionaries has more nans than wanted."""

        valid = True
        for k,v in dic.items():
            try:
                number_of_nans = len(np.nonzero(v.mask)[0])
                if (number_of_nans / len(v)) >= limit:
                    valid = False
                    return valid
            except AttributeError as e:
                pass
        return valid

    # }}}

    # compute buoy heading {{{
    def compute_heading(self):
        """Compute the heading from different sources. Return heading in deg."""

        # TODO: when magnetometre will be available choose it as default
        if hasattr(self, "mag"):
            pass

        # heading signature
        heading_sig = (self.sig["heading"]/100) % 360
        
        # heading maximet
        maximet_angle = self.metadata["sensors"]["maximet"]["maximet_angle"]
        true_wnd, rel_wnd = self.met["true_wind_dir"], self.met["relative_wind_dir"]
        heading_met = (true_wnd - rel_wnd + maximet_angle) % 360

        # TODO:
        # the low frequency heading means the angle between new BOMM y-axis and
        # true north. Magnetic deviation is taken from GPS measurements. All in
        # degrees, the mag deviation or declination is added to the current
        # magnetic mesurement ---> (check this, im not pretty sure)
        if np.isnan(heading_sig.filled(np.nan)).all():
            heading = heading_met - self.gps["mag_var"][0] * 0
        else:
            heading = heading_sig - self.gps["mag_var"][0] * 0

        return heading % 360
    # }}}

    # motion matrices {{{
    def motion_matrices(self):
        """Matrices of the accelerometer, gyroscope and euler angles"""

        # check for anomalous data
        for k, v in self.ekx.items():
            if k not in ["time"]:
                self.ekx[k][abs(v) > 1E5] = np.nan

        # fill nans when possible
        for k, v in self.ekx.items():
            if k not in ["time"]:
                number_of_nans = np.isnan(v).sum()
                if (number_of_nans / len(v)) < 0.1:
                    self.ekx[k][np.isnan(v)] = np.nanmean(v)


        # cutoff and merging frequency
        fs = self.metadata["sensors"]["ekinox"]["sampling_frequency"]
        fc = self.metadata["sensors"]["ekinox"]["cutoff_frequency"]
        fm = self.metadata["sensors"]["ekinox"]["gyro_merging_frequency"]

        # construct accelerometer and gyroscope tuples
        # apply a rotation to an ENU frame of reference
        R = (np.pi, 0, np.pi/2)
        self.Acc = motcor.vector_rotation((self.ekx["accel_x"],
            self.ekx["accel_y"], self.ekx["accel_z"]), R)
        #
        self.Gyr = motcor.vector_rotation((self.ekx["gyro_x"],
            self.ekx["gyro_y"],  self.ekx["gyro_z"]),  R)

        # integrate accel and gyro to obtain euler angles
        phi, theta = motcor.pitch_and_roll(*self.Acc, *self.Gyr,
                fs=fs, fc=fc, fm=fm)
        #
        # compute bomm heading and the merge with ekinox
        # original heading lecture is an angle measured from north, so we need
        # do 90-angle. Then we need to substract 90 to orientate with x axis
        heading = np.radians((90 - self.compute_heading() - 90) % 360)
        psi = motcor.yaw_from_magnetometer(self.Gyr[2], heading,
                fs=fs, fc=fc, fm=fm)
        #
        # TODO: from BOMM3 the ekinox was updated to output euler angles
        #       so we need choose phi and theta directly and psi from the
        #       combination between the magnetometre and the ekinox
        self.Eul = (phi, theta, psi)

        # finally save the mean euler angles in the results dictionary
        self.r["roll"] = nanmean(self.Eul[0]*180/np.pi, isangle=True)
        self.r["pitch"] = nanmean(self.Eul[1]*180/np.pi, isangle=True)
        self.r["yaw"] =  nanmean(self.Eul[2]*180/np.pi, isangle=True)
        #
        list_of_variables = {
                "roll"    : "average_roll_angle",
                "pitch"   : "average_pitch_angle",
                "yaw"     : "average_yaw_angle"
                }
        self.list_of_variables = {**self.list_of_variables, **list_of_variables}

    # }}}


    # air data {{{
    def air_data(self):
        """Compute averages of the air-quality data. Includes location from GPS."""

        # TODO: It is strongly recommended to limit the range of the
        #       data in the netcdf valid_range attribute.
        # compute variables from Maximet
        tWdir = simple_despike(self.met["true_wind_dir"], isangle=True)
        rWdir = simple_despike(self.met["relative_wind_dir"], isangle=True)
        Wspd = simple_despike(self.met["wind_speed"])
        Pa = simple_despike(self.met["atm_pressure"])
        Ta = simple_despike(self.met["air_temp"])
        rhum = simple_despike(self.met["rel_humidity"])
        DP = simple_despike(self.met["dew_point"])
        total_rain = simple_despike(self.met["total_rain"])
        rain_rate = simple_despike(self.met["rain_intensity"])
        
        # compute variables from Proceanus
        aCO2 = simple_despike(self.pro["air_co2"])
        wCO2 = simple_despike(self.pro["wat_co2"])
        ahum = simple_despike(self.pro["air_humidity"])
        rhoa = air_dens(Ta, rhum, Pa)

        # compute data from gps
        lat = convert_to_decimal_degree(simple_despike(self.gps["latitude"]))
        lon = convert_to_decimal_degree(simple_despike(self.gps["longitude"]))
        
        # save data in the output dictionary
        list_of_variables = {
                "tWdir"      : "true_wind_direction",
                "rWdir"      : "relative_wind_direction",
                "Wspd"       : "wind_speed",
                "Pa"         : "air_pressure",
                "Ta"         : "air_temperature",
                "rhum"       : "relative_humidity",
                "DP"         : "dew_point_temperature",
                "total_rain" : "total_rainfall",
                "rain_rate"  : "rainfall_rate",
                "aCO2"       : "air_co2",
                "wCO2"       : "water_co2",
                "ahum"       : "air_humidity",
                "rhoa"       : "air_density",
                "lat"        : "latitude",
                "lon"        : "longitude"
                }
        #
        for k, v in list_of_variables.items():
            self.r[k] = eval(k)
        #
        # append to global list of variables
        self.list_of_variables = {**self.list_of_variables, **list_of_variables}
        
    # }}}

    # water data {{{
    def water_data(self):
        """Compute averages of the water-quality data."""

        # TODO: It is strongly recommended to limit the range of the
        #       data in the netcdf valid_range attribute.
        # compute variables from RBR
        pH = simple_despike(self.rbr["ph"])
        Sw = simple_despike(self.rbr["salinity"])
        Tw = simple_despike(self.rbr["temperature"])
        Cw = simple_despike(self.rbr["conductivity"])
        depth = simple_despike(self.rbr["depth"])
        rhow = gsw.rho(Sw, Tw, depth)
        dissoxy = simple_despike(self.rbr["dissoxy"])

        # save data in the output dictionary
        list_of_variables = {
                "pH"      : "pH",
                "Sw"      : "water_salinity",
                "Tw"      : "water_temperature",
                "Cw"      : "water_conductivity",
                "rhow"    : "water_density",
                "depth"   : "water_depth",
                "dissoxy" : "dissolved_oxygen",
                }
        #
        for k, v in list_of_variables.items():
            self.r[k] = eval(k)
        #
        # append to global list of variables
        self.list_of_variables = {**self.list_of_variables, **list_of_variables}
    # }}}

    # wave data {{{
    def wave_data(self):
        """Compute directional wave spectrum and other wave parameters."""

        # check for nans: if both are valid do nothing, else raise exception
        if (self.check_nans(self.ekx) and self.check_nans(self.wav)):
            pass
        else:
            raise Exception("Number of nans is more than 30%")

        # # if heading is greater than 90 degress return an error
        # std_yaw = np.nanstd(self.Eul[2]) * 180/np.pi
        # if std_yaw > 15:
            # raise Exception(f"BOMM has veered too much: {std_yaw:.2f} deg")

        # conversion factor (from counts to meters)
        fac = 3.5/4095

        # check dimensions
        nfft = 1024
        npoint = 6
        ntime = len(self.wav["time"])

        # get the sampling frequency and the resampling factor
        fs = self.metadata["sensors"]["wstaff"]["sampling_frequency"]
        q = int(100/fs)

        # check waestaffs in use
        valid_wires = [i+1 for i in range(npoint)
            if isvalid_wstaff(self.wav[f"ws{i+1}"])]
        #
        # if bomm1--> force to remove first wire
        if self.metadata["name"] == "bomm1_its":
            valid_wires = valid_wires[1:]
        #
        # index of valid wires
        valid_wires_index = [w - 1 for w in valid_wires]

        # detect the most convinient cutoff frecuency
        fc = self.metadata["sensors"]["ekinox"]["cutoff_frequency"]
        # ff, SS = welch(self.ekx["accel_z"], fs=100, nfft=2**13)
        # SS[np.logical_or(ff>0.5, ff<0.05)] = 0.0
        # fp = ff[np.argmax(SS)]
        # fc = 0.5 * fp

        # determinte position of the wavestaffs
        # TODO: since offset depends on each bomm, they should be passed
        #       as an input argument.
        x_offset, y_offset, z_offset = -0.339, 0.413, 4.45
        xx, yy = wdm.reg_array(N=5, R=0.866, theta_0=90)
        # xx, yy = xx + x_offset, yy + y_offset

        # allocate variables
        S = np.zeros((int(nfft/2+1), npoint))
        X, Y, Z = (np.zeros((ntime, npoint)) for _ in range(3))
        #
        # apply the correction to the surface elevation and compute fourier spc
        for i, (x, y), in enumerate(zip(xx, yy)):
            #
            # get suface elevation at each point
            z = self.wav[f"ws{i+1}"] * fac + z_offset
            #
            # apply motion correction
            X[:,i], Y[:,i], Z[:,i] = motcor.position_correction((x,y,z),
                    self.Acc, self.Eul, fs=fs, fc=fc, q=q)
            #
            # remove surface elevation mean value
            Z[:,i] = Z[:,i] - np.nanmean(Z[:,i])
            #
            # compute fourier spectrum
            # TODO: compute spectrum with homemade pwelch, it will allow to
            # discard the blocks containing nan data.
            ffrq, S[:,i] = welch(Z[:,i], fs=fs, nfft=nfft, overlap=int(nfft/4))

        # save wavestaff position for later
        self.X, self.Y, self.Z = X, Y, Z
        self.valid_wires = valid_wires
        self.valid_wires_index = valid_wires_index

        # limit to the half of the nfft to remove high frequency noise
        S = np.mean(S[1:int(nfft/4)+1,valid_wires_index], axis=1)
        ffrq = ffrq[1:int(nfft/4)+1]

        # compute directional wave spectrum
        # TODO: create a anti-aliasing filter to decimate the time series
        d = lambda x: x[:,valid_wires_index]
        wfrq, dirs, E = wdm.wave_spectrum("fdir", d(Z), d(X), d(Y),
                fs=fs, limit=np.pi, omin=-5, omax=1, nvoice=16, ws=(30,5))
        
        # compute bulk wave parameters and stokes drift magnitude
        Hm0, Tp, pDir, mDir = wave_parameters(wfrq, dirs, E)
        Us0 = stokes_drift(ffrq, S, z=0.0)

        # save data in the output dictionary
        list_of_variables = {
                "ffrq" : "fourier_frequencies",
                "S"    : "frequency_spectrum",
                "wfrq" : "wavelet_frequencies",
                "dirs" : "wavelet_directions",
                "E"    : "directional_wave_spectrum",
                "Hm0"  : "significant_wave_height",
                "Tp"   : "peak_wave_period",
                "pDir" : "peak_wave_direction",
                "mDir" : "average_wave_direction",
                "Us0"  : "surface_stokes_drift"
                }
        #
        for k, v in list_of_variables.items():
            self.r[k] = eval(k)
        #
        # append to global list of variables
        self.list_of_variables = {**self.list_of_variables, **list_of_variables}
    # }}}

    # wind data {{{
    def wind_data(self):
        """Compute momentum flux and atmospheric parameters."""

        # check for nans: if both are valid do nothing, else raise exception
        if (self.check_nans(self.ekx) and self.check_nans(self.wnd)):
            pass
        else:
            raise Exception("Number of nans is more than 30%")

        # cutoff frecuency
        fc = 0.05

        # apply the correction to the anemometer data
        # TODO: This also should be passed as an argument or readed from the
        #       matadata yaml file.
        L = (0.339, -0.413, 13.01)
        # L = (0, 0, 13.01)
        sonic_angle = self.metadata["sensors"]["sonic"]["sonic_angle"] + 90
        sonic_height = self.metadata["sensors"]["sonic"]["sonic_height"]
        #
        U_unc = (self.wnd["u_wind"], self.wnd["v_wind"], self.wnd["w_wind"])
        U_rot = motcor.vector_rotation(U_unc, (0,0,sonic_angle), units="deg")
        U_cor = motcor.velocity_correction(U_rot, self.Acc, self.Eul, L, fs=100, fc=fc)

        # save corrected and uncorrected for later
        self.U_unc = U_unc
        self.U_rot = U_rot
        self.U_cor = U_cor

        # compute momentum fluxes
        T = self.wnd["sonic_temp"] + 273.15 # <--- convert to Kelvin
        uw, vw, wT = eddy_correlation_flux(U_cor[0], U_cor[1], U_cor[2], T)

        # compute average wind speed from anemometer
        Ua, Va, Ts = nanmean(U_cor[0]), nanmean(U_cor[1]), nanmean(T) - 273.15
        
        # perform correctction due to atmospheric instability
        #
        # air-sea density ratio
        default_value = lambda x,y: x if ~np.isnan(x) else y
        rhoa = default_value(self.r["rhoa"], 1.20)
        rhow = default_value(self.r["rhow"], 1024)
        dens_rel = rhoa / rhow
        #
        # air-side and water-side friction velocities
        ustar = (uw**2 + vw**2) ** 0.25
        wstar = np.sqrt(dens_rel) * ustar
        #
        # monin-obukhov similarity parameter
        z_by_L = monin_obukhov(ustar, wT, T, z=sonic_height)
        zL = np.nanmean(z_by_L)
        #
        # wind speed a 10 m - no neutral conditions
        kappa = 0.4
        Uspd = abs(U_cor[0] + 1j*U_cor[1])
        U10 = np.nanmean(Uspd) + (ustar / kappa) * np.log(10 / sonic_height)

        # average wind speed at neutral conditions
        U10N = nanmean(wind_speed_neutral(z_by_L, Uspd, ustar))
        # 
        # save data in the output dictionary
        list_of_variables = {
                "Ua"    : "eastward_wind_component",
                "Va"    : "northward_wind_component",
                "Ts"    : "sonic_air_temperature",
                "uw"    : "upward_eastward_momentum_flux_in_air",
                "vw"    : "upward_northward_momentum_flux_in_air",
                "wT"    : "upward_sensible_heat_flux",
                "ustar" : "airside_friction_velocity",
                "wstar" : "waterside_friction_velocity",
                "U10N"  : "10m_neutral_wind_speed",
                "zL"    : "monin_obukhov_stability_parameter"
                }
        #
        for k, v in list_of_variables.items():
            self.r[k] = eval(k)
        #
        # append to global list of variables
        self.list_of_variables = {**self.list_of_variables, **list_of_variables}
    # }}}

    # current data {{{
    def current_data(self):
        """Get data from the signature current profiler and the vector velocim."""

        # TODO: aqui el z-profile solo debe ser de 0 a ncel, es decir indices ya
        # que en realidad la presion va a cambiar con el tiempo, asi que hay qye
        # corregit eso

        # check for nans: if both are valid do nothing, else raise exception
        if (self.check_nans(self.sig) and self.check_nans(self.vec)):
            pass
        else:
            raise Exception("Number of nans is more than 30%.")

        # compute data from vector.
        # convert from mm/s to m/s
        v1 = simple_despike(self.vec["vel_b1"]) / 1000.
        v2 = simple_despike(self.vec["vel_b2"]) / 1000.
        v3 = simple_despike(self.vec["vel_b3"]) / 1000.

        # compute the data from the signature
        ncell = 10
        cell_size = nanmean(self.sig["cell_size"]/1000)  # <- mm to meters
        z_lower = -nanmean(self.sig["pressure"])/1000    # <- mbar to quasi-meters
        z_upper = z_lower + ncell*cell_size
        z_profile = np.linspace(z_lower, z_upper, ncell)
        #
        valid_beams = [b for b in self.sig.keys() if b.startswith("vel_b")]
        vel_b1 = np.nanmean(self.sig[valid_beams[0]] / 1000., axis=0)
        vel_b2 = np.nanmean(self.sig[valid_beams[1]] / 1000., axis=0)
        vel_b3 = np.nanmean(self.sig[valid_beams[2]] / 1000., axis=0)
        vel_b4 = np.nanmean(self.sig[valid_beams[3]] / 1000., axis=0)
        
        # save data in the output dictionary
        list_of_variables = {
                "v1" : "vector_beam1_velocity",
                "v2" : "vector_beam2_velocity",
                "v3" : "vector_beam3_velocity",
                "z_profile": "depth_profile",
                "vel_b1": "signature_beam1_velocity",
                "vel_b2": "signature_beam2_velocity",
                "vel_b3": "signature_beam3_velocity",
                "vel_b4": "signature_beam4_velocity",
                }
        #
        for k, v in list_of_variables.items():
            self.r[k] = eval(k)
        #
        # append to global list of variables
        self.list_of_variables = {**self.list_of_variables, **list_of_variables}
    # }}}

    # electronics data {{{
    def electronics_data(self):
        """Get data from the internal cilinder control variables."""

        # compute data from marvi
        # convert from mm/s to m/s
        Ti = simple_despike(self.mvi["temperature"])
        Pi = simple_despike(self.mvi["pressure"])
        Hi = simple_despike(self.mvi["humidity"])
        
        # save data in the output dictionary
        list_of_variables = {
                "Ti" : "internal_temperature",
                "Pi" : "internal_pressure",
                "Hi" : "internal_humidity"
                }
        #
        for k, v in list_of_variables.items():
            self.r[k] = eval(k)
        #
        # append to global list of variables
        self.list_of_variables = {**self.list_of_variables, **list_of_variables}
    # }}}

# }}}


if __name__ == "__main__":
    pass

# === end of file ===
