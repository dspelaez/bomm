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
import sys
import gsw
#
import wdm
#
import motion_correction as motcor



# global variables
# global varibles {{{
number_of_minutes = 30
basepath = "/Volumes/BOMM/cigom/data/bomm1_its/level1"
# }}}


# functions to compute importan variables
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

    return Hm0, Tp1, pDir, mDir

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
            v[nans] = 0
        else:
            raise Exception("More than 10% of invalid data")

    # align with max variability axis (average V = 0)
    theta = np.arctan2(np.nanmean(V), np.nanmean(U)) #<- from U to V counterclockwise
    U_stream =  U*np.cos(theta) + V*np.sin(theta)
    V_stream = -U*np.sin(theta) + V*np.cos(theta)

    # align with the flow to do mean W equals zero
    phi = np.arctan2(np.nanmean(W), np.nanmean(U_stream)) #<- from U to W counterclockwise
    U_proj =  U_stream*np.cos(phi) + W*np.sin(phi)
    V_proj =  V_stream.copy()
    W_proj = -U_stream*np.sin(phi) + W*np.cos(phi)

    def flux(x, y, method="cospectrum"):
        if method == "cospectrum":
            f, Cuw = signal.csd(x, y, fs=100, nperseg=len(x))
            ix = np.logical_and(f>1/25, f<5)
            return np.trapz(Cuw[ix].real, f[ix])
        elif method == "classic":
            return nanmean(x * y)
        else:
            raise ValueError("Method must be `classic` or `cospectrum`")

    # compute turbulent fluxes
    method = "classic"
    d = lambda x: x-nanmean(x)
    u, v, w, T = d(U_proj), d(V_proj), d(W_proj), d(T)
    uw, vw, wT = flux(u, w, method), flux(v, w, method), flux(w, T, method)
    

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
def janssen_source_inpt(frq, dirs, E, ustar, udir):
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
    integral = lambda X: np.trapz(np.trapz(X, frqs), np.radians(dirs))

    # funtion to recursevily estimate wind input
    def source_input_estimation(ze):

        # compute directional distribution
        dir2 = (dirs - udir) % 360
        cos = np.cos(np.radians(dirs-udir))
        cos[np.logical_and(dir2>90, dir2<270)] = 0.

        # compute Miles parameter
        ustar_along_wind = abs(ustar * cos)
        wave_age = c[None,:] / ustar_along_wind[:,None]
        mu = (g * ze / c[None,:]**2) * np.exp(kappa * wave_age)
        
        # compute beta paremter
        # ---> Masterbroek eq13
        beta = (1.2 / kappa**2) * mu * np.log(mu)**4
        beta[mu > 1] = 0.

        # compute source term
        factor = w[None,:] * dens_rel * beta * (ustar / c[None,:]) ** 2
        return factor * cos[:,None]**2 *  E

    # compute phase speed
    w = 2 * np.pi * frqs
    k = sp.wavenumber(frqs, d=120)
    c = w / k

    # compute roughness lenght iteratively
    tau = rhoa * ustar**2
    z0 = alpha_garret * ustar**2 / g

    # first estimation
    Sin = source_input_estimation(z0)
    tauw = rhow*g * integral(Sin/c[None,:])
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
    

def wind_speed_neutral(zL, U, ustar):
    """Compute U10N with stability function for mometum dut to Donelan 1990."""

    # find stable and unstable indices
    ix_stab = zL > 0
    ix_unst = zL < 0

    # preallocate array
    Psi = np.zeros_like(zL) * np.nan

    # for unstable conditions: zL<0
    Psi[ix_unst] = (1 + 15.2 * np.abs(zL[ix_unst]))**-0.25
    Psi[ix_stab] = 1 + 4.8*zL[ix_stab]

    # compute wind speed for neutral conditions
    kappa = 0.4
    UzN = U + (ustar / kappa) * Psi

    # compute wind at 10 meters above sea level
    U10N = UzN + (ustar / kappa) * np.log(10 / 6.5)

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
            return np.angle(nanmean(np.exp(1j*np.radians(x)))) * 180/np.pi
        else:
            return np.mean(x[~np.isnan(x)])
# }}}

# get data from netcdf {{{
def get_netcdf_data(grp, date, number_of_minutes=30, only=None):
    """Return the data corresponding to the netCDF group for a specific date.
    
    Args: TODO
    Returns:
        Dictionary containig all variables.
    
    """

    # start and final indices
    fs = grp.sampling_frequency
    if isinstance(fs, str):
        fs = eval(grp.sampling_frequency)

    N = int(fs * 24 * 3600) # number of samples in a day
    hour, minute = date.hour, date.minute

    # check number of minutes
    if number_of_minutes > N:
        raise ValueError(f"Number of minutes must be less than a day. Max={N}.")

    # start and final index
    i = int(fs*hour*3600 + fs*minute*60)
    j = i + int(fs*number_of_minutes*60)

    dic = {}
    dic["time"] = nc.num2date(grp["time"][i:j], grp["time"].units)
    if only:
        for k in only:
            dic[k] = grp[k][i:j]
    else:
        for k in grp.variables.keys():
            if k not in ["time"]:
                dic[k] = grp[k][i:j]

    return dic
    # }}}

# convert to coordinate {{{
def convert_to_coordinate(x):
    """Convert ungly number to latitude of longitude coordinates."""
     
    def function(x):
        a, b = str(x).split(".")
        deg, mn = int(a[:-2]), int(a[-2:]) + float("0." + b)
        return deg +  mn/60
    
    return np.array(list(map(function, x)))
# }}}


# main class {{{
class ProcessingData(object):

    """
    This class contains methods to process the level-1 netCDF4 files for the
    CICESE-BOMM (Oceanographic and Marine Meteorology Buoys).
    """

    _list_of_dictionaries = "ekx wnd gps mvi met pro rbr sig vec wav".split()
    __slots__ = "list_of_variables r Acc Gyr Eul".split() + _list_of_dictionaries

    # private methods {{{
    def __init__(self, date):
        """Function to initialize the class.

        Args:
            date (datetime): datetime object
        """

        # date and filename
        nm = number_of_minutes
        filename = f"{basepath}/{date.strftime('%Y%m%d')}.nc"

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

        # compute the motion matrices needed to the correction
        self.motion_matrices()
    # }}}

    # motion matrices {{{
    def motion_matrices(self):
        """Matrices of the acceleromter, gyroscope and euler angles"""

        # heading signature
        heading_sig = (self.sig["heading"]/100) % 360
        
        # heading maximet
        true_wnd, rel_wnd = self.met["true_wind_dir"], self.met["relative_wind_dir"]
        heading_met = (true_wnd - rel_wnd + 60) % 360

        # the low frequency heading means the angle between new BOMM y-axis and
        # true north. Magnetic deviation is taken from GPS measurements. All in
        # degrees
        if np.isnan(heading_sig.filled(np.nan)).all():
            heading = heading_met + self.gps["mag_var"][0]
        else:
            heading = heading_sig + self.gps["mag_var"][0]

        # construct accelerometer and gyroscope tuples
        # apply a rotation to an ENU frame of reference
        R = (np.pi, 0, np.pi/2)
        self.Acc = motcor.vector_rotation((self.ekx["accel_x"],
            self.ekx["accel_y"], self.ekx["accel_z"]), R)
        #
        self.Gyr = motcor.vector_rotation((self.ekx["gyro_x"],
            self.ekx["gyro_y"],  self.ekx["gyro_z"]),  R)


        # integrate accel and gyro to obtain euler angles
        phi, theta = motcor.pitch_and_roll(*self.Acc, *self.Gyr, fs=100, fc=0.04)
        psi = motcor.yaw_from_magnetometer(self.ekx["gyro_z"],
                np.radians(heading), fs=100, fc=0.04)
        self.Eul = (phi, theta, (-psi)%(2*np.pi))

        # finally save the mean euler angles in the results dictionary
        self.r["roll"] = simple_despike(self.Eul[0]*180/np.pi, isangle=True)
        self.r["pitch"] = simple_despike(self.Eul[1]*180/np.pi, isangle=True)
        self.r["yaw"] = simple_despike(self.Eul[2]*180/np.pi, isangle=True)
        #
        list_of_variables = {
                "roll"  : "average_roll_angle",
                "pitch" : "average_pitch_angle",
                "yaw"   : "average_yaw_angle"
                }
        self.list_of_variables = {**self.list_of_variables, **list_of_variables}

    # }}}


    # air data {{{
    def air_data(self):
        """Compute averages of the air-quality data. Includes location from GPS."""

        # TODO: It is strongly recommended to limit the range of the
        #       data in the netcdf valid_range attribute.
        # compute variables from Maximet
        twDir = simple_despike(self.met["true_wind_dir"], isangle=True)
        rwDir = simple_despike(self.met["relative_wind_dir"], isangle=True)
        Wspd = simple_despike(self.met["wind_speed"])
        Pa = simple_despike(self.met["atm_pressure"])
        Ta = simple_despike(self.met["air_temp"])
        rhum = simple_despike(self.met["rel_humidity"])
        DP = simple_despike(self.met["dew_point"])
        pr = simple_despike(self.met["total_rain"])
        ipr = simple_despike(self.met["rain_intensity"])
        
        # compute variables from Proceanus
        aCO2 = simple_despike(self.pro["air_co2"])
        wCO2 = simple_despike(self.pro["wat_co2"])
        ahum = simple_despike(self.pro["air_humidity"])
        rhoa = air_dens(Ta, rhum, Pa)

        # compute data from gps
        lat = simple_despike(self.gps["latitude"]) 
        lon = simple_despike(self.gps["longitude"]) 
        
        # save data in the output dictionary
        list_of_variables = {
                "twDir" : "true_wind_direction",
                "rwDir" : "relative_wind_direction",
                "Wspd"  : "wind_speed",
                "Pa"    : "air_pressure",
                "Ta"    : "air_temperature",
                "rhum"  : "relative_humidity",
                "DP"    : "dew_point",
                "pr"    : "total_rain",
                "ipr"   : "rain_intensity",
                "aCO2"  : "air_co2",
                "wCO2"  : "water_co2",
                "ahum"  : "air_humidity",
                "rhoa"  : "air_density",
                "lat"   : "latitude",
                "lon"   : "longitude"
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

        # check dimensions
        ntime, npoint = len(self.wav["time"]), len(self.wav)-1

        # determinte position of the wavestaffs
        xx, yy = wdm.reg_array(N=5, R=0.866, theta_0=180)

        # allocate variables
        S = np.zeros((int(ntime/20+1), npoint))
        X, Y, Z = (np.zeros((ntime, npoint)) for _ in range(3))
        #
        # apply the correction to the surface elevation and compute fourier spc
        for i, (x, y), in enumerate(zip(xx, yy)):
            #
            # get suface elevation at each point
            z = self.wav[f"ws{i+1}"] * 3.5/4095
            #
            # apply motion correction
            X[:,i], Y[:,i], Z[:,i] = motcor.position_correction((x,y,z),
                    self.Acc, self.Eul, fs=20, fc=0.04, q=5)
            #
            # compute fourier spectrum
            ffrq, S[:,i] = signal.welch(Z[:,i], fs=20, 
                    nperseg=int(ntime/10), noverlap=int(ntime/20))

        # compute directional wave spectrum
        # TODO: create a anti-aliasing filter to decimate the time series
        d = lambda x: x[::5,:]
        wfrq, dirs, E, D = wdm.fdir_spectrum(d(Z), d(X), d(Y), fs=4,
                limit=np.pi, omin=-5, omax=1, nvoice=16, ws=(30, 1))
        
        # compute bulk wave parameters and stokes drift magnitude
        Hm0, Tp, pDir, mDir = wave_parameters(wfrq, dirs, E)
        Us0 = stokes_drift(ffrq, S.mean(1), z=0.0)

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
                "Us0"   : "surface_stokes_drift"
                }
        #
        for k, v in list_of_variables.items():
            self.r[k] = eval(k)
        #
        # append to global list of variables
        self.list_of_variables = {**self.list_of_variables, **list_of_variables}
    # }}}

    # wind data {{{
    def wind_data(self, rotation_angle=30):
        """Compute momentum flux and atmospheric parameters."""

        # apply the correction to the anemometer data
        L = (0,0,13)
        U_unc = (self.wnd["u_wind"], self.wnd["v_wind"], self.wnd["w_wind"])
        U_rot = motcor.vector_rotation(U_unc, (0,0,rotation_angle), units="deg")
        U_cor = motcor.velocity_correction(U_rot, self.Acc, self.Eul, L, fs=100, fc=0.04)

        # compute momentum fluxes
        T = self.wnd["sonic_temp"] + 273.15 # <--- convert to Kelvin
        uw, vw, wT = eddy_correlation_flux(U_cor[0], U_cor[1], U_cor[2], T)
        
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
        zL = monin_obukhov(ustar, wT, T, z=L[2])
        #
        # average wind speed at neutral conditions
        U10N = nanmean(wind_speed_neutral(zL, abs(U_cor[0]+1j*U_cor[1]), ustar))
        # 
        # average wind speed from anemometer
        Ua, Va, Ta = nanmean(U_cor[0]), nanmean(U_cor[1]), nanmean(T) - 273.15

        # save data in the output dictionary
        list_of_variables = {
                "Ua"    : "zonal_wind_component",
                "Va"    : "meridional_wind_component",
                "Ta"    : "sonic_air_temperature",
                "uw"    : "zonal_momentum_flux",
                "vw"    : "meridional_momentum_flux",
                "wT"    : "sensible_heat_flux",
                "ustar" : "airside_friction_velocity",
                "wstar" : "waterside_friction_velocity",
                "U10N"  : "10m_wind_speed"
                }
        #
        # list_of_variables = "Ua Va Ta uw vw wT U10N"
        for k, v in list_of_variables.items():
            self.r[k] = eval(k)
        #
        # append to global list of variables
        self.list_of_variables = {**self.list_of_variables, **list_of_variables}
    # }}}
    
    # current data {{{
    def current_data(self):
        """Get data from the signature current profiler and the vector velocim."""

        # compute data from vector.
        # convert from mm/s to m/s
        v1 = simple_despike(self.vec["vel_b1"]) / 1000.
        v2 = simple_despike(self.vec["vel_b2"]) / 1000.
        v3 = simple_despike(self.vec["vel_b3"]) / 1000.

        # copute the data from the signature
        ncell = 10
        cell_size = nanmean(self.sig["cell_size"]/1000)  # <- mm to meters
        z_lower = -nanmean(self.sig["pressure"])/1000    # <- mbar to quasi-meters
        z_upper = z_lower + ncell*cell_size
        z_profile = np.linspace(z_lower, z_upper, ncell)
        #
        sig_beams = [b for b in self.sig.keys() if b.startswith("vel_b")]
        vel_b1 = np.nanmean(self.sig["vel_b1"] / 1000., axis=0)
        vel_b2 = np.nanmean(self.sig["vel_b2"] / 1000., axis=0)
        vel_b3 = np.nanmean(self.sig["vel_b3"] / 1000., axis=0)
        vel_b5 = np.nanmean(self.sig["vel_b5"] / 1000., axis=0)
        
        # save data in the output dictionary
        list_of_variables = {
                "v1" : "vector_beam1_velocity",
                "v2" : "vector_beam2_velocity",
                "v3" : "vector_beam3_velocity",
                "z_profile": "depth_profile",
                "vel_b1": "signature_beam1_velocity",
                "vel_b2": "signature_beam2_velocity",
                "vel_b3": "signature_beam3_velocity",
                "vel_b5": "signature_beam5_velocity",
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
        """Get data from the internal cilinder and the ekinox motion sensor."""

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

    date = dt.datetime(2017, 11, 17, 0, 0, 0)
    date = dt.datetime(2017, 11, 17, 1, 0, 0)
    self = ProcessingData(date)
    self.air_data()
    self.water_data()
    self.wave_data()
    self.wind_data()
    self.current_data()
    self.electronics_data()


# === end of file ===
