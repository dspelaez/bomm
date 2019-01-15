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
#
import wdm
#
import motion_correction as motcor
from utils import get_netcdf_data, nanmean, wavenumber


# generate dates {{{
def get_dates(number_of_minutes=30):
    start = dt.datetime(2017, 11, 17)
    final = dt.datetime(2018,  2, 1)
    while start < final:
        yield start
        start += dt.timedelta(minutes=number_of_minutes)
# }}}

# motion matrices {{{
def motion_matrices(ekx, sig, met, gps):
    """Matrices of the acceleromter, gyroscope and euler angles"""

    # heading signature
    heading_sig = (sig["heading"]/100) % 360
    
    # heading maximet
    true_wnd, rel_wnd = met["true_wind_dir"], met["relative_wind_dir"]
    heading_met = (true_wnd - rel_wnd + 60) % 360

    # the low frequency heading means the angle between new BOMM y-axis and
    # true north. Magnetic deviation is taken from GPS measurements. All in
    # degrees
    if np.isnan(heading_sig.filled(np.nan)).all():
        heading = heading_met + gps["mag_var"][0]
    else:
        heading = heading_sig + gps["mag_var"][0]

    # construct accelerometer and gyroscope tuples
    # apply a rotation to an ENU frame of reference
    R = (np.pi, 0, np.pi/2)
    Acc = motcor.vector_rotation((ekx["accel_x"], ekx["accel_y"], ekx["accel_z"]), R)
    Gyr = motcor.vector_rotation((ekx["gyro_x"],  ekx["gyro_y"],  ekx["gyro_z"]),  R)


    # integrate accel and gyro to obtain euler angles
    phi, theta = motcor.pitch_and_roll(*Acc, *Gyr, fs=100, fc=0.04)
    psi = motcor.yaw_from_magnetometer(ekx["gyro_z"], np.radians(heading),
            fs=100, fc=0.04)
    Eul = (phi, theta, (-psi)%(2*np.pi))

    return Acc, Gyr, Eul
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

    return Hs, Tp, pDir, mDir

# }}}

# compute stokes drift prodfile{{{
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


# correction of data {{{
def data_correction(vwnd, wav, ekx, sig, met, gps):
    """
    """

    # create out dictionary
    out = {}

    # get the motion matrices
    Acc, Gyr, Eul = motion_matrices(ekx, sig, met, gps)

    # --- correction of wavestaff data ---

    # determinte position of the wavestaffs
    xx, yy = wdm.reg_array(N=5, R=0.866, theta_0=180)

    # apply the correction to the surface elevation
    S = np.zeros((int(len(Z)/8+1), len(wav)-1))
    X, Y, Z = (np.zeros((len(wav["time"]), len(wav)-1)) for _ in range(3))
    for i, (x, y), in enumerate(zip(xx, yy)):
        #
        # get suface elevation at each point
        z = wav[f"ws{i+1}"] * 3.5/4095
        #
        # apply motion correction
        X[:,i], Y[:,i], Z[:,i] = motcor.position_correction((x,y,z),
                Acc, Eul, fs=20, fc=0.04, q=5)
        #
        # compute fourier spectrum
        ffrq, S[:,i] = signal.welch(Z[:,i], fs=20, 
                nperseg=len(Z)/4, noverlap=len(Z)/8)

    # compute directional wave spectrum
    # TODO: create a anti-aliasing filter to decimate the time series
    wfrq, dirs, E, D = wdm.fdir_spectrum(Z[::5,:], X[::5,:], Y[::5,:], fs=4,
            limit=np.pi, omin=-5, omax=1, nvoice=16, ws=(30, 1))
    
    # compute bulk wave parameters and stokes drift magnitude
    Hs, Tp, pDir, mDir = wave_parameters(wfrq, dirs, E)
    Us = stokes_drift(ffrq, S.mean(1), z=0.0)


    # save data in the output dictionary
    # out["ffrq"], out["S"] = ffrq, S
    # out["wfrq"], out["dirs"], out["E"], out["D"] = wfrq, dirs, E, D
    # out["Hs"], out["Tp"], out["pDir"], out["mDir"] = Hs, Tp, pDir, mDir
    # out["Us"] =  Us


    # --- correction of sonic anemometer data ---

    # apply the correction to the anemometer data
    # TODO: check if 30 degrees rotation was performed
    L = (0, 0, 13)
    U_unc = (wnd["u_wind"], wnd["v_wind"], wnd["w_wind"])
    U_rot = motcor.vector_rotation(U_unc, (0,0,-30), units="deg")
    U_cor = motcor.velocity_correction(U_rot, Acc, Eul, L, fs=100, fc=0.04)

    # compute momentum fluxes
    T = wnd["sonic_temp"] + 273.15 # <--- convert to Kelvin
    uw, vw, wT = eddy_correlation_flux(U_cor[0], U_cor[1], U_cor[2], T)
    
    # compute ustar, wstar and wind speed at neutral conditions
    rhoa = 1.25
    rhow = 1020
    dens_rel = rhoa / rhow
    ustar = (uw**2 + vw**2) ** 0.25
    wstar = np.sqrt(dens_rel) * ustar
    zL = monin_obukhov(ustar, wT, T, z=L[2])
    U10N = nanmean(wind_speed_neutral(zL, abs(U_proj+1j*V_proj), ustar))

    # save data in the output dictionary
    out["Ua"], out["Va"], out["Ts"] = nanmean(U_cor[0]), nanmean(U_cor[1]), nanmean(T)

    # compute the momentum fluxes
# }}}


class ProcessedData(object):

    """This class has methods to process the BOMM data at level 2."""

    def __init__(self):
        """TODO: to be defined1. """
        







def get_heading_data(date, number_of_minutes=30, figname=None):
    """
    """

    # filename
    path = "/Volumes/BOMM/cigom/data/bomm1_its/level1"
    filename = f"{path}/{date.strftime('%Y%m%d')}.nc"

    # open file
    with nc.Dataset(filename, "r") as data:

        # load data from netcdf
        ekx = get_netcdf_data(data["ekinox"],    date, number_of_minutes)
        wnd = get_netcdf_data(data["sonic"],     date, number_of_minutes)
        gps = get_netcdf_data(data["gps"],       date, number_of_minutes)
        mvi = get_netcdf_data(data["marvi"],     date, number_of_minutes)
        met = get_netcdf_data(data["maximet"],   date, number_of_minutes)
        pro = get_netcdf_data(data["proceanus"], date, number_of_minutes)
        rbr = get_netcdf_data(data["rbr"],       date, number_of_minutes)
        sig = get_netcdf_data(data["signature"], date, number_of_minutes)
        vec = get_netcdf_data(data["vector"],    date, number_of_minutes)
        wav = get_netcdf_data(data["wstaff"],    date, number_of_minutes)




if __name__ == "__main__":
    
    # plot each day of heading
    for date in get_dates(number_of_minutes=24*60):
        print(f"Plotting heading for {date}")
        figname = f"heading/{date.strftime('%Y%m%d')}.png"
        get_heading_data(date, number_of_minutes=24*60, figname=figname)















# write variables {{{
def write_variables(metadata, sensor, grp):
    """Create NetCDF variables and attributtes for the given grp."""

    # get variables info from the metadata
    variables = metadata["sensors"][sensor]["variables"]

    # create time variable
    nctime = grp.createVariable("time", "f8", "time", fill_value=False)
    nctime.setncattr("units", global_time_units)
    nctime.setncattr("calendar", "gregorian")

    # create each variable
    for k, v in variables.items():
        #
        # check if variable is 2d or 1d
        if isinstance(v["column"], list):
            ncell = v["column"][1] - v["column"][0] + 1
            try:
                #
                # create number of cell dimension
                grp.createDimension("cell", ncell)
            except:
                pass
            #
            # create two dimensional variable 
            var = grp.createVariable(k, "f8", ("time", "cell"), fill_value=np.nan)
        else:
            #
            # create one dimensional variable 
            var = grp.createVariable(k, "f8", "time", fill_value=np.nan)
        #
        # write variable attributes
        for attr, val in v.items():
            if attr not in ["column"]:
                var.setncattr(attr, val)
# }}}

# write group {{{
def write_group(b, dataset, sensor, day, logfile):
    """Write variables and attributes associated to each group."""

    # create group related with the sensor and associate with time
    grp = dataset.createGroup(sensor)

    # get sampling frequency and seconds per file
    N  = b._getsecperfile(sensor)
    fs = b._getsampfreq(sensor)
    
    # compute the number of samples per day
    samples_per_day = int(fs * 60 * 60 * 24)
    samples_per_file = int(fs * N)

    # create time dimension and assing attributes
    grp.createDimension("time", samples_per_day)
    
    # write global attributes for each sensor
    for attr, val in b.metadata["sensors"][sensor].items():
        if attr not in ["variables", "seconds_per_file"]:
            grp.setncattr(attr, val)

    # create variables
    write_variables(b.metadata, sensor, grp)

    # loop for each data
    date, end = day, day + dt.timedelta(days=1, seconds=-N)
    i, j = 0, samples_per_file
    #
    while date <= end:

        # progress bar
        progress = 100 - 100 * (end-date).total_seconds() / (3600 * 24)
        sys.stdout.write(f" {sensor:10s}: {progress:5.1f}%  ---> {date}\r")
        sys.stdout.flush()

        # load data
        dic = b.read(sensor, date, logfile=open(logfile, "a"))

        # assign data to netctd variable
        for name, value in dic.items():

            # write variables
            if name not in ["time"]:

                # check if variables is 2d or 1d array
                if value.ndim == 1:
                    grp[name][i:j] = value
                else:
                    grp[name][i:j,:] = value

            # time variable
            else:
                grp["time"][i:j] = nc.date2num(value, global_time_units)
        
        # update counter
        i, j = j, j + samples_per_file

        # update date
        date += dt.timedelta(seconds=N) 
    
    # new line in the progress bar
    sys.stdout.write("\n")

# }}}

# write netcdf {{{
def write_netcdf(metafile):

    """This function writes a NetCDF4 file from the BOMM data.

    The function was writen to convert the BOMM raw data to a distributable
    NetCDF4 format. The functions only require the metatadata in a YAML file.
    This will write a NetCDF4 file for each day as specified in the YAML file.
    Each NetCDF4 file contains one group per buoy sensor and each group contains
    the variables specified in the YAML file as well as the metadata.

    Args:
        metafile (str): Name of the metadata YAML file.
    """

    # create instance of the bomm.ReadRowData class
    b = ReadRawData(metafile)

    # starting and final days
    day = dt.datetime.strptime(b.metadata["t_ini"], "%Y-%m-%d")
    end = dt.datetime.strptime(b.metadata["t_fin"], "%Y-%m-%d")

    # restart logfile if exist
    logfile = os.path.splitext(metafile)[0] + ".log"
    with open(logfile, "w"):
        pass

    # global time units
    global global_time_units
    global_time_units = f"seconds since 1970-01-01 00:00:00"
    
    # loop for each day
    while day <= end:
        
        # netcdf filename
        fname = f"{b.basepath}/../level1/{day.strftime('%Y%m%d')}.nc"

        # name of the group associated with each day
        with nc.Dataset(fname, "w") as dataset:

            # write global attrs
            for gbl_name, gbl_values in b.metadata.items():
                if gbl_name not in ["name", "basepath", "sensors", "t_ini", "t_fin"]:
                    dataset.setncattr(gbl_name, gbl_values)

            # write data for each sensor
            print("=" * 57, end="\n")
            sensors = b.metadata["sensors"].keys()
            for sensor in sensors:
                #
                # compute current time
                now = time.time()
                #
                # perform heavy task
                write_group(b, dataset, sensor, day, logfile)
                #
                # print elapsed time
                etime = time.time() - now
                print("-" * 46 + f" {etime/60:.2f} mins", end="\n")

        # update date
        day += dt.timedelta(days=1) 
# }}}



if __name__ == "__main__":
    
    # execute the code if the valid arg is passed
    if len(sys.argv) == 2:
        try:
            metafile = sys.argv[1]
            write_netcdf(metafile)
        except:
            raise Exception("An error was occurred")
    else:
        raise ValueError("Invalid number of arguments")




# === end of file ===

