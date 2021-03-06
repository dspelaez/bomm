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
This module contains functions to apply the correction of the
data due to the BOMM inertial motion.
"""


import numpy as np
import scipy.signal as signal
import pandas as pd


# --- helper functions ---
# butterworth lowpass and highpass filter {{{
def butterworth_filter(data, fs=100, fc=None, order=5, kind="low"):
    """Butterworth low- and hihg-pass filter implementation."""

    if fc is not None:
        b, a = signal.butter(order, fc/(0.5*fs), btype=kind, analog=False)
        data_filtered = signal.filtfilt(b, a, data)
        if hasattr(data, "mask"):
            return np.ma.masked_array(data_filtered, mask=data.mask)
        else:
            return data_filtered
    else:
        return data
# }}}

# detrend with no nans {{{
def detrend(y, degree=1):
    """Nice detrend function to handle with NaNs."""
    
    # remove nans
    if hasattr(y, "mask"):
        if len(np.nonzero(y.mask)[0]) == len(y):
            raise Exception("Array is full of nans")
    else:
        ixnan = np.isnan(y)
        y[ixnan] = np.nanmean(y)

    # fit a polinomial
    x = np.linspace(0, len(y), len(y))
    p = np.polyfit(x, y, degree)

    # remove trend
    return y - np.polyval(p, x)
# }}}

# resample data to the same sampling frequency {{{
def resample(x, y):
    """Interpolates `y` data into `x` size and returns `y_new`
    
    This function uses pandas for an accurate resample. The `x` data is the
    fast signal and the `y` data is the slow signal. This function interpolates
    `y` data into the `x` data.
 
    """

    x_time = np.linspace(0, 100, len(x))
    y_time = np.linspace(0, 100, len(y))

    s = pd.Series(y, index=y_time)

    # check the number of nans
    n = len(y)
    n_nans = s.isnull().sum().max()
    if n_nans / n > 0.1:
        raise Exception(f"Number of NaNs is {n_nans:d} out of {n:d}.")

    # drop missing values only if are less than 10 percent of the data
    # TODO: not more than 100 consecutive missing values
    s = s.dropna(how="any")

    # remove duplicate indices if they exist
    s = s[~s.index.duplicated(keep='first')]

    # sort data in ascending and reindex to the new time
    # i still dont know what is the difference between ffill/bfill
    s = s.sort_index().reindex(x_time, limit=1, method="bfill").ffill()

    # crate new dictionary for output
    if hasattr(x, "mask"):
        return np.ma.masked_array(s.values, mask=x.mask)
    else:
        return s.values
# }}}

# complementery filter in using a digital filter {{{
def complementary_filter(signal_a, signal_b, fs, fm, filter_order=2):
    """"Returns a merge between two angles signal_a and signal_b in radians."""
    
    # merge the signals using the merge frequency fm
    s = butterworth_filter(np.exp(1j*signal_a), fs, fm, filter_order, kind="low") + \
        butterworth_filter(np.exp(1j*signal_b), fs, fm, filter_order, kind="high")

    return np.angle(s)

# }}}

# integration in the frequency domain {{{
def fft_integration(data, fs, fc=None, order=-1):
    """Intergration of a signal in the frequency domain using FFT.
    
    This function implements the integration in the time domain by means of the
    Fast Fourier Transform. It also performs a band pass filter, removing all
    the unwanted frequencies.

    Args:
        signal (array): Numpy 1d-array with the data.
        fs (float or int): Sampling frequency.
        fc (float or tuple): This is the cut-off frequency. If fc is floating or
            integer a lowpass filter is performed. If fc is a list a band
            pass-pass filter is made between the two given frequencies.
        order (integer): Indicates the order of the integration. Negative number
            indicates integration while positive indicates differentiontion (not
            implemented yet).

    Return (array): Signal integrated.
    """

    # check for nans if more than 10 percents
    nans = np.isnan(data)
    if len(nans.nonzero()[0]) / len(data) < 0.1:
        data[nans] = np.nanmean(data[~nans])
    else:
        return data * np.nan
    
    # if order == 0 do nothing
    if order == 0:
        return data

    # if order > 0 raise an error
    if order > 0:
        raise ValueError("Order must be a negative integer")

    # get frequency array
    N = len(data)
    freqs = np.fft.fftfreq(N, 1/fs)
    
    # the first element of freqs array is zero, so we have
    # to discard it to avoid division by zero
    # the factor of integration is iw
    factor = 1j*2*np.pi*freqs[1:]
    
    # compute fft of the signal for the non-zero frequencies
    # and apply integration factor
    fft = np.zeros(len(freqs), 'complex')
    fft[1:] = np.fft.fft(data)[1:] * factor**order
    
    # high pass filter
    if fc is None:
        return np.fft.ifft(fft).real
    #
    elif isinstance(fc, float) or isinstance(fc, int):
        #
        if False: # <-- ideal filter just in case 
            ix = abs(freqs) <= fc
            fft[ix] = 0.
        #
        else: # <- freqs response of a 3th order butterworth filter
            b, a = signal.butter(3, fc, 'high', analog=True)
            w, h = signal.freqs(b, a, worN=freqs)
            fft = fft * abs(h)

    return np.fft.ifft(fft).real
# --- }}}

# rotation matrix in three dimensions {{{
def vector_rotation(U, E, units="rad"):
    """Apply a three dimensional rotation of the vector U given the angles T.
    
    Args:
        U (tuple): Components of the vector U = (u, v, w)
        E (tuple): Rotation angles corresponding to roll (x), pitch (y) and
            yaw (z), respectively. E = (phi, theta, psi)
        units (str): Flag to choose input angles between degrees or radians.

    Return (float): Components of the rotated vector.
    """

    # unpack tuples
    u, v, w = U
    phi, theta, psi = E

    # if input angles are in degrees, convert it to radians
    if units == "deg":
        phi, theta, psi = np.radians(phi), np.radians(theta), np.radians(psi)
    elif units == "rad":
        pass
    else:
        raise ValueError("units must be either def or rad")

    # TODO: check validity of angles
    # phi and psi must be between -pi and pi
    # theta musr be between -pi/2 and  pi/2

    # compute sins and cosines
    c_phi, c_theta, c_psi = np.cos(phi), np.cos(theta), np.cos(psi)
    s_phi, s_theta, s_psi = np.sin(phi), np.sin(theta), np.sin(psi)

    # apply the rotation of each components
    #
    # first component
    u_rot = (c_theta*c_psi)                     * u + \
            (s_phi*s_theta*c_psi - c_phi*s_psi) * v + \
            (c_phi*s_theta*c_psi + s_phi*s_psi) * w

    # second component
    v_rot = (c_theta*s_psi)                     * u + \
            (s_phi*s_theta*s_psi + c_phi*c_psi) * v + \
            (c_phi*s_theta*s_psi - s_phi*c_psi) * w

    # third component
    w_rot = (-s_theta)      * u + \
            (s_phi*c_theta) * v + \
            (c_phi*c_theta) * w

    return u_rot, v_rot, w_rot


# }}}

# get an array of a regular distribution of wavestaffs {{{
def wavestaff_coordinates(N, R=0.866, theta_0=-270):
    """This function returns position a regular array

    Function to get the coordinates (x, y) of and an array of wavestaffs formed by
    an regular N-vertices figured centered in (0,0) increasing counterclockwise
    Args:
        N (float): Number of vertices
        R (float): Separation
        theta_0 (float): Starting angle. 270 for the Ekinox reference frame.
    
    Returns:
        x (1d-array): x-coordinates of the array
        y (1d-array): y-coordinates of the array
    """

    theta = np.arange(0, 360, 360/N) - theta_0
    x, y  = [0], [0]
    x = np.append(x, R * np.cos(theta * np.pi / 180.))
    y = np.append(y, R * np.sin(theta * np.pi / 180.))

    return x, y
# }}}


# --- quaternions ---
# euler_to_quaternions {{{
def euler_to_quaternions(T):
    """Returns the quaterions associatted with the angles T=(phi,theta,psi)."""

    # unpack tuples
    phi, theta, psi = T

    # compute sins and cosines
    c_phi, c_theta, c_psi = np.cos(phi/2), np.cos(theta/2), np.cos(psi/2)
    s_phi, s_theta, s_psi = np.sin(phi/2), np.sin(theta/2), np.sin(psi/2)

    # compute quaternions components
    # q0 = 0.5*np.sqrt(1 + c_theta*s_psi + s_phi*s_theta*s_psi + \
            # c_phi*c_psi + c_phi*c_theta)
    # #
    # q1 = (s_phi*c_theta - c_phi*s_theta*s_psi + s_phi*c_psi) / (4*q0)
    # #
    # q2 = (c_phi*s_theta*c_psi + s_phi*s_psi + s_theta) / (4*q0)
    # #
    # q3 = (c_theta*s_psi - s_phi*s_theta*c_psi + c_phi*s_psi) / (4*q0)

    q0 = c_phi * c_theta * c_psi + s_phi * s_theta * s_psi
    q1 = s_phi * c_theta * c_psi - c_phi * s_theta * s_psi
    q2 = c_phi * s_theta * c_psi + s_phi * c_theta * s_psi
    q3 = c_phi * c_theta * s_psi - s_phi * s_theta * c_psi

    return (q0, q1, q2, q3)

# }}}

# quaternion_product {{{
def quaternion_product(q, p):
    """Returns the quaterion porduct o p=(p0,p1,p2,p3) and q=(q0,q1,q2,q3)"""

    # unpack tuples
    q0, q1, q2, q3 = q
    p0, p1, p2, p3 = p

    r0 = p0*q0 - p1*q1 - p2*q2 - p3*q3
    r1 = p0*q1 + p1*q0 + p2*q3 - p3*q2
    r2 = p0*q2 - p1*q3 + p2*q0 + p3*q1
    r3 = p0*q3 + p1*q2 - p2*q1 + p3*q0

    return (r0, r1, r2, r3)

# }}}

# quaternion_conjugate {{{
def quaternion_conjugate(q):
    """Returns the quaterion conjugate"""
    q0, q1, q2, q3 = q
    return (q0, -q1, -q2, -q3)
# }}}

# quaternion_inverse {{{
def quaternion_inverse(q):
    """Returns the quaterion conjugate"""

    norm = quaternion_norm(q)
    q_conj = quaternion_conjugate(q)

    return (q_conj[0]/norm, q_conj[1]/norm, q_conj[2]/norm, q_conj[3]/norm)
# }}}

# quaternion_norm {{{
def quaternion_norm(q):
    """Returns the quaterion conjugate"""
    q0, q1, q2, q3 = q
    return q0**2 + q1**2 + q2**2 + q3**2
# }}}

# quaternion_theta {{{
def quaternion_theta(q):
    """Returns the angle for the axis angle representation"""
    q0, q1, q2, q3 = q
    return 2 * np.arctan2(np.sqrt( q1**2 + q2**2 + q3**2), q0)
# }}}

# quaternion_rotation {{{
def quaternion_rotation(v, q, mode="frame"):
    """Returns the vector v rotated into the quaternion q"""

    p = (0, v[0], v[1], v[2])
    qinv = quaternion_inverse(q)
    if mode=="frame":
        r = quaternion_product(q, quaternion_product(p, qinv)) #qpq*
    elif mode=="point":
        r = quaternion_product(qinv, quaternion_product(p, q)) #q*pq
    else:
        raise Exception("Mode must be `frame` or `point`")
    
    return r[1:]
# }}}


# --- compute yaw, pitch and roll ---
# compute tilt from accelerations {{{
def tilt_from_accerometer(ax, ay, az):
    """Compute the inclination from the acceleromter signal as complex number."""

    # using the SBG rotation matrix and arctan
    phi = np.arctan2(ay, np.sqrt(ax**2 + az**2))
    theta = np.arctan2(-ax, np.sqrt(ay**2 + az**2))
    # alternatives with some slightlty differeces
    # phi = np.arctan(ay / az)
    # theta = np.arctan(-ax / (ay*np.sin(phi) + az*np.cos(phi)))
    # phi = np.arctan2(ay, az)
    # theta = np.arctan2(-ax, az)

    return phi, theta

# }}}

# compute pitch and roll using a complementary_filter {{{
def pitch_and_roll(ax, ay, az, wx, wy, wz, fs=100, fc=0.05, fm=1):
    """Euler angles using a complementary filter in frequency domain."""

    # compute pitch and roll from acclerometers
    phi_acc, theta_acc = tilt_from_accerometer(ax, ay, az)

    # compute pitch and roll from gyrospcope
    get_angle = lambda x: fft_integration(x, fs=fs, fc=fc, order=-1)
    phi_gyr, theta_gyr = get_angle(wx), get_angle(wy)

    # complementary filter
    phi = complementary_filter(phi_acc, phi_gyr, fs, fm=fm)
    theta = complementary_filter(theta_acc, theta_gyr, fs, fm=fm)

    # return data
    return phi, theta
# }}}

# orientation from magnetic north {{{
def yaw_from_magnetometer(wz, heading, fs=100, fc=0.05, fm=1):
    """Returns the yaw angle measured clockwise from north.
    
    This function computes the yaw angle which is equivalent to the buoy
    orientation. The function requires the data from the accelerometer and the
    magnetometer heading. Data are interpolated to the maximum sampling
    frquency. Uses a complementary digital filter to merge the high frequency
    gyroscope data with the low frequency magnetometer.

    Args:
        wz (float): Angular rate of change in rad/s.
        heading (float): Heading angle from magnetometer or signature in rad.
            Fortunately the heading usually follows the nautical convention,
            this means that the angle is measured clockwise which is consistent
            withe the Ekinox convention for the angles (Tait-Bryan or North,
            Easth, Down).

    Returns (float): Orientation respect to magnetic north.
    """

    # interpolate to the `wz` sampling frquency
    heading_fast = resample(wz, np.cos(heading)) + \
        1j* resample(wz, np.sin(heading))

    # compute psi angle from the magnetometre
    psi_mag = np.mod(np.ma.angle(heading_fast), 2*np.pi)
    mean_psi = np.nanmean(psi_mag)

    # compute pitch and roll from gyrospcope
    psi_gyr = fft_integration(wz, fs=fs, fc=fc, order=-1)

    # complementary filter
    psi = complementary_filter(psi_mag-mean_psi, psi_gyr, fs, fm)

    return np.mod(psi + mean_psi, 2*np.pi)
# }}}


# --- compute position and velocities ---
# position correction {{{
def position_correction(X, A, E, fs=20, fc=0.05, q=5, full=False):
    """Correcion of the position and the surface elevation.

    This function applies the correction of the surface elevation measured by
    the wavestaffs due to the buoy inertial motion. The correction is perform
    not only to the surface elevation (z coordinate) but also for the position
    in the x-y-plane. So, the input/output is the uncorrected/corrected
    time-varying position vector of the water surface.

    The equation to perfom such correction is given by
    
                              //                 /
      x_E =   R x_B    +    R || a_B dt   +    R | curl { Om_B, x_B } dt  
                              //                 /
              ---v---    -------v-------      ------------v--------------
               x_obs          x_acc                     x_rot

    where R is a rotation matrix.

           |  cosp*cosy  sinr*sinp*cosy-cosr*siny  cosr*sinp*cosy+sinr*siny |
       R = |  cosp*siny  sinr*sinp*siny+cosr*cosy  cosr*sinp*siny-sinr*cosy |
           |   -sinp            sinr*cosp               cosr*cosp           |

    in which, `p` is pitch (theta), `r` is roll (phi) and `y` is yaw (psi)

    In the same way, the matrix of angular rate of changes is given by:

              | - dpdt siny + drdt cosp cosy |
      Omega = |   dpdt cosy + drdt cosp siny |  
              |      dydt   - drdt sinp      |          

    Args:
        X (tuple): Contains the elements of X=(x,y,z). Each component is a time
            series given in a numpy 1d array.
        A (tuple): Contains the time series of the accelerations A=(ax, ay, az).
        E (dict): Contains the time series of the euler angles
             given by E=(roll, pitch, yaw).
        fs (float): Sampling frequency of the time series.
        fc (float): Cut-ff frequency for the integration. Could be a tuple.
        q (int): Factor to decimate the accelerometer data into wavestaff data.
            This value must be 5 since the ekinox frequency is 5 times the
            wavestaff frequency. When wavestaff measured at 10 Hz q must be 10.
        full (bool): Return full values or only the corrected ones. Default False.

    Returns: Tuple with the X tuple corrected. 

    Note:
        Arrays must be clean before attempting to apply the correction.

    References:
        * Anctil Donelan Drennan Graber 1994, JAOT 11, 1144-1150
        * Drennan Donelan Madsen Katsaros Terray Flagg 1994, JAOT 11, 1109-1116
    """

    # substract gravity acceleration effect
    G = vector_rotation((0,0,-9.8), E)

    # apply double integration in the frequency domain
    P = tuple(fft_integration(a, fs*q, fc=fc, order=-2) for a,g in zip(A,G))
    
    # compute the derivative of the euler angles
    D = tuple(np.gradient(e, 1/(fs*q)) for e in E)

    # decimate the high frequency signals to the given frecuency
    # note than in this function i use the following equivalences
    #   roll  --> r --> phi   --> E[0]
    #   pitch --> p --> theta --> E[1]
    #   yaw   --> y --> psi   --> E[2]
    decimate = lambda x, q: x[::q]
    E_down = tuple(decimate(e, q) for e in E) # <- Euler
    P_down = tuple(decimate(p, q) for p in P) # <- Position
    D_down = tuple(decimate(d, q) for d in D) # <- Derivative

    # compute sines and cosines
    roll, pitch, yaw = E_down

    # convert observations into the inertial frame
    x_obs, y_obs, z_obs = vector_rotation(X, (roll, pitch, yaw))
    
    # correction due to translations
    x_acc, y_acc, z_acc = vector_rotation(P_down, (roll, pitch, yaw))

    # correction due to rotation
    xB, yB, zB = X
    droll, dpitch, dyaw = D_down
    curl = (fft_integration( dpitch*xB -   dyaw*yB, fs, fc, order=-1),
            fft_integration(-dpitch*zB +   dyaw*xB, fs, fc, order=-1),
            fft_integration(  droll*yB - dpitch*xB, fs, fc, order=-1))
    x_rot, y_rot, z_rot = vector_rotation(curl, (roll, pitch, yaw))

    # compute earth-based water position
    xE = x_obs + x_acc + x_rot * 0
    yE = y_obs + y_acc + y_rot * 0
    zE = z_obs + z_acc + z_rot * 0

    # return data
    if full:
        return (x_obs,y_obs,z_obs), (x_acc,y_acc,z_acc), (x_rot,y_rot,z_rot)
    else:
        return xE, yE, zE

# --- }}}

# velocity correction {{{
def velocity_correction(U, A, E, L=(0,0,0), fs=100, fc=0.05, full=False):
    """Correcion of the position and the surface elevation.

    This function applies the correction of the wind speed measured by the sonic
    anemometer due to the buoy inertial motion. 

    The equation to perfom such correction is given by
    
                              /                 
      u_E =   R u_B    +    R | a_B dt   +  R curl { Om_B, L } dt  
                              /                
             ---v---       -----v------     ----------v----------
              u_obs           u_acc                 u_rot

    where R is a rotation matrix.

           |  cosp*cosy  sinr*sinp*cosy-cosr*siny  cosr*sinp*cosy+sinr*siny |
       R = |  cosp*siny  sinr*sinp*siny+cosr*cosy  cosr*sinp*siny-sinr*cosy |
           |   -sinp            sinr*cosp               cosr*cosp           |

    in which, `p` is pitch (theta), `r` is roll (phi) and `y` is yaw (psi)

    In the same way, the matrix of angular rate of changes is given by:

              | - dpdt siny + drdt cosp cosy |
      Omega = |   dpdt cosy + drdt cosp siny |  
              |      dydt   - drdt sinp      |          

    Args:
        U (tuple): Contains the elements of U=(u,v,w). Each component is a time
            series given in a numpy 1d array.
        A (tuple): Contains the time series of the accelerations A=(ax, ay, az).
        E (dict): Contains the time series of the euler angles
             given by E=(roll, pitch, yaw).
        L (tuple): Coordinates of the anemometer respect to IMU.
        fs (float): Sampling frequency of the time series.
        fc (float): Cut-off frequency for the integration. Could be a tuple.
        full (bool): Return full values or only the corrected ones. Default False.

    Returns: Tuple with the X tuple corrected. 

    Note:
        Arrays must be clean before attempting to apply the correction.

    References:
        * Anctil Donelan Drennan Graber 1994, JAOT 11, 1144-1150
        * Drennan Donelan Madsen Katsaros Terray Flagg 1994, JAOT 11, 1109-1116
    """

    # substract gravity acceleration effect
    G = vector_rotation((0,0,-9.8), E)
    
    # apply integration in the frequency domain and 
    # compute the derivative of the euler angles
    V = tuple(fft_integration(a, fs, fc, order=-1) for a,g in zip(A,G))
    D = tuple(np.gradient(e, 1/fs) for e in E)

    # compute sines and cosines
    # note than in this function i use the following equivalences
    #   roll  --> r --> phi   --> E[0]
    #   pitch --> p --> theta --> E[1]
    #   yaw   --> y --> psi   --> E[2]
    roll, pitch, yaw = E

    # convert observations into the inertial frame
    u_obs, v_obs, w_obs = vector_rotation(U, (roll, pitch, yaw))
    
    # correction due to translations
    u_acc, v_acc, w_acc = vector_rotation(V, (roll, pitch, yaw))

    # correction due to rotation
    Lx, Ly, Lz = L
    droll, dpitch, dyaw = D
    curl = (fft_integration( dpitch*Lz -   dyaw*Ly, fs, fc=0, order=-1),
            fft_integration(-dpitch*Lz +   dyaw*Lx, fs, fc=0, order=-1),
            fft_integration(  droll*Ly - dpitch*Lx, fs, fc=0, order=-1))
    u_rot, v_rot, w_rot = vector_rotation(curl, (roll, pitch, yaw))

    # compute earth-based water position
    uE = u_obs + u_acc + u_rot
    vE = v_obs + v_acc + v_rot
    wE = w_obs + w_acc + w_rot
    
    if full:
        return (u_obs,v_obs,w_obs), (u_acc,v_acc,w_acc), (u_rot,v_rot,w_rot)
    else:
        return uE, vE, wE

# --- }}}


# --- run as a script ---
if __name__ == "__main__":
    pass


# --- end of file ---
