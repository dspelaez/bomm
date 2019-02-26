#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2019 Daniel Santiago <dpelaez@cicese.edu.mx>
#
# Distributed under terms of the GNU/GPL license.

"""
This script contains functions that perform a check for the proccesed data.
"""

# import libs
import numpy as np
import datetime as dt
import matplotlib.pyplot as plt
import matplotlib.animation as animation
#
import wdm
import motion_correction as motcor
from processing_data import ProcessingData
from processing_data import eddy_correlation_flux
from wdm.spectra import polar_spectrum
#
plt.ion()


# function to load data {{{
def load_data(p):
    """Return essential data"""
    
    # directional spectra
    wfrq, dirs, E = p.r["wfrq"], p.r["dirs"], p.r["E"]

    # sonic anemometer
    Ua, Va = p.r["Ua"], p.r["Va"]

    # yaw and maximet data
    yaw = p.r['heading']
    Wspd, tWdir = p.r["U10N"], (270 - p.r["tWdir"]) % 360

    # do some conversions
    Uy, Vy = np.cos(yaw*np.pi/180), np.sin(yaw*np.pi/180)
    Um, Vm = (Wspd * np.cos(tWdir * np.pi/180),
              Wspd * np.sin(tWdir * np.pi/180))

    return wfrq, dirs, E, Ua, Va, yaw, Wspd, tWdir, Uy, Vy, Um, Vm
# }}}

# plot wave spectra {{{
def plot_wave_spectra(metafile, date):
    """This function plot nine sucesive wave spectra for the given date."""

    # create instance of ProcessingData class
    p = ProcessingData(metafile)

    # create canvas
    fig, ax = plt.subplots(3, 3, figsize=(9,9))
    fig.subplots_adjust(top=.95, bottom=.05, left=.05, right=.95, wspace=.1, hspace=.1)
    ax = np.ravel(ax)

    # for each data
    for i in range(9):
        
        invalid_data = True
        while invalid_data:
            try:
                date += dt.timedelta(minutes=10)
                p.run(date)
                wfrq, dirs, E, Ua, Va, yaw, Wspd, tWdir, Uy, Vy, Um, Vm = load_data(p)
                print(f"Data from {date} plotted")
                invalid_data = False
            except:
                print(f"Data from {date} not avaliable")
                invalid_data = True

        # plot spectrum
        if i in [2,5,8]:
            polar_spectrum(wfrq, dirs, E, label=True,
                    smin=-2., smax=2., fmax=0.5, ax=ax[i], cbar=True)
        else:
            polar_spectrum(wfrq, dirs, E, label=True,
                    smin=-2., smax=2., fmax=0.5, ax=ax[i], cbar=False)
        # 
        # compute components of each angle
        #
        # plot arrows
        ax[i].quiver(0, 0, Ua, Va, scale=20, color="r")
        ax[i].quiver(0, 0, Um, Vm, scale=20, color="b")
        ax[i].quiver(0, 0, Uy, Vy, scale=1, color="y")

        # plot wind-sea / swell delimiter
        wdirs = np.radians((dirs-tWdir))
        fcut = 0.83 * 9.8 / (2 * np.pi * Wspd * np.cos(wdirs)**1)
        # fcut[abs(wdirs)>=np.pi/3] = np.nan
        fcutx, fcuty = (fcut * np.cos(dirs*np.pi/180),
                        fcut * np.sin(dirs*np.pi/180))
        ax[i].plot(fcutx, fcuty, lw=0.5, ls="--", color="0.5")

        # set wind label
        ulabel = f"$u_{{10}} = {Wspd:.2f}$ m/s\n" + \
                f"$\\theta_{{u}} = {tWdir:.2f}^\circ$"
        ax[i].text(0.01, 0.98, ulabel, transform=ax[i].transAxes, ha="left", va="top")
        
        # set title
        title = p.r["time"].strftime("%Y-%m-%d %H:%M:%S")
        ax[i].set_title(title)
        
        # tune up
        if i not in [6,7,8]:
            ax[i].set_xlabel("")
            ax[i].set_xticklabels([''])
        #
        if i not in [0,3,6]:
            ax[i].set_ylabel("")
            ax[i].set_yticklabels([''])
    
    # return figure objects
    return fig, ax
# }}}

# yaw_angle {{{
def yaw_angle(p):
    """Comparison of the yaw angle from magnetometer and signature"""
    
    # get maximet angle
    met_angle = p.metadata["sensors"]["maximet"]["maximet_angle"]

    # create canvas
    fig, ax =  plt.subplots(1, figsize=(6,3))

    # yaw from different sources
    fac = 180/np.pi
    yaw_com = p.Eul[2] * fac
    yaw_sig = (90-(p.sig["heading"]/100 + 0*p.gps["mag_var"][0])) % 360
    #
    yaw_met = (p.met["true_wind_dir"] - p.met["relative_wind_dir"] + met_angle)
    yaw_met = (90-(yaw_met + 0*p.gps["mag_var"][0])) % 360
    #
    yaw_gyr = motcor.fft_integration(p.Gyr[2], fs=100, fc=1/25, order=-1) * fac
    yaw_gyr = (yaw_gyr + np.nanmean(yaw_com)) % 360

    # plot
    s = lambda t: (t - t[0]) / 60.
    ax.plot(s(p.ekx["time"]), yaw_com, color="darkred", label="Comp. filter", lw=2)
    ax.plot(s(p.ekx["time"]), yaw_gyr, color="0.75", label="Gyroscope")
    ax.plot(s(p.sig["time"]), yaw_sig, color="indigo", label="Signature")
    ax.plot(s(p.met["time"]), yaw_met, color="orange", label="Maximet")
    
    # tuneup
    fig.legend(loc="upper center", ncol=4)
    ax.set_xlabel(f"Minutes of {p.r['time'].strftime('%Y-%m-%d %H:%M')}")
    ax.set_ylabel("$\psi [^\circ]$")

    return fig, ax

# }}}

# cutoff effect on turbulence statistic {{{
def cutoff_effect(p):
    """Sensitivity analysis of the cut-off effect on turbulence statistics."""

    # get ekinox data
    Acc, Gyr, Eul = p.Acc, p.Gyr, p.Eul
    
    # sonic position
    L = (0, 0, 13.01)
    sonic_angle = p.metadata["sensors"]["sonic"]["sonic_angle"]
    sonic_height = p.metadata["sensors"]["sonic"]["sonic_height"]

    # loop for each cutoff
    stats = {k:[] for k in ["varu", "varv", "varw", "couw", "covw"]}
    cutoffs = 1./ np.arange(.1, 60, .5)
    for fc in cutoffs:
        #
        U_unc = (p.wnd["u_wind"], p.wnd["v_wind"], p.wnd["w_wind"])
        U_cor = motcor.velocity_correction(U_unc, Acc, Eul, L, fs=100, fc=fc)
        #
        u, v, w = U_cor[0], U_cor[1], U_cor[2]
        T = p.wnd["sonic_temp"] + 273.15 # <--- convert to Kelvin
        #
        uw, vw, wT = eddy_correlation_flux(u, v, w, T)
        #
        stats["varu"].append(np.var(u))
        stats["varv"].append(np.var(v))
        stats["varw"].append(np.var(w))
        stats["couw"].append(np.cov(u,w)[1][0])
        stats["covw"].append(np.cov(v,w)[1][0])


    # plot results
    fig, ax = plt.subplots(1, figsize=(6,3.5))
    ax.plot(1./cutoffs, stats["varu"]/stats["varu"][0])
    ax.plot(1./cutoffs, stats["varv"]/stats["varv"][0])
    ax.plot(1./cutoffs, stats["varw"]/stats["varw"][0])
    ax.plot(1./cutoffs, stats["couw"]/stats["couw"][0])
    ax.plot(1./cutoffs, stats["covw"]/stats["covw"][0])
    ax.set_ylim((0,1.6))
# }}}

# animation of motion correction {{{
def bomm_animation(p):
    """This function perform an animation of the moving wavestaffs array."""

    # load wavestaff position after correction
    xx, yy = wdm.reg_array(N=5, R=0.866, theta_0=180)
    #
    # get the sampling frequency and the resampling factor
    fs = p.metadata["sensors"]["wstaff"]["sampling_frequency"]
    q = int(100/fs)
    #
    # apply the correction to the surface elevation and compute fourier sp
    ntime, npoint = len(p.wav["time"]), 6
    X, Y, Z = (np.zeros((ntime, npoint)) for _ in range(3))
    for i, (x, y), in enumerate(zip(xx, yy)):
        z = p.wav[f"ws{i+1}"] * 3.5/4095 + 4.45
        X[:,i], Y[:,i], Z[:,i] = motcor.position_correction((x,y,z),
                p.Acc, p.Eul, fs=fs, fc=0.04, q=q)

    # load components of the bomm orientation
    wfrq, dirs, E, Ua, Va, yaw, Wspd, tWdir, xUy, xVy, Um, Vm = load_data(p)

    # load time varying yaw
    Uy, Vy = np.cos(p.Eul[2]), np.sin(p.Eul[2])

    # time series
    t = p.wav["time"] - p.wav["time"][0]
    z = Z[:,0] - np.nanmean(Z[:,0])

    # start animnation
    def init():
        point.set_data(t[0], z[0])
        line0.set_data(t[0:1000], z[:1000])
        line1.set_data(X[0,:], Y[0,:])
        line2.set_UVC(Uy[0], Vy[0])
        line2.set_offsets([X[0,0], Y[0,0]])
        return point, line0, line1, line2

    def update(i):
        #
        if (i % 1000) == 0:
            line0.set_data(t[i:i+1000], z[i:i+1000])
            ax1.set_xlim((t[i], t[i+1000]))
        #
        point.set_data(t[i], z[i])
        line1.set_data(X[i,:], Y[i,:])
        line2.set_UVC(Uy[i], Vy[i])
        line2.set_offsets([X[i,0], Y[i,0]])
        
        return point, line0, line1, line2

    # create canvas
    fig = plt.figure(figsize=(8,8))
    ax1 = plt.subplot2grid((3,3), (0,0), colspan=3)
    ax2 = plt.subplot2grid((3,3), (1,0), colspan=2, rowspan=2)
    ax3 = plt.subplot2grid((3,3), (1,2))
    # ax4 = plt.subplot2grid((3,3), (2,2))

    # plot time series
    ax1.set_ylim((-np.ceil(np.nanmax(z)), np.ceil(np.nanmax(z))))
    ax1.set_ylabel("$\\eta(t)$")

    # plot spectrum
    polar_spectrum(wfrq, dirs, E, label=True, smin=-2., smax=2.,
            fmax=0.5, ax=ax3, cbar=True)
    ax3.quiver(0, 0, Ua, Va, scale=20, color="r")
    ax3.quiver(0, 0, Um, Vm, scale=20, color="b")
    ax3.quiver(0, 0, xUy, xVy, scale=1, color="y")
    ax3.set_ylabel('')
    
    # position of animation axes
    ax2.set_aspect("equal")
    ax2.set_xlim((-3,3))
    ax2.set_ylim((-3,3))
    ax2.set_xlabel("$x$ [m]")
    ax2.set_ylabel("$y$ [m]")
    point, = ax1.plot([], [], "oy", ms=6)
    line0, = ax1.plot([], [], ls="-", color="k")
    line1, = ax2.plot([], [], "o-y")
    line2  = ax2.quiver([], [], [], [], scale=10)
    anim = animation.FuncAnimation(fig, update, init_func=init,frames=ntime,
                                   interval=20, blit=True)
# }}}


if __name__ == "__main__":

    metafile = "../metadata/bomm1_its.yml" 
    fig, ax = plot_wave_spectra(metafile, dt.datetime(2018,1,10,4,0))
    fig. savefig("bomm1_its.png", dpi=600)
    
    metafile = "../metadata/bomm2_its.yml" 
    fig, ax = plot_wave_spectra(metafile, dt.datetime(2018,4,7,13))
    fig. savefig("bomm2_its.png", dpi=600)
