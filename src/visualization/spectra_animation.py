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
import netCDF4 as nc
import datetime as dt
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.dates as mdates
import locale
import os
#
from src.processing.wdm.spectra import polar_spectrum
#
plt.ioff()
np.warnings.filterwarnings('ignore')
locale.setlocale(locale.LC_TIME, "es_ES")


# main class {{{
class PlotSpectra(object):

    """Class to create a cool animation of the wave spectra"""

    def __init__(self, filename, t_ini, t_end):
        """Initialize the class"""

        # load netcdf file
        dataset = nc.Dataset(filename, "r")

        # load time and fine coincident dates
        time = nc.num2date(dataset["time"][:], dataset["time"].units)
        i, j = np.argmin(abs(time - t_ini)), np.argmin(abs(time - t_end))
        self.time = time[i:j]
        
        # directional spectra
        self.wfrq = dataset["wfrq"][:]
        self.dirs = dataset["dirs"][:]
        self.E = dataset["E"][i:j,:]
        self.Hs = dataset["Hm0"][i:j]
        self.Tp = dataset["Tp"][i:j]

        # stokes drift
        # self.Us = self.Hs * np.cos(np.radians(dataset["mDir"][i:j]))
        # self.Vs = self.Hs * np.sin(np.radians(dataset["mDir"][i:j]))

        # sonic anemometer
        self.Ua = dataset["Ua"][i:j]
        self.Va = dataset["Va"][i:j]

        # yaw and maximet data
        self.yaw = dataset['heading'][i:j]
        self.Wspd = dataset["U10N"][i:j]
        self.tWdir = (270 - dataset["tWdir"][i:j]) % 360
        self.aWdir = np.arctan2(self.Va, self.Ua) * 180/np.pi

        # drag coefficient
        self.CD = dataset["ustar"][i:j]**2 / self.Wspd**2
        self.zL = dataset["zL"][i:j]

        # do some conversions
        self.Uy = np.cos(self.yaw*np.pi/180)
        self.Vy = np.sin(self.yaw*np.pi/180)
        self.Um = self.Wspd * np.cos(self.tWdir * np.pi/180)
        self.Vm = self.Wspd * np.sin(self.tWdir * np.pi/180)

        # creat folder to store figures
        bomm_name = os.path.split(fname)[-1][:9]
        t_ini_str = t_ini.strftime('%Y%m%d')
        t_fin_str = t_ini.strftime('%Y%m%d')
        self.folder = f"./animation_{bomm_name}_{t_ini_str}_{t_fin_str}/"

    
    def set_limits(self, x, ax):
        """Set the limit of the axes."""

        xmin, xmax = np.floor(np.nanmin(x)), np.ceil(np.nanmax(x))
        ax.set_ylim((xmin, xmax))


    def stokes_drift(self, f, S, z=0):
        """Compute stokes drift profile as Breivik et al 2016 eq5."""
        
        # angular frequency and spectrum in right units
        g = 9.8
        w = 2*np.pi * f
        k = w**2 / g
        Sw = S / (2*np.pi)
        
        fac = 2 / g
        dummy = w**3 * Sw * np.exp(2*k*z)
        return np.trapz(fac*dummy, w)

    
    def plot_wave_spectrum(self, i, ax):
        """Make a plot of the wave spectrum for the given i index."""

        polar_spectrum(self.wfrq, self.dirs, self.E[i,:,:],
                label=True, smin=-2., smax=2., fmax=0.5, ax=ax, cbar=True)
        #
        # plot_arrows
        ax.quiver(0, 0, self.Ua[i], self.Va[i], scale=30,  color="blue")
        ax.quiver(0, 0, self.Um[i], self.Vm[i], scale=30,  color="darkblue")
        ax.quiver(0, 0, self.Uy[i], self.Vy[i], scale=1,   color="gold")
        # ax.quiver(0, 0, self.Us[i], self.Vs[i], scale=10, color="darkred")

        # plot wind-sea / swell delimiter
        wdirs = np.radians((self.dirs - self.tWdir[i]))
        fcut = 0.83 * 9.8 / (2 * np.pi * self.Wspd[i] * np.cos(wdirs)**1)
        # fcut[abs(wdirs)>=np.pi/3] = np.nan
        fcutx = fcut * np.cos(self.dirs*np.pi/180)
        fcuty = fcut * np.sin(self.dirs*np.pi/180)
        ax.plot(fcutx, fcuty, lw=0.5, ls="-", color="0.5")

        # # set wind label
        ulabel = f"$u_{{10}} = {self.Wspd[i]:.2f}$ m/s\n" + \
                f"$\\theta_{{u}} = {self.tWdir[i]:.2f}^\circ$"
        ax.text(0.01, 0.98, ulabel, transform=ax.transAxes, ha="left", va="top")
        
        title = self.time[i].strftime("%Y-%m-%d %H:%M:%S")
        ax.set_title(title)


    def make_plot(self, i):
        """Start the figure and plot permanent data"""

        # create canvas
        fig = plt.figure(figsize=(8,4))
        ax0 = plt.subplot2grid(shape=(3,2), loc=(0,0), rowspan=3)
        ax1 = plt.subplot2grid(shape=(3,2), loc=(0,1))
        ax2 = plt.subplot2grid(shape=(3,2), loc=(1,1))
        ax3 = plt.subplot2grid(shape=(3,2), loc=(2,1))
        fig.subplots_adjust(top=.95, bottom=.1, left=.06, right=.93, wspace=.2, hspace=.2)

        ax1.plot(self.time, self.Hs, c="k")
        ax2.plot(self.time, self.Tp, c="k")
        ax3.plot(self.time, self.Wspd, c="k")
        #
        ax1.set_ylabel("$H_{m0}\,\mathrm{[m]}$")
        ax2.set_ylabel("$T_{p}\,\mathrm{[m]}$")
        ax3.set_ylabel("$U_{10}\,\mathrm{[m]}$")
        #
        self.set_limits(self.Hs, ax1)
        self.set_limits(self.Tp, ax2)
        self.set_limits(self.Wspd, ax3)
        #
        for ax in (ax1, ax2, ax3):
            ax.yaxis.tick_right()
            ax.yaxis.set_major_locator(plt.MaxNLocator(4))
            ax.yaxis.set_label_position("right")
            ax.yaxis.set_ticks_position("both")
            ax.xaxis.set_minor_locator(mdates.HourLocator(range(0,24,3)))
            ax.xaxis.set_major_locator(mdates.DayLocator(interval=1))
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d\n%Y"))


        ax1.set_xticklabels([''])
        ax2.set_xticklabels([''])
        # 
        # ax3.set_xlabel(f"Hours since {self.time[0].strftime('%Y/%m/%d')}")
        #
        self.plot_wave_spectrum(i, ax0)
        point1, = ax1.plot(self.time[i], self.Hs[i], "oy", ms=3)
        point2, = ax2.plot(self.time[i], self.Tp[i], "oy", ms=3)
        point3, = ax3.plot(self.time[i], self.Wspd[i], "oy", ms=3)

        return fig, ax0, ax1, ax2, ax3


    def animate(self):
        """Loop for each time."""

        # create folder if it does not exist
        os.system(f"mkdir -p {self.folder}")

        for i in range(len(self.time)):
            fig, *ax = self.make_plot(i)
            figname = f"{self.folder}/{self.time[i].strftime('%Y%m%d%H%M')}.png"
            fig.savefig(figname, dpi=100)
            print(f"Plotting file ---> {figname}")
            plt.close()

        c = f"convert {self.folder}/*.png -delay 100 -quality 50 {self.folder}/movie.gif"
        os.system(c)

# }}}


if __name__ == "__main__":

    basepath = "/Volumes/Boyas/bomm_database/data/"
    fname = basepath + "bomm1_per1/level2/bomm1_per1_level2_30min.nc"
    t_ini = dt.datetime(2018, 11,  9)
    t_end = dt.datetime(2018, 11, 15)
    s = PlotSpectra(fname, t_ini, t_end)
    s.animate()

