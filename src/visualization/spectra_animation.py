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
import os
#
from src.processing.wdm.spectra import polar_spectrum
#
plt.ioff()
np.warnings.filterwarnings('ignore')



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

        # sonic anemometer
        self.Ua = dataset["Ua"][i:j]
        self.Va = dataset["Va"][i:j]

        # yaw and maximet data
        self.yaw = dataset['heading'][i:j]
        self.Wspd = dataset["U10N"][i:j]
        # self.tWdir = (270 - dataset["tWdir"][i:j]) % 360
        self.tWdir = np.arctan2(self.Va, self.Ua) * 180/np.pi

        # drag coefficient
        self.CD = dataset["ustar"][i:j]**2 / self.Wspd**2
        self.zL = dataset["zL"][i:j]

        # do some conversions
        self.Uy = np.cos(self.yaw*np.pi/180)
        self.Vy = np.sin(self.yaw*np.pi/180)
        self.Um = self.Wspd * np.cos(self.tWdir * np.pi/180)
        self.Vm = self.Wspd * np.sin(self.tWdir * np.pi/180)

    
    def set_limits(self, x, ax):
        """Set the limit of the axes."""

        xmin, xmax = np.floor(np.nanmin(x)), np.ceil(np.nanmax(x))
        ax.set_ylim((xmin, xmax))

    
    def plot_wave_spectrum(self, i, ax):
        """Make a plot of the wave spectrum for the given i index."""

        polar_spectrum(self.wfrq, self.dirs, self.E[i,:,:],
                label=True, smin=-2., smax=2., fmax=0.5, ax=ax, cbar=True)
        #
        # plot_arrows
        ax.quiver(0, 0, self.Ua[i], self.Va[i], scale=20, color="r")
        # ax.quiver(0, 0, Um[i], Vm[i], scale=20, color="b")
        ax.quiver(0, 0, self.Uy[i], self.Vy[i], scale=1, color="y")

        # plot wind-sea / swell delimiter
        wdirs = np.radians((self.dirs - self.tWdir[i]))
        fcut = 0.83 * 9.8 / (2 * np.pi * self.Wspd[i] * np.cos(wdirs)**1)
        # fcut[abs(wdirs)>=np.pi/3] = np.nan
        fcutx = fcut * np.cos(self.dirs*np.pi/180)
        fcuty = fcut * np.sin(self.dirs*np.pi/180)
        ax.plot(fcutx, fcuty, lw=0.5, ls="-", color="0.5")

        # # set wind label
        # ulabel = f"$u_{{10}} = {Wspd[i]:.2f}$ m/s\n" + \
                # f"$\\theta_{{u}} = {tWdir[i]:.2f}^\circ$"
        # ax.text(0.01, 0.98, ulabel, transform=ax.transAxes, ha="left", va="top")
        
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
        fig.subplots_adjust(top=.95, bottom=.1, left=.06, right=.93, wspace=.2, hspace=.1)

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
            ax.yaxis.set_label_position("right")
            ax.xaxis.set_minor_locator(mdates.HourLocator(interval=1))
            ax.xaxis.set_major_locator(mdates.HourLocator(interval=6))
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%H"))
            ax.set_xticklabels([''])
        # 
        ax3.set_xlabel(f"Hours since {self.time[0].strftime('%Y/%m/%d')}")
        #
        self.plot_wave_spectrum(i, ax0)
        # pcolor = ax0.collections[0]
        # arrow1 = ax0.collections[4]
        # arrow2 = ax0.collections[5]
        point1, = ax1.plot(self.time[i], self.Hs[i], "oy", ms=5)
        point2, = ax2.plot(self.time[i], self.Tp[i], "oy", ms=5)
        point3, = ax3.plot(self.time[i], self.Wspd[i], "oy", ms=5)

        return fig, ax0, ax1, ax2, ax3


    def animate(self):
        """Loop for each time."""

        for i in range(len(self.time)):
            fig, *ax = self.make_plot(i)
            figname = f"./figures/{self.time[i].strftime('%Y%m%d%H%M')}.png"
            fig.savefig(figname, dpi=300)
            print(f"Plotting file ---> {figname}")
            plt.close()

        os.system("convert figures/*.png -delay 100 figures/movie.gif")

    
    def plot_drag_coefficient(self):
        """Plot drag coefficient for specific dates"""
        
        def smith(u):
            return 1E-3*(0.63+0.066*u)

        def large_pond(u):
            u, CD = np.array(u), np.empty_like(u)
            CD[u<=3] = 0.62+1.56/u[u<=3]
            CD[np.logical_and(u>3, u<=10)] = 1.14
            CD[u>10] = 0.49+0.065*u[u>10]
            return 1E-3 * CD

        fig, ax = plt.subplots(1, figsize=(6,3.5))
        sc = ax.scatter(self.Wspd, self.CD, s=5, c=self.Hs, vmin=0, vmax=1.6)
        cb = fig.colorbar(sc)
        cb.set_label("$H_{m0}$ [m]")
        
        xx = np.linspace(1, np.nanmax(self.Wspd), 100)
        ax.plot(xx, smith(xx), lw=1.2, ls="--", c="k",
                label="Smith (1970)", zorder=2)
        ax.plot(xx, large_pond(xx), lw=1.2, ls="-", c="k",
                label="Large y Pond (1981)", zorder=2)

        ax.set_xlim((0.,14))
        ax.set_ylim((-0.001,0.015))
        ax.set_xlabel("$U_{10N}$ [m/s]")
        ax.set_ylabel("$C_{D}$")
        ax.legend()
        fig.savefig("figures/drag_coefficient.png", dpi=600)
# }}}







if __name__ == "__main__":

    path = "../.."
    filename = f"{path}/data/bomm2_its/level2/bomm2_its.level2.10min.nc"
    t_ini = dt.datetime(2018, 4, 11)
    t_end = dt.datetime(2018, 4, 14)
    s = PlotSpectra(filename, t_ini, t_end)
    s.animate()
    s.plot_drag_coefficient()

