#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2019 Daniel Santiago <dpelaez@cicese.edu.mx>
#
# Distributed under terms of the GNU/GPL license.

"""
This script contains functions that perform a check for the proccesed data.

TODO:
    - [ ] Avoid non-avaliable spectra
    - [ ] Function to convert from PNG to MP4

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
# import wdm
import src.processing.motion_correction as motcor
from src.processing.wdm.spectra import polar_spectrum
from src.processing.processing_data import (
        ProcessingData, eddy_correlation_flux, nanmean, stokes_drift
        )
#
plt.ion()
np.warnings.filterwarnings('ignore')
locale.setlocale(locale.LC_TIME, "es_ES")


# main class {{{
class PlotSpectra(object):

    """Class to create a cool animation of the wave spectra"""

    def __init__(self, metafile, t_ini, t_end, number_of_minutes=30):
        """Initialize the class"""

        # perform the data processing
        self.metafile = metafile
        self.p = ProcessingData(self.metafile, number_of_minutes)
        self.metadata = self.p.metadata

        # load netcdf file
        # TODO: extrat filename from METAFILE
        bomm_name = self.metadata["name"]
        nm = int(number_of_minutes)
        filename = f"{bomm_name}/level2/{bomm_name}_level2_{nm}min.nc" 
        self.filename = self.metadata["basepath"] + filename
        dataset = nc.Dataset(self.filename, "r")

        # load time and fine coincident dates
        time = nc.num2date(dataset["time"][:], dataset["time"].units)
        i, j = np.argmin(abs(time - t_ini)), np.argmin(abs(time - t_end))
        self.time = time[i:j]

        # bomm title
        self.title = dataset.title[:10].strip()
        
        # directional spectra
        self.wfrq = dataset["wfrq"][:]
        self.dirs = dataset["dirs"][:]
        self.E = dataset["E"][i:j,:]
        self.Hs = dataset["Hm0"][i:j]
        self.Tp = dataset["Tp"][i:j]

        # stokes drift
        dirr = np.radians(self.dirs)
        Ex = np.trapz(np.cos(dirr[None,:,None]) * self.E, x=dirr, axis=1)
        Ey = np.trapz(np.sin(dirr[None,:,None]) * self.E, x=dirr, axis=1)
        self.Us = self.stokes_drift(self.wfrq, Ex, z=0)
        self.Vs = self.stokes_drift(self.wfrq, Ey, z=0)

        # sonic anemometer
        self.Ua = dataset["Ua"][i:j]
        self.Va = dataset["Va"][i:j]
        self.ustar = dataset["ustar"][i:j]
        
        # yaw and maximet data
        self.yaw = dataset['heading'][i:j]
        self.Wspd = dataset["U10N"][i:j]
        self.tWdir = (270 - dataset["tWdir"][i:j]) % 360
        self.aWdir = np.arctan2(self.Va, self.Ua) * 180/np.pi

        # wind stress 
        angle = -np.radians(self.aWdir)
        taux = -dataset["rhoa"][i:j] * dataset["uw"][i:j]
        tauy = -dataset["rhoa"][i:j] * dataset["vw"][i:j]
        self.Tx =  taux * np.cos(angle) + tauy * np.sin(angle)
        self.Ty = -taux * np.sin(angle) + tauy * np.cos(angle)

        # drag coefficient
        self.CD = dataset["ustar"][i:j]**2 / self.Wspd**2
        self.zL = dataset["zL"][i:j]

        # do some conversions
        self.Uy = np.cos(self.yaw*np.pi/180)
        self.Vy = np.sin(self.yaw*np.pi/180)
        self.Um = self.Wspd * np.cos(self.tWdir * np.pi/180)
        self.Vm = self.Wspd * np.sin(self.tWdir * np.pi/180)

        # creat folder to store figures
        t_ini_str = t_ini.strftime('%Y%m%d')
        t_end_str = t_end.strftime('%Y%m%d')
        # TODO: choose anoher path to store animation
        self.folder = f"./animation_{bomm_name}_{t_ini_str}_{t_end_str}/"


    def remove_outliers(self, x):
        """Recursively remove outliers from a give signal"""

        # compute mean and standar deviation
        xmean, xstd = nanmean(x), np.nanstd(x)

        # first remove values lying 5 time std
        x_clean = x.copy()
        x_clean[abs(x - xmean) > 5*xstd] = np.nan

        return x_clean


    def get_high_frequency_data(self, i):
        """Get first level data corresponding to the same date"""

        # detrend function
        detrend = lambda x: x - nanmean(x)

        date = self.time[i]
        self.p.run(date)
        self.t_wind = np.arange(0, len(self.p.wnd["time"])) / 60 / 100
        self.u_wind = self.remove_outliers(self.p.U_cor[0])
        self.v_wind = self.remove_outliers(self.p.U_cor[1])
        self.w_wind = self.remove_outliers(self.p.U_cor[2])

    
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
        return np.trapz(fac*dummy[:,:80], w[:80], axis=1)

    
    def plot_wave_spectrum(self, i, ax):
        """Make a plot of the wave spectrum for the given i index."""

        polar_spectrum(self.wfrq, self.dirs, self.E[i,:,:],
                label=True, smin=-3., smax=2, fmax=0.5, ax=ax, cbar=True)
        #
        # plot_arrows
        ax.quiver(0, 0, self.Ua[i], self.Va[i], scale=30, color="blue")
        ax.quiver(0, 0, self.Tx[i], self.Ty[i], scale=0.5, color="darkblue")
        ax.quiver(0, 0, self.Uy[i], self.Vy[i], scale=1, color="gold")
        ax.quiver(0, 0, self.Us[i], self.Vs[i], scale=0.3, color="darkred")

        # plot wind-sea / swell delimiter
        wdirs = np.radians((self.dirs - self.tWdir[i]))
        fcut = 0.83 * 9.8 / (2 * np.pi * self.Wspd[i] * np.cos(wdirs)**1)
        # fcut[abs(wdirs)>=np.pi/3] = np.nan
        fcutx = fcut * np.cos(self.dirs*np.pi/180)
        fcuty = fcut * np.sin(self.dirs*np.pi/180)
        ax.plot(fcutx, fcuty, lw=0.5, ls="-", color="0.5")

        # TODO:
        # - add Ustar and Ustokes

        # # set wind label
        ulabel = f"$u_{{10}} = {self.Wspd[i]:.2f}$ m/s\n" + \
                f"$\\theta_{{u}} = {self.tWdir[i]:.2f}^\circ$"
        ax.text(0.01, 0.98, ulabel, transform=ax.transAxes, ha="left", va="top")
        
        title = self.time[i].strftime("%Y-%m-%d %H:%M:%S")
        ax.set_title(title)

        # remove lower labels
        ax.set_xlabel('')
        ax.set_xticklabels('')


    def make_plot(self, i):
        """Start the figure and plot permanent data"""

        # create canvas
        fig = plt.figure(figsize=(10,7))
        #
        ax0 = fig.add_axes([0.05, 0.40, 0.42, 0.55])
        ax1 = fig.add_axes([0.55, 0.78, 0.40, 0.17])
        ax2 = fig.add_axes([0.55, 0.59, 0.40, 0.17])
        ax3 = fig.add_axes([0.55, 0.40, 0.40, 0.17])
        #
        bx1 = fig.add_axes([0.05, 0.23, 0.90, 0.10])
        bx2 = fig.add_axes([0.05, 0.13, 0.90, 0.10])
        bx3 = fig.add_axes([0.05, 0.03, 0.90, 0.10])

        # fig.subplots_adjust(top=.95, bottom=.1, left=.06, right=.93, wspace=.2, hspace=.2)

        ax1.set_title(self.title)
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

        # plot high frequency data
        try:
            self.get_high_frequency_data(i)
            # 
            bx1.plot(self.t_wind, self.u_wind, color="0.9")
            bx2.plot(self.t_wind, self.v_wind, color="0.9")
            bx3.plot(self.t_wind, self.w_wind, color="0.9")
            #
            n_smooth = 200
            bx1.plot(self.t_wind[::n_smooth], self.u_wind[::n_smooth])
            bx2.plot(self.t_wind[::n_smooth], self.v_wind[::n_smooth])
            bx3.plot(self.t_wind[::n_smooth], self.w_wind[::n_smooth])
        except:
            pass

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
        # ffmpeg -i movie.gif -movflags faststart -pix_fmt yuv420p -vf "scale=trunc(iw/2)*2:trunc(ih/2)*2" movie.mp4
        # ffmpeg -y -framerate 2 -pattern_type glob -i '*.png' -movflags faststart -pix_fmt yuv420p -vf "scale=trunc(iw/2)*2:trunc(ih/2)*2" movie.mp4

# }}}


if __name__ == "__main__":

    # define initial and final time (no more than 3 or 4 days)
    t_ini = dt.datetime(2018, 10, 15)
    t_end = dt.datetime(2018, 10, 18)
    s = PlotSpectra("../../metadata/bomm1_per1.yml", t_ini, t_end, 30)
    s.animate()

    # case when minium Hs
    # t_ini = dt.datetime(2018, 9, 17, 12, 0)
    # t_end = dt.datetime(2018, 9, 21, 12, 0)
    # s = PlotSpectra("../../metadata/bomm1_per1.yml", t_ini, t_end)
    # s.animate()

    if False:
        t_ini = dt.datetime(2018, 1, 17)
        t_end = dt.datetime(2018, 1, 19)
        s = PlotSpectra("../../metadata/bomm1_its.yml", t_ini, t_end, 10)
        s.animate()

        t_ini = dt.datetime(2018, 1, 30)
        t_end = dt.datetime(2018, 1, 31)
        s = PlotSpectra("../../metadata/bomm1_its.yml", t_ini, t_end, 10)
        s.animate()
