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
import matplotlib.colors as mcolors
import locale
import os
#
from matplotlib.colors import LogNorm, LinearSegmentedColormap
#
# import wdm
import src.processing.motion_correction as motcor
from src.processing.wdm.spectra import polar_spectrum, colorax
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
        self.yaw = dataset['yaw'][i:j]
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
        self.Uy = np.cos((self.yaw + 90) * np.pi/180)
        self.Vy = np.sin((self.yaw + 90) * np.pi/180)
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
        #
        self.t_wave = np.arange(0, len(self.p.wav["time"])) / 60 / 20
        self.Z = self.p.Z

    
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
        qva = ax.quiver(0, 0, self.Ua[i], self.Va[i], scale=30, color="steelblue")
        qvt = ax.quiver(0, 0, self.Tx[i], self.Ty[i], scale=0.5, color="darkblue")
        qvs = ax.quiver(0, 0, self.Us[i], self.Vs[i], scale=0.3, color="darkred")
        qvy = ax.quiver(0, 0, self.Uy[i], self.Vy[i], scale=1, color="gold", headwidth=0)

        ax.quiverkey(qva, 0.10, -0.08, 5.00, label="$U_{10} = 5\,\mathrm{m/s}$")
        ax.quiverkey(qvs, 0.38, -0.08, 0.05, label="$U_s = 5\,\mathrm{cm/s}$")
        ax.quiverkey(qvt, 0.66, -0.08, 0.10, label="$\\tau = 0.1\,\mathrm{N/m^2}$")
        ax.quiverkey(qvy, 0.94, -0.08, 0.20, label="$\psi$")

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
        ulabel = f"$U_{{10}} = {self.Wspd[i]:.2f}$ m/s\n" + \
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
        fig = plt.figure(figsize=(10,8))
        #
        ax0 = fig.add_axes([0.05, 0.50, 0.42, 0.45])
        ax1 = fig.add_axes([0.55, 0.82, 0.40, 0.13])
        ax2 = fig.add_axes([0.55, 0.66, 0.40, 0.13])
        ax3 = fig.add_axes([0.55, 0.50, 0.40, 0.13])
        #
        bx1 = fig.add_axes([0.05, 0.33, 0.90, 0.10])
        bx2 = fig.add_axes([0.05, 0.23, 0.90, 0.10])
        bx3 = fig.add_axes([0.05, 0.13, 0.90, 0.10])
        #
        cx0 = fig.add_axes([0.05, 0.03, 0.90, 0.10])


        ax1.set_title(self.title)
        ax1.plot(self.time, self.Wspd, c="k")
        ax2.plot(self.time, self.Hs,   c="k")
        ax3.plot(self.time, self.Tp,   c="k")
        #
        ax1.set_ylabel("$U_{10}\,\mathrm{[m/s]}$")
        ax2.set_ylabel("$H_{m0}\,\mathrm{[m]}$")
        ax3.set_ylabel("$T_{p}\,\mathrm{[s]}$")
        #
        self.set_limits(self.Wspd, ax1)
        self.set_limits(self.Hs, ax2)
        self.set_limits(self.Tp, ax3)
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
        # plot high frequency data
        try:
            self.get_high_frequency_data(i)
            # 
            bx1.plot(self.t_wind, self.u_wind, color="0.9")
            bx2.plot(self.t_wind, self.v_wind, color="0.9")
            bx3.plot(self.t_wind, self.w_wind, color="0.9")
            # bx1.set_ylim((-20,20))
            # bx2.set_ylim((-20,20))
            # bx3.set_ylim((-5,5))
            #
            n_smooth = 200
            bx1.plot(self.t_wind[::n_smooth], self.u_wind[::n_smooth])
            bx2.plot(self.t_wind[::n_smooth], self.v_wind[::n_smooth])
            bx3.plot(self.t_wind[::n_smooth], self.w_wind[::n_smooth])
            #
            bx1.set_ylabel("$u\,\mathrm{[m/s]}$")
            bx2.set_ylabel("$v\,\mathrm{[m/s]}$")
            bx3.set_ylabel("$w\,\mathrm{[m/s]}$")
            #
            cx0.plot(self.t_wave, self.Z[:,self.p.valid_wires_index])
            cx0.legend(self.p.valid_wires, loc=0, ncol=6)
            cx0.set_ylabel("$\\eta\,\mathrm{[m]}$")
            # cx0.set_ylim((-2,2))
            #
            for ax in (bx1, bx2, bx3, cx0):
                ax.yaxis.set_label_position("right")

        except:
            pass

        # plot wave spectrum
        self.plot_wave_spectrum(i, ax0)
        point1, = ax1.plot(self.time[i], self.Wspd[i], "oy", ms=3)
        point2, = ax2.plot(self.time[i], self.Hs[i], "oy", ms=3)
        point3, = ax3.plot(self.time[i], self.Tp[i], "oy", ms=3)


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


    def time_directional_spectrum(self, fname=None):
        """Nice way to visualize the time-varying directional wave spectrum"""

        # determine number of time
        ntime = len(self.time)
        step = np.max((1, int(ntime/50)))

        # compute energy components
        dirr = np.radians(self.dirs)
        S = np.trapz(self.E, x=dirr, axis=1).T
        Ex = np.trapz(np.cos(dirr[None,:,None]) * self.E, x=dirr, axis=1).T / S
        Ey = np.trapz(np.sin(dirr[None,:,None]) * self.E, x=dirr, axis=1).T / S

        # remove some elements
        ix_remove = np.logical_or(self.wfrq < 0.06, self.wfrq > 0.8)
        Ex[ix_remove] = np.nan
        Ey[ix_remove] = np.nan

        # create colormap
        smin, smax = -3, 1
        colors = ["#FFFFFF", "#01DFA5", "#FE642E", "#08298A", "#01A9DB"]
        cmap = mcolors.LinearSegmentedColormap.from_list('cmap', colors, N=1024)
        norm = mcolors.LogNorm(vmin=10**smin, vmax=10**smax)

        # do plot spectrum
        fig, ax = plt.subplots(1, figsize=(7,4))
        ax.set_title(self.title)
        #
        ax.plot(s.time, 1/self.Tp, ".", ms=3, color="0.5", alpha=0.5)
        pc = ax.pcolormesh(self.time, self.wfrq, S, cmap=cmap, norm=norm)
        qv = ax.quiver(self.time[::step], self.wfrq[::4], Ex[::4,::step], Ey[::4,::step],
                scale=50, angles="uv", color="black")

        # plot wind at 0.9 freq
        qw = ax.quiver(self.time[::step], 0.85, self.Ua[::step], self.Va[::step],
                scale=200, width=0.004, headwidth=2.5, angles="uv", color="blue")
        ax.quiverkey(qw, 0.80, 1.02, 10, label="$U_{10} = 10\,\mathrm{m/s}$",
                labelpos="E")

        # colorbar
        fig.colorbar(pc, ax=ax, cax=colorax(ax))#, ticks=10**np.arange(smin, smax+1))

        # tweak the axes
        ax.set_xlim((self.time[0], self.time[-1]+dt.timedelta(minutes=30)))
        ax.set_ylim((0.05, 0.95))
        ax.set_ylabel("$f\,\mathrm{[Hz]}$")
        #
        ax.xaxis.set_minor_locator(mdates.HourLocator(range(0,24,3)))
        ax.xaxis.set_major_locator(mdates.DayLocator(interval=1))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d\n%Y"))

        if fname:
            fig.savefig(fname, dpi=600)

        return fig, ax

# }}}


if __name__ == "__main__":

    # case when we have a cold front observed by a sar image
    if True:
        t_ini = dt.datetime(2018, 9, 11)
        t_end = dt.datetime(2018, 9, 16)
        s = PlotSpectra("../../metadata/bomm1_per1.yml", t_ini, t_end, 30)
        # s.animate()
        s.time_directional_spectrum("./events/spectra_event_1.png")

    # define initial and final time (no more than 3 or 4 days)
    if True:
        t_ini = dt.datetime(2018, 10, 15)
        t_end = dt.datetime(2018, 10, 20)
        s = PlotSpectra("../../metadata/bomm1_per1.yml", t_ini, t_end, 30)
        s.animate()
        s.time_directional_spectrum("./events/spectra_event_2.png")

    # case when we observed cross-winds and slanting fecth
    if True:
        t_ini = dt.datetime(2018, 11, 9)
        t_end = dt.datetime(2018, 11, 16)
        s = PlotSpectra("../../metadata/bomm1_per1.yml", t_ini, t_end, 30)
        s.animate()
        s.time_directional_spectrum("./events/spectra_event_3.png")

    # plot specific date only
    if False:

        # bomm1-its
        date = dt.datetime(2017, 12, 8, 14, 0)
        t_ini = date - dt.timedelta(days=1)
        t_end = date + dt.timedelta(days=1)
        s = PlotSpectra("../../metadata/bomm1_its.yml", t_ini, t_end, 10)
        i = np.argmin(abs(s.time - date))
        s.make_plot(i)
        
        # bomm1-per1
        date = dt.datetime(2018, 10, 16, 6, 0)
        t_ini = date - dt.timedelta(days=1)
        t_end = date + dt.timedelta(days=1)
        s = PlotSpectra("../../metadata/bomm1_per1.yml", t_ini, t_end, 10)
        i = np.argmin(abs(s.time - date))
        s.make_plot(i)

    # terrasar x
    if True:
        
        dates_tsx = [
                '20180713T122722', '20180714T003126', '20180720T000255',
                '20180822T002322', '20180917T122725', '20180928T122726',
                '20181016T002302', '20181020T122727', '20181021T003133',
                '20181112T003131', '20181118T002259'
                ]

        for strdate in dates_tsx:
            date = dt.datetime.strptime(strdate, "%Y%m%dT%H%M%S")
            t_ini = date - dt.timedelta(days=1)
            t_end = date + dt.timedelta(days=1)
            s = PlotSpectra("../../metadata/bomm1_per1.yml", t_ini, t_end, 10)
            i = np.argmin(abs(s.time - date))
            fig, *ax = s.make_plot(i)
            fig.savefig(f"./terrasarx/{strdate}_10min.png", dpi=300)
            plt.close(fig)

    # sentinel
    if True:

        dates_snt = [
                '20180712T003311', '20180805T003313', '20180807T122429',
                '20180817T003314', '20180819T122429', '20180831T122430',
                '20180910T003315', '20180912T122430', '20181023T002513',
                '20181109T003315', '20181111T122431', '20181203T003315',
                '20181205T122430'
                ]

        for strdate in dates_snt:
            date = dt.datetime.strptime(strdate, "%Y%m%dT%H%M%S")
            t_ini = date - dt.timedelta(days=1)
            t_end = date + dt.timedelta(days=1)
            s = PlotSpectra("../../metadata/bomm1_per1.yml", t_ini, t_end, 10)
            i = np.argmin(abs(s.time - date))
            fig, *ax = s.make_plot(i)
            fig.savefig(f"./sentinel/{strdate}_10min.png", dpi=300)
            plt.close(fig)
