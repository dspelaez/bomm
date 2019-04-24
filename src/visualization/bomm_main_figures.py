#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2019 Daniel Santiago <dpelaez@cicese.edu.mx>
#
# Distributed under terms of the GNU/GPL license.


import numpy as np
import netCDF4 as nc
import datetime as dt
import scipy.stats as stats
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.colors as mcolors
import matplotlib.gridspec as gridspec
import locale
#
from matplotlib.colors import LogNorm, LinearSegmentedColormap
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
#
from src.processing.wdm.spectra import polar_spectrum, colorax
#
plt.ioff()
locale.setlocale(locale.LC_TIME, "es_ES")


# filter warningns
np.warnings.filterwarnings('ignore')


# despiking signal {{{
def despike(signal, threshold=0.1):
    """Identify and remove the spikes and return a despiked signal."""

    # identify nans
    nans = np.isnan(signal)
    if len(np.where(nans)[0]) == len(signal):
        return signal
    
    # remove nans and compute limts
    x = signal[~nans]
    median = np.median(x)
    qr = np.percentile(x, (1,99))
    lower_limit, upper_limit = (1-threshold)*qr[0], (1+threshold)*qr[1]

    # indentify the very important peaks
    nans[signal < lower_limit] = True
    nans[signal > upper_limit] = True

    # restore nans in x
    output = signal.copy()
    output[nans] = np.nan

    return output


# }}}

# wave-spectra-timeseries {{{
def wave_spectra_timeseries(dataset, i, j):

    # load data
    time = nc.num2date(dataset["time"][i:j], dataset["time"].units)
    Hs, Tp, U10N = dataset["Hm0"][i:j], dataset["Tp"][i:j], dataset["U10N"][i:j]
    frqs, S = dataset["ffrq"][:], dataset["S"][i:j].T

    # remove outliers in the spectrum
    ix, = np.where(S[-1,:] > 1E-4)
    Hs[ix], Tp[ix], S[:, ix] =  np.nan, np.nan, np.nan

    # create figure
    fig = plt.figure(figsize=(6.2, 6.2))
    ax1 = plt.subplot2grid((5, 1), (0, 0))
    ax2 = plt.subplot2grid((5, 1), (1, 0))
    ax3 = plt.subplot2grid((5, 1), (2, 0))
    ax4 = plt.subplot2grid((5, 1), (3, 0), rowspan=2)

    for ax in (ax1, ax2, ax3):
        ax.set_xlim((time[0], time[-1]))
        ax.xaxis.set_major_locator(mdates.DayLocator(interval=17))
        ax.set_xticklabels([])

    # add data
    ax1.set_title(dataset.title[:9])
    ax1.plot(time, despike(Hs), color="k")
    ax1.set_ylabel("$H_{m0}\,\mathrm{[m]}$")
    #
    ax2.plot(time, despike(Tp), color="k")
    ax2.set_ylabel("$T_{p}\,\mathrm{[s]}$")
    #
    ax3.plot(time, despike(U10N), color="k")
    ax3.set_ylabel("$U_{10N}\,\mathrm{[m/s]}$")
    #
    colors = ["#FFFFFF", "#01DFA5", "#FE642E", "#08298A", "#01A9DB"]
    cmap = mcolors.LinearSegmentedColormap.from_list('cmap', colors, N=1024)
    norm = mcolors.Normalize(vmin=-3, vmax=1)
    #
    pc = ax4.pcolormesh(time, frqs[:64], np.log10(S[:64,:]), cmap=cmap, norm=norm)
    ax4.set_ylim((0, 0.6))
    ax4.set_ylabel("$f\,\mathrm{[Hz]}$")
    #
    cax = inset_axes(ax4,
            width="3%",
            height="100%",
            loc=3,          
            bbox_to_anchor=(1.02, 0., 1, 1),
            bbox_transform=ax4.transAxes,
            borderpad=0,
            )
    cb = fig.colorbar(pc, cax=cax)
    cb.set_label("$\mathrm{log}_{10} E$")
    #
    for ax in (ax1, ax2, ax3, ax4):
        ax.yaxis.set_major_locator(plt.MaxNLocator(5))
        ax.set_xlim((time[0], time[-1]))
        ax.xaxis.set_major_locator(plt.MaxNLocator(10))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d\n%Y"))
    #
    ax1.set_xticklabels([])
    ax2.set_xticklabels([])
    ax3.set_xticklabels([])
    #
    ax4.set_xlabel("Tiempo [UTC]")
    fig.align_ylabels((ax1,ax2, ax3, ax4))
    fig.autofmt_xdate(rotation=0, ha="center")
    #
    fname = dataset.title[:9].lower().replace("-", "_") + "_wave_parameters.png"
    fig.savefig(fname, dpi=600)


# }}}

# generate sub-plot {{{
def generate_subplot(dataset, i, j):

    # load data
    time = nc.num2date(dataset["time"][i:j], dataset["time"].units)
    U10N, tWdir = dataset["U10N"][i:j], dataset["tWdir"][i:j]
    Tw, Ta = dataset["Tw"][i:j], dataset["Ta"][i:j]
    Pa, rhum = dataset["Pa"][i:j], dataset["rhum"][i:j]
    v1, v2 = dataset["v1"][i:j], dataset["v2"][i:j]
    zL, wT = dataset["zL"][i:j], dataset["wT"][i:j]
    Hs = dataset["Hm0"][i:j]

    fig = plt.figure(figsize=(6.2, 6.2))
    gs = gridspec.GridSpec(4,1, bottom=0.1, top=.95)
    ax1 = plt.subplot(gs[0,0])
    ax2 = plt.subplot(gs[1,0])
    ax3 = plt.subplot(gs[2,0])
    ax4 = plt.subplot(gs[3,0])

    ax1.set_title(dataset.title[:9])
    ax1.plot(time, despike(U10N), c="k")
    ax1.set_ylabel("Rapidez del\nviento [m/s]")
    #
    ax1b = ax1.twinx()
    ax1b.plot(time[::6], tWdir[::6]%360, ".", c="0.5")
    ax1b.set_ylabel("Dirección del\nviento [$^\circ{}$]")
    ax1b.set_ylim((0,360))
    ax1b.set_yticks((0,90,180,270,360))
    ax1b.set_yticklabels(("N","E","S","W","N"))
    ax1.set_zorder(ax1b.get_zorder()+1)
    ax1.patch.set_visible(False)

    ax2.plot(time, Ta, color="steelblue", label="$T_\mathrm{aire}$")
    ax2.plot(time, Tw, color="darkred", label="$T_\mathrm{agua}$")
    ax2.legend(loc=0, ncol=2)
    ax2.set_ylabel("Temperatura [$^\circ{}$C]")

    ax3.plot(time, Pa, c="k", label="$P_\mathrm{atm}$")
    ax3.set_ylabel("Presión atm. [hPa]")
    #
    ax3b = ax3.twinx()
    ax3b.plot(time, rhum, c="0.5", label="$H_R$")
    ax3b.set_ylabel("Humedad\nrelativa [\%]")
    ax3b.set_ylim((-10,110))
    ax3.legend(loc=3, ncol=2)
    ax3b.legend(loc=3, bbox_to_anchor=(0.15,0))

    vmax = np.nanmax(wT)
    ax4.scatter(time[wT>0], wT[wT>0], c="red", s=3)
    ax4.scatter(time[wT<0], wT[wT<0], c="blue", s=3)
    ax4.set_ylim((-vmax, vmax))
    ax4.set_ylabel("Flujo de calor\nsensible [K m/s]")
    
    for ax in (ax1, ax2, ax3, ax4):
        ax.yaxis.set_major_locator(plt.MaxNLocator(5))
        ax.set_xlim((time[0], time[-1]))
        ax.xaxis.set_major_locator(plt.MaxNLocator(10))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d\n%Y"))

    ax1.set_xticklabels([])
    ax2.set_xticklabels([])
    ax3.set_xticklabels([])

    ax4.set_xlabel("Tiempo [UTC]")
    fig.align_ylabels((ax1,ax2, ax3, ax4))
    fig.autofmt_xdate(rotation=0, ha="center")
    #
    fname = dataset.title[:9].lower().replace("-", "_") + "_main_variables.png"
    fig.savefig(fname, dpi=600)

# }}}

# wind direction histogram {{{
def wind_direction_histogram(dataset, i, j):

    # load data
    time = nc.num2date(dataset["time"][i:j], dataset["time"].units)
    U10N = dataset["U10N"][i:j]
    Ua, Va = dataset["Ua"][i:j], dataset["Va"][i:j]

    tWdir = (270 - dataset["tWdir"][i:j]) % 360
    aWdir = np.arctan2(Va, Ua) * 180/np.pi
    delta = ((tWdir - aWdir) + 180) % 360 - 180

    fig = plt.figure(figsize=(4.2, 3.2))
    ax = fig.add_subplot(111)

    ax.hist(delta[~np.isnan(delta)], 360, normed=True)
    ax.set_xlim((-25, 25))

    ax.set_title(dataset.title[:9])

    ax.set_xlabel("$\\theta_\mathrm{Maximet} - \\theta_\mathrm{Anem\\acute{\mathrm{o}}metro}$")
    ax.set_ylabel("Probabilidad de ocurrencia")

    fname = dataset.title[:9].lower().replace("-", "_") + "_wind_direction_histogram.png"
    fig.savefig(fname, dpi=600)

# }}}

# directional_spectrum {{{
def directional_spectrum(dataset, i):

    # load data
    time = nc.num2date(dataset["time"][i], dataset["time"].units)
    wfrq, dirs, E = dataset["wfrq"][:], dataset["dirs"][:], dataset["E"][i,:,:]
    Ua, Va = dataset["Ua"][i], dataset["Va"][i]
    yaw = dataset["heading"][i]
    U10N, tWdir = dataset["U10N"][i], (270 - dataset["tWdir"][i]) % 360

    # remove energy in frequencies less than fcut
    fcut = 0.1
    E[:, wfrq < fcut] = 0.

    # plot spectrum 
    fig, ax = plt.subplots(figsize=(4,4))
    polar_spectrum(wfrq, dirs, E, smin=-3., smax=2., label=1, fmax=0.6, ax=ax)
    #
    # plot maximet true wind direction
    Um, Vm = U10N*np.cos(tWdir*np.pi/180), U10N*np.sin(tWdir*np.pi/180)
    ax.quiver(0, 0, Um, Vm, scale=40, color="b")
    #
    # plot bomm orientation as very long yellow arrow
    Uy, Vy = np.cos(yaw*np.pi/180), np.sin(yaw*np.pi/180)
    ax.quiver(0, 0, Uy, Vy, scale=1, color="y")
    #
    # plot the wind vector from sonic anemometer
    # ax.quiver(0, 0, Ua, Va, scale=20, color="r")

    title = time.strftime("%Y-%m-%d %H:%M:%S") + f"\n$U_{{10N}}$ = {U10N:.2f} m/s"
    ax.set_title(title)

    fig.savefig("direactional_spectrum_" + time.strftime("%Y%m%d%H%M"), dpi=600)


# }}}

# drag coefficient vs wind speed at 10 meters {{{
#
# compute parameterizations
def smith(u):
    return 1E-3*(0.63+0.066*u)

def large_pond(u):
    u, CD = np.array(u), np.empty_like(u)
    CD[u<=3] = 0.62+1.56/u[u<=3]
    CD[np.logical_and(u>3, u<=10)] = 1.14
    CD[u>10] = 0.49+0.065*u[u>10]
    return 1E-3 * CD

# drag coefficient vs wind speed at 10 meters
def plot_drag_coefficient(dataset, i , j):
    """Do a nice plot of the relation between wind speed and ustar."""

    # load data
    ustar = despike(dataset["ustar"][i:j].data)
    U10N = despike(dataset["U10N"][i:j].data)

    # remove outliers
    ustar[ustar > 0.8] = np.nan
    
    # remove nans
    ix = np.logical_or(np.isnan(ustar), np.isnan(U10N))
    CD = ustar**2 / U10N**2
    x, y  = U10N[~ix], CD[~ix]
    xx = np.linspace(1, x.max(), 100)

    # compute binned average and standar deviation
    binx = np.arange(0.5, np.floor(xx.max())+0.5, 0.5)
    bavg, be, bn = stats.binned_statistic(x, y, statistic='mean', bins=binx)
    bstd, be, bn = stats.binned_statistic(x, y, statistic='std', bins=binx)
    bins = be[1:] - (be[1]-be[0])/2
    fac = 1

    # plot
    fig, ax = plt.subplots(1, 1, figsize=(5,3))
    cax = colorax(ax)
    ax.set_title(dataset.title[:9])
    cax.set_title("$H_{m0}$")
    #
    # CD[CD < 0.0002] = np.nan
    # CD[CD > 0.0100] = np.nan
    Hs = dataset["Hm0"][i:j].data
    sc = ax.scatter(U10N, CD, s=2, c=Hs, vmin=0.0, vmax=np.floor(np.nanmax(Hs)))
    cb = fig.colorbar(sc, cax=cax)
    ax.plot(xx, smith(xx), lw=1.2, ls="--", c="k",
            label="Smith (1980)", zorder=2)
    ax.plot(xx, large_pond(xx), lw=1.2, ls="-", c="k",
            label="Large y Pond (1981)", zorder=2)
    ax.errorbar(bins, bavg, yerr=fac*bstd, fmt='o', capsize=3, c="k", 
            mfc="0.9", mec="k", ms=5, zorder=1, errorevery=1)
    ax.set_xlabel("$U_{10}\,\mathrm{[m/s]}$")
    ax.set_ylabel("$\mathrm{C_D}$")
    ax.legend()
    #
    fname = dataset.title[:9].lower().replace("-", "_") + "_drag_coefficient.png"
    fig.savefig(fname, dpi=600)

# --- }}}


if __name__ == "__main__":

    # set basepath
    basepath = "/Volumes/Boyas/bomm_database/data/"
    i, j = 0, -1
    
    # BOMM1-PER1
    fname = basepath + "bomm1_per1/level2/bomm1_per1_level2_30min.nc"
    dataset = nc.Dataset(fname, "r")
    wave_spectra_timeseries(dataset, i, j)
    plot_drag_coefficient(dataset, i, j)
    generate_subplot(dataset, i, j)
    wind_direction_histogram(dataset, i, j)

    # BOMM1-ITS
    fname = basepath + "bomm1_its/level2/bomm1_its_level2_30min.nc"
    dataset = nc.Dataset(fname, "r")
    wave_spectra_timeseries(dataset, i, j)
    plot_drag_coefficient(dataset, i, j)
    generate_subplot(dataset, i, j)
    wind_direction_histogram(dataset, i, j)

    # BOMM2-ITS
    fname = basepath + "bomm2_its/level2/bomm2_its_level2_30min.nc"
    dataset = nc.Dataset(fname, "r")
    wave_spectra_timeseries(dataset, i, j)
    plot_drag_coefficient(dataset, i, j)
    generate_subplot(dataset, i, j)
    wind_direction_histogram(dataset, i, j)
