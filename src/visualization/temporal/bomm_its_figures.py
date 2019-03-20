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
#
from matplotlib.colors import LogNorm, LinearSegmentedColormap
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
#
plt.ion()


# filter warningns
np.warnings.filterwarnings('ignore')

# laod bomm1-level2 dataset
fname = "/Volumes/BOMM/cigom/data/bomm2_its/level2/bomm2_its.level2.nc"
dataset = nc.Dataset(fname, "r")



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

# time-series of wave-spectra {{{
def wave_spectra_timeseries(dataset):

    # load data
    time = nc.num2date(dataset["time"][:], dataset["time"].units)
    time = np.array([np.datetime64(t) for t in time]) # experimental
    Hs, Tp, U10N = dataset["Hm0"][:], dataset["Tp"][:], dataset["U10N"][:]
    frqs, S = dataset["ffrq"][:], dataset["S"][:].T

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
    ax1.plot(time, despike(Hs), color="k")
    ax1.set_ylim((0,4))
    ax1.set_yticks([0.5,1.5,2.5,3.5])
    ax1.set_ylabel("$H_{m0}\,\mathrm{[m]}$")
    #
    ax2.plot(time, despike(Tp), color="k")
    ax2.set_ylim((5,24))
    ax2.set_yticks([7,12,17,22])
    ax2.set_ylabel("$T_{p}\,\mathrm{[s]}$")
    #
    ax3.plot(time, U10N, color="k")
    ax3.set_ylim((0,16))
    ax3.set_yticks([2, 6, 10, 14])
    ax3.set_ylabel("$U_{10N}\,\mathrm{[m/s]}$")
    #
    colors = ["#FFFFFF", "#01DFA5", "#FE642E", "#08298A", "#01A9DB"]
    cmap = mcolors.LinearSegmentedColormap.from_list('cmap', colors, N=1024)
    norm = mcolors.Normalize(vmin=-2, vmax=1)
    #
    pc = ax4.pcolormesh(time, frqs[:64], np.log10(S[:64,:]), cmap=cmap, norm=norm)
    ax4.set_ylim((0,0.4))
    ax4.set_xlim((time[0], time[-1]))
    ax4.set_xlabel("Tiempo [UTC]")
    ax4.set_ylabel("$f\,\mathrm{[Hz]}$")
    ax4.xaxis.set_major_locator(mdates.DayLocator(interval=17))
    ax4.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
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
    fig.savefig("bomm2_its_wave_parameters.png", dpi=600)


# }}}

# generate sub-plot {{{
def generate_subplot(dataset):

    # load data
    time = nc.num2date(dataset["time"][:], dataset["time"].units)
    time = np.array([np.datetime64(t) for t in time]) # experimental
    U10N, tWdir = dataset["U10N"][:], dataset["tWdir"][:]
    Tw, Ta = dataset["Tw"][:], dataset["Ta"][:]
    Pa, rhum = dataset["Pa"][:], dataset["rhum"][:]
    v1, v2 = dataset["v1"][:], dataset["v2"][:]
    Hs = dataset["Hm0"][:]

    fig = plt.figure(figsize=(6.2, 5.0))
    gs = gridspec.GridSpec(3,1, bottom=0.1, top=.95)
    ax1 = plt.subplot(gs[0,0])
    ax2 = plt.subplot(gs[1,0])
    ax3 = plt.subplot(gs[2,0])

    ax1.set_title(dataset.title[:9])
    ax1.plot(time, U10N, c="k")
    ax1.set_ylim((0,15))
    ax1.set_ylabel("Rapidez del\nviento [m/s]")
    #
    ax1b = ax1.twinx()
    ax1b.plot(time[::6], tWdir[::6]%360, ".", c="0.5")
    ax1b.set_ylabel("Dirección del\nviento [$^\circ{}$]")
    ax1b.set_ylim((0,360))
    ax1b.set_yticks((0,90,180,270,360))
    ax1.set_zorder(ax1b.get_zorder()+1)
    ax1.patch.set_visible(False)

    ax2.plot(time, Ta, color="steelblue", label="$T_\mathrm{aire}$")
    ax2.plot(time, Tw, color="darkred", label="$T_\mathrm{agua}$")
    ax2.set_ylim((8,22))
    ax2.legend(loc=0, ncol=2)
    ax2.set_ylabel("Temperatura [$^\circ{}$C]")

    ax3.plot(time, Pa, c="k")
    ax3.set_ylabel("Presión atm. [hPa]")
    #
    ax3b = ax3.twinx()
    ax3b.plot(time, rhum, c="0.5")
    ax3b.set_ylabel("Humedad\nrelativa [\%]")
    ax3b.set_ylim((-10,110))
    
    for ax in (ax1, ax2):
        ax.set_xticklabels([])
        ax.set_xlim((time[0], time[-1]))
        ax.xaxis.set_major_locator(mdates.DayLocator(interval=15))

    ax3.set_xlabel("Tiempo [UTC]")
    ax3.set_xlim((time[0], time[-1]))
    ax3.xaxis.set_major_locator(mdates.DayLocator(interval=15))
    ax3.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    fig.align_ylabels((ax1,ax2,ax3))
    #
    fig.savefig("variables_ambientales.png", dpi=600)

# }}}

# directional_spectrum {{{
def directional_spectrum(dataset, i):

    # import local functions
    from wdm.spectra import polar_spectrum

    # load data
    time = nc.num2date(dataset["time"][i], dataset["time"].units)
    wfrq, dirs, E = dataset["wfrq"][:], dataset["dirs"][:], dataset["E"][i,:,:]
    Ua, Va = dataset["Ua"][i], dataset["Va"][i]
    yaw = dataset["heading"][i]
    U10N, tWdir = dataset["U10N"][i], (270 - dataset["tWdir"][i]) % 360

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
def plot_drag_coefficient(dataset):
    """Do a nice plot of the relation between wind speed and ustar."""

    # load data
    ustar = despike(dataset["ustar"][:])
    U10N = despike(dataset["U10N"][:])
    U10N[U10N > 13] = np.nan

    # remove nans
    ix = np.logical_or(np.isnan(ustar), np.isnan(U10N))
    CD = ustar**2 / U10N**2
    x, y  = U10N[~ix], CD[~ix]
    xx = np.linspace(1, x.max(), 100)

    # compute binned average and standar deviation
    bavg, be, bn = stats.binned_statistic(x, y, statistic='mean', bins=25)
    bstd, be, bn = stats.binned_statistic(x, y, statistic='std', bins=25)
    bins = be[1:] - (be[1]-be[0])/2
    fac = 1

    # plot
    fig, ax = plt.subplots(1, 1, figsize=(5,3))
    # cax = gph.colorax(ax)
    # cax.set_title("$u_{\star a} / c_p$")
    #
    CD[CD < 0.0002] = np.nan
    CD[CD > 0.0200] = np.nan
    sc = ax.scatter(U10N, CD, s=2, cmap="gist_earth", vmin=0.0, vmax=0.05)
    ax.plot(xx, smith(xx), lw=1.2, ls="--", c="k",
            label="Smith (1970)", zorder=2)
    ax.plot(xx, large_pond(xx), lw=1.2, ls="-", c="k",
            label="Large y Pond (1981)", zorder=2)
    ax.errorbar(bins, bavg, yerr=fac*bstd, fmt='o', capsize=3, c="k", 
            mfc="0.9", mec="k", ms=5, zorder=1, errorevery=1)
    ax.set_xlabel("$U_{10}\,\mathrm{[m/s]}$")
    ax.set_ylabel("$\mathrm{C_D}$")
    ax.legend()
    #
    fig.savefig("drag_coefficient.png", dpi=600)

# --- }}}


if __name__ == "__main__":
    wave_spectra_timeseries(dataset)
    plot_drag_coefficient(dataset)
    generate_subplot(dataset)

    for i in np.random.choice(len(dataset["time"][:]), size=10):
        try:
            directional_spectrum(dataset, i)
        except:
            pass
