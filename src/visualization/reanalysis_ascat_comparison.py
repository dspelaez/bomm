#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2019 Daniel Santiago <dpelaez@cicese.edu.mx>
#
# Distributed under terms of the GNU/GPL license.

"""

"""


import numpy as np
import netCDF4 as nc
import datetime as dt
import matplotlib.pyplot as plt
from src.processing.wdm.spectra import colorax
# plt.ion()


# load dataset
name = "http://apdrc.soest.hawaii.edu/dods/public_data/satellite_product/ASCAT/daily"
dataset = nc.Dataset(name, "r")

# load complete time, lon a lat
time = nc.num2date(dataset["time"][:].data, dataset["time"].units)
lat = dataset["lat"][:]
lon = dataset["lon"][:]

# perform a subset
# date = dt.datetime(2018,9,10)
# date = dt.datetime(2018,10,13)
date = dt.datetime(2018,9,12,12,24)
ixtime = np.argmin(abs(time - date))
ixlat = np.where(np.logical_and(lat>18, lat<30))[0]
ixlon = np.where(np.logical_and(lon>-100, lon<-90))[0]

# compute data at bomm location
bomm_lon, bomm_lat = -96.6245, 24.6028
ixlat_bomm = np.argmin(abs(lat[ixlat] - bomm_lat))
ixlon_bomm = np.argmin(abs(lon[ixlon] - bomm_lon))

# create figure canvas
fig, ax = plt.subplots(3,3, figsize=(10,10))
ax = np.ravel(ax)
#
for i in range(9):
    
    # download data at the subset
    wspd = dataset["wsp"][ixtime,ixlat,ixlon]
    uwnd = dataset["uwnd"][ixtime,ixlat,ixlon]
    vwnd = dataset["vwnd"][ixtime,ixlat,ixlon]

    # load data at bomm point
    wspd_bomm = wspd[ixlat_bomm, ixlon_bomm]
    uwnd_bomm = uwnd[ixlat_bomm, ixlon_bomm]
    vwnd_bomm = vwnd[ixlat_bomm, ixlon_bomm]
    wdir_bomm = np.arctan2(vwnd_bomm, uwnd_bomm) * 180/np.pi % 360

    # plot pcolor and quiver
    pc = ax[i].pcolormesh(lon[ixlon], lat[ixlat], wspd, vmin=0, vmax=20)
    ax[i].quiver(lon[ixlon], lat[ixlat], uwnd, vwnd, scale=500)
    ax[i].plot(bomm_lon, bomm_lat, "p", color="gold", mec="red")
    ax[i].set_title(f"{time[ixtime].strftime('%Y-%m-%d')}")

    # create label with data at bomm location
    s = f"$U_{{10}}={wspd_bomm:.2f}\;\mathrm{{m/s}}$\n" + \
        f"$\\theta_{{u}}={wdir_bomm:.2f}^\circ$"
    #
    bbox = {"boxstyle": "round", "fc": "white"}
    arrowprops = {"arrowstyle": "->", "connectionstyle": "angle3,angleA=90,angleB=0"}
    #
    ax[i].annotate(s, (bomm_lon, bomm_lat), (bomm_lon-3, bomm_lat-6),
        bbox=bbox, arrowprops=arrowprops)

    # nex time
    ixtime +=1

# place colorbar to the last one subplot
cax = colorax(ax[5])
fig.colorbar(pc, cax=cax)
fig.tight_layout()
fig.savefig(f"wind_{date.strftime('%Y%m%d')}.png", dpi=600)

