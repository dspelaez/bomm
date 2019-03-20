#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2019 Daniel Santiago <dpelaez@cicese.edu.mx>
#
# Distributed under terms of the GNU/GPL license.

"""
This script plots a simple rose for both wave and wind direction.
"""


import numpy as np
import netCDF4 as nc
import matplotlib.pyplot as plt
plt.ion()



# wind rose {{{
def windrose(wspd, wdir, rbins):
    
    # remove nans if np.ma instance
    ix = np.logical_or(wspd.mask, wdir.mask)
    wspd = wspd.data[~ix]
    wdir = wdir.data[~ix]

    # histogram of directions
    bins = (rbins, range(0,360+15,15))
    H, wspd_edges, wdir_edges = np.histogram2d(wspd, wdir, bins)
    H = 100 * H / len(wspd)
    wspd_center = wspd_edges[1:] - 0.5 * (wspd_edges[1]-wspd_edges[0])
    wdir_center = wdir_edges[1:] - 0.5 * (wdir_edges[1]-wdir_edges[0])
    N = len(wspd_center)
    f = np.pi/180
    wdth = 0.90 * wdir_edges[1] * f
    
    # colors
    colors = plt.cm.get_cmap('jet', N)

    # figure hanldlers
    fig, ax = plt.subplots(1,1, figsize=(6,6))
    axp = fig.add_axes(ax.get_position(), frameon=False, polar=True)

    b = [0] * N
    b[0] = axp.bar(wdir_center*f, H[0], width=wdth, color=colors(0))
    for i in range(N-1):
        b[i+1] = axp.bar(wdir_center*f, H[i+1], bottom=sum(H[:i+1]), 
                width=wdth, color=colors(i+1), edgecolor="k")
    
    # plot legend
    l = [f"{int(wspd_edges[i])}-{int(wspd_edges[i+1])} m/s" for i in range(N)]
    axp.legend(b, l, loc=3, ncol=1, facecolor="w", frameon=True)

    # remove bg axes ticks
    ax.set_xticks([])
    ax.set_yticks([])

    text_kw = {"transform":ax.transAxes, "ha":"center", "va":"center"}
    bbox = {"fc":"w", "ec":"w"}
    axp.text(0.50, 0.95, "N", **text_kw, bbox=bbox)
    axp.text(0.95, 0.50, "E", **text_kw, bbox=bbox)
    axp.text(0.50, 0.05, "S", **text_kw, bbox=bbox)
    axp.text(0.05, 0.50, "W", **text_kw, bbox=bbox)
    
    # tune polar axes
    axp.grid(True, color="0.5", lw=.5)
    # axp.set_theta_zero_location("N")
    # axp.set_theta_direction(-1)
    axp.set_xticklabels([])

    # modify rlabels
    # rlabel = axp.get_yticks()
    # axp.set_rlim((0,np.ceil(np.max(rlabel))))
    axp.set_rlim((0, np.ceil(np.max(rbins))))
    axp.set_rlabel_position(135)

    return fig, ax, axp
# }}}


if __name__ == "__main__":
    
    # load data
    dataset = nc.Dataset("../../data/bomm1_its/level2/bomm1_its.level2.nc")
    
    # wind speed and direction
    wspd = dataset["U10N"][:]
    wdir = (270 - dataset["tWdir"][:]) % 360
    fig, ax, axp = windrose(wspd, wdir, [0,2,4,6,8,10,12,14,16,18])

    # wave height and direction
    wspd = dataset["Hm0"][:]
    wspd.mask[wspd.data>10] = True
    wdir = dataset["pDir"][:] % 360
    fig, ax, axp = windrose(wspd, wdir, [0,1,2,3,4,5,6,7,8])
