#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2019 Daniel Santiago <dpelaez@cicese.edu.mx>
#
# Distributed under terms of the GNU/GPL license.

"""
Validation of the Euler angles from EKINOX using the SIGNATURE
"""


import numpy as np
import datetime as dt
import matplotlib.pyplot as plt
#
from spectra_animation import PlotSpectra
from src.processing import motion_correction as motcor

plt.ion()


# --- create instace of PlotSpectra class
t_ini = dt.datetime(2018, 7, 11)
t_end = dt.datetime(2018, 7, 13)
s = PlotSpectra("../../metadata/bomm1_per1.yml", t_ini, t_end, 10)
try:
    s.get_high_frequency_data(10)
except:
    s.get_high_frequency_data(10)


# --- collect signature data
stime = s.p.sig["time"]# - s.p.sig["time"][0]
sroll = s.p.sig["roll"]/100 * np.pi/180
spitch = s.p.sig["pitch"]/100 * np.pi/180
syaw = s.p.sig["heading"]/100 * np.pi/180
#
sphi = spitch + np.radians(20)
stheta = -sroll.copy()

# --- collect ekinox data
ax, ay, az = s.p.Acc
wx, wy, wz = s.p.Gyr
atime = s.p.ekx["time"]# - s.p.ekx["time"][0]

# sampling and merging frequencies
fs, fc = 100, 1/12.5

# -- create figures
for fm in (0.04, 0.08, 1):

    # tilt from accelerometers
    phi_acc = np.arctan2(ay, np.sqrt(ax**2 + az**2))
    theta_acc = np.arctan2(-ax, np.sqrt(ay**2 + az**2))

    # compute pitch and roll from gyrospcope
    get_angle = lambda x: motcor.fft_integration(x, fs=fs, fc=fc, order=-1)
    phi_gyr, theta_gyr = get_angle(wx), get_angle(wy)

    # complementary filter
    phi = motcor.complementary_filter(phi_acc, phi_gyr, fs, fm=fm)
    theta = motcor.complementary_filter(theta_acc, theta_gyr, fs, fm=fm)

    # crate figure
    fig, (ax1, ax2) = plt.subplots(2,1)

    # plot phi
    ax1.plot(stime, sphi, color="black", label="signature")
    ax1.plot(atime, phi_acc, color="0.7", label="acelerometro")
    ax1.plot(atime, phi_gyr, color="green", label="giroscopio")
    ax1.plot(atime, phi, color="magenta", lw=1.25, label="filtro compl.")

    # plot theta
    ax2.plot(stime, stheta, color="black", label="signature")
    ax2.plot(atime, theta_acc, color="0.7", label="acelerometro")
    ax2.plot(atime, theta_gyr, color="green", label="giroscopio")
    ax2.plot(atime, theta, color="magenta", lw=1.25, label="filtro compl.")

    ax1.legend(loc=1, ncol=4, bbox_to_anchor=(1,1.2))
    ax1.set_ylabel("$\phi$")
    ax2.set_ylabel("$\\theta$")
    # ax1.set_xlim((100,200))
    # ax2.set_xlim((100,200))
