#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2019 Daniel Santiago <dpelaez@cicese.edu.mx>
#
# Distributed under terms of the GNU/GPL license.


import numpy as np
import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.gridspec as gridspec
#
plt.ion()

# load data {{{
def load_data_from_url(buoy, date):
    
    yy, mm, dd = date.strftime("%y"), date.strftime("%m"), date.strftime("%d")
    base_url = f"http://cigom-boyas.cicese.mx/data/{buoy.upper()}"
    fname = f"{base_url}/{yy}/{mm}/{buoy}-{yy}{mm}{dd}.csv"

    dateparse = lambda x: pd.datetime.strptime(x, "%y%m%d%H")
    df = pd.read_csv(
            fname,
            header=None,
            names={k:k for k in range(60)},
            parse_dates=[0],
            date_parser=dateparse
            )
    return df.set_index(0)
# }}}

# append data frame {{{
def append_dataframe(buoy, t_ini, t_fin):
    
    df = pd.DataFrame()
    date = t_ini
    while date <= t_fin:
        
        try:
            df = df.append(load_data_from_url(buoy, date))
        except:
            pass

        date += dt.timedelta(days=1)

    return df
# }}}

# generate plot {{{
def generate_plot(df, column, name, limits=None, *args, **kwargs):

    fig = plt.figure(figsize=(6.2, 3.0))
    ax = fig.add_subplot(111)

    ax.plot(df[column], *args, **kwargs)
    ax.set_xlabel("Tiempo [UTC]")
    ax.set_ylabel(name)
    ax.set_xlim((t_ini, t_fin))
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=12))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    if limits:
        ax.set_ylim(limits)

    return fig, ax
# }}}

# generate sub-plot {{{
def generate_subplot(df, t_ini, t_fin, folder):

    fig = plt.figure(figsize=(6.2, 5.0))
    gs = gridspec.GridSpec(3,1, bottom=0.1, top=.95)
    ax1 = plt.subplot(gs[0,0])
    ax2 = plt.subplot(gs[1,0])
    ax3 = plt.subplot(gs[2,0])

    ax1.set_title(folder.upper().replace("_", " "))
    ax1.plot(df[4], c="k")
    ax1.set_ylabel("Rapidez del\nviento [m/s]")
    #
    ax1b = ax1.twinx()
    ax1b.plot(df[5][::6]%360, ".", c="0.5")
    ax1b.set_ylabel("Dirección del\nviento [$^\circ{}$]")
    ax1b.set_ylim((0,360))
    ax1b.set_yticks((0,90,180,270,360))
    ax1.set_zorder(ax1b.get_zorder()+1)
    ax1.patch.set_visible(False)

    ax2.plot(df[8], color="steelblue", label="$T_\mathrm{aire}$")
    ax2.plot(df[13], color="darkred", label="$T_\mathrm{agua}$")
    ax2.legend(loc=0, ncol=2)
    ax2.set_ylabel("Temperatura [$^\circ{}$C]")

    ax3.plot(df[6], c="k")
    ax3.set_ylabel("Presión atm. [hPa]")
    #
    ax3b = ax3.twinx()
    ax3b.plot(df[7], c="0.5")
    ax3b.set_ylabel("Humedad\nrelativa [\%]")
    ax3b.set_ylim((-10,110))

    for ax in (ax1, ax2):
        ax.set_xticklabels([])
        ax.set_xlim((t_ini, t_fin))
        ax.xaxis.set_major_locator(mdates.DayLocator(interval=12))

    ax3.set_xlabel("Tiempo [UTC]")
    ax3.set_xlim((t_ini, t_fin))
    ax3.xaxis.set_major_locator(mdates.DayLocator(interval=12))
    ax3.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    fig.align_ylabels((ax1,ax2,ax3))

    return fig
# }}}

# make figures {{{
def make_figures(df, folder):

    if False:

        name = "Temperatura [$^\circ{}C$]"
        fig, ax = generate_plot(df, [8,13], name, None)
        plt.legend(("$T_a$", "$T_w$"))
        fig.savefig(folder + "/temperatura.png", dpi=600)

        name = "Presión atmósferica [hPa]"
        fig, ax = generate_plot(df, 6, name, None)
        fig.savefig(folder + "/presion_atmosferica.png", dpi=600)

        name = "Rapidez del viento [m/s]"
        fig, ax = generate_plot(df, 4, name, None)
        fig.savefig(folder + "/rapidez_viento.png", dpi=600)

        name = "Dirección del viento [$^\circ{}$C]"
        fig, ax = generate_plot(df, 5, name, None, ".")
        fig.savefig(folder + "/direccion_viento.png", dpi=600)

        name = "Humedad relativa [\%]"
        fig, ax = generate_plot(df, 7, name, None)
        fig.savefig(folder + "/humedad_relativa.png", dpi=600)
    
    # subplot
    fig = generate_subplot(df, t_ini, t_fin, folder=folder)
    fig.savefig(folder + "/main_variables.png", dpi=600)


# }}}


if __name__ == "__main__":

    t_ini = dt.datetime(2018,7,20)
    t_fin = dt.datetime(2018,9,25)
    df = append_dataframe("BOC9", t_ini, t_fin)
    make_figures(df, folder="boc9_lgv")
    

    t_ini = dt.datetime(2018,12,13)
    t_fin = dt.datetime(2019,2,12)
    df = append_dataframe("BOC10", t_ini, t_fin)
    make_figures(df, folder="boc10_lgv")


    t_ini = dt.datetime(2018,12,13)
    t_fin = dt.datetime(2019,2,12)
    df = append_dataframe("BOC2", t_ini, t_fin)
    make_figures(df, folder="boc2_ant")
