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
import src.processing.wdm.spectra as spc
import calendar
import ftplib
import os
plt.ioff()


# download file from ifremer ftp server
def download_from_ftp(filename):
    """Copy file from FTP to local directory"""

    local_path = "/Volumes/Boyas/iowaga/"
    
    with ftplib.FTP("ftp.ifremer.fr") as ftp:
        
        # login the ftp
        ftp.login()
        ftp.cwd("ifremer/ww3/HINDCAST/")

        # create local folder if it does not exist
        folder = os.path.join(local_path, os.path.split(filename)[0])
        if not os.path.exists(folder):
            os.makedirs(folder)
        
        print(f"Downloading from FTP: {filename}")
        with open(os.path.join(local_path, filename), "wb") as f:
            cmd = f"RETR {filename}"
            try:
                ftp.retrbinary(cmd, f.write)
                return True
            except:
                print("File couldn't be downloaded")
                return False


# load dataset
def load_dataset(year, month, region="GLOBAL", wind_source="ECMWF"):
    """Returns all the netcdf datasets por the specific region"""

    local_path = "/Volumes/Boyas/iowaga/"
    basename = f"{region}/{year}_{wind_source}/partitions"

    if region == "GLOBAL":
        grid = "30M"
        rcode = "GLOB"
    else:
        grid = "10M"

    dic = {}
    list_of_variables = ["phs", "pdir", "ptp"]
    for variable in list_of_variables:
        for partition in range(6):
            
            # create filename
            parameter = f"{variable}{partition}"
            filename = f"{basename}/WW3-{rcode}-{grid}_{year}{month:02d}_{parameter}.nc"
            
            # check if local filename exist
            local_filename = os.path.join(local_path, filename)
            if os.path.exists(local_filename):
                dataset = nc.Dataset(local_filename, "r")
            #
            # if not, download it from ftp server
            else:
                result = download_from_ftp(filename)
                if result:
                    dataset = nc.Dataset(local_filename, "r")

            # store in a dictionary
            dic[parameter] = dataset

    return dic


# extract data in a specific point
def extract_point(dic, date, lon=-96.6245, lat=24.6028):
    """Extract the partition parameters in a specific point"""

    results = {}
    for parameter, dataset in dic.items():
        
        # choose the interest variables
        time = nc.num2date(dataset["time"][:].data, dataset["time"].units)
        glat = dataset["latitude"][:]
        glon = dataset["longitude"][:]
        
        # find the indices of the specific pont
        ixtime = np.argmin(abs(time - date))
        ixlat = np.argmin(abs(glat - lat))
        ixlon = np.argmin(abs(glon - lon))

        # load the variables in a pont
        value = np.float32(dataset[parameter][ixtime, ixlat, ixlon])
        results[parameter] = value

    # save lon, lat and time
    results["lon"] = glon[ixlon]
    results["lat"] = glat[ixlat]
    results["time"] = time[ixtime]

    return results


def get_spectrum(results):
    """Plot a single spectrum for the parameters passed in results"""

    # create spectrum using a sech function
    frqs = np.arange(0.01, 1, 0.01)
    dirs = np.arange(360)
    E = np.zeros((len(dirs), len(frqs)))
    #
    for partition in range(6):
        Hs = results[f"phs{partition}"]
        Tp = results[f"ptp{partition}"]
        pdir = (270-results[f"pdir{partition}"]) % 360
        #
        if Hs != 0:
            E += spc.dirspec(frqs, dirs, Hs, Tp, pdir, func="sech2")

    return frqs, dirs, E


def plot_spectrum(date, lon, lat):
    """Create a 3x3 plot of spectra each 3 hours."""

    # create canvas
    fig, ax = plt.subplots(3, 3, figsize=(9,9))
    fig.subplots_adjust(top=.95, bottom=.05, left=.05, right=.95, wspace=.1, hspace=.1)
    ax = np.ravel(ax)

    # load data
    dic = load_dataset(date.year, date.month, region="GLOBAL")
    #
    for i in range(9):
        #
        results = extract_point(dic, date, lon=lon, lat=lat)
        frqs, dirs, E = get_spectrum(results)
        #
        if i in [2,5,8]:
            spc.polar_spectrum(frqs, dirs, E, smin=-2., smax=2., fmax=0.5, 
                               label=True, cbar=True, ax=ax[i])
        else:
            spc.polar_spectrum(frqs, dirs, E, smin=-2., smax=2., fmax=0.5, 
                               label=True, cbar=False, ax=ax[i])
        
        # set title
        title = date.strftime("%Y-%m-%d %H:%M:%S")
        ax[i].set_title(title)
        
        # tune up axes
        if i not in [6,7,8]:
            ax[i].set_xlabel("")
            ax[i].set_xticklabels([''])
        #
        if i not in [0,3,6]:
            ax[i].set_ylabel("")
            ax[i].set_yticklabels([''])
        
        # next spectrum
        date += dt.timedelta(hours=3)

    return fig, ax


def plot_comparison(bomm, date, lat, lon):
    """"Direct comparison of iowaga and bomm spectrum for a specific date."""

    # bomm-name
    # bomm = "bomm1_per1"
    # lon, lat = -96.6245, 24.6028
    # bomm = "bomm1_its"
    # lon, lat = -116.83, 31.82

    # load bomm dataset
    path = f"/Volumes/Boyas/bomm_database/data/{bomm}/level2/"
    dataset = nc.Dataset(path + f"{bomm}_level2_30min.nc")
    time = nc.num2date(dataset["time"][:].data, dataset["time"].units)
    #
    # find closest match
    i = np.argmin(abs(time - date))
    E_b = dataset["E"][i,:,:]
    frqs_b, dirs_b, = dataset["wfrq"][:], dataset["dirs"][:],

    # load iowaga data
    dic = load_dataset(date.year, date.month, region="GLOBAL")
    results = extract_point(dic, date, lon, lat)
    frqs_i, dirs_i, E_i = get_spectrum(results)

    # make plot
    fig, ax = plt.subplots(1, 2, figsize=(8,4))
    fig.subplots_adjust(top=.9, bottom=.1, left=.1, right=.9, wspace=.1, hspace=.1)
    #
    ax[0].set_title(date.strftime("%Y-%m-%d %H:%M:%S"))
    ax[1].set_title(results["time"].strftime("%Y-%m-%d %H:%M:%S"))
    #
    spc.polar_spectrum(frqs_b, dirs_b, E_b, smin=-3., smax=2., fmax=0.5, 
                       label=True, cbar=False, ax=ax[0])
    #
    spc.polar_spectrum(frqs_i, dirs_i, E_i, smin=-3., smax=2., fmax=0.5, 
                       label=True, cbar=True, ax=ax[1])

    # plot thingies
    Ua, Va = dataset["Ua"][i], dataset["Va"][i]
    qva = ax[0].quiver(0, 0, Ua, Va, scale=30, color="blue")
    ax[0].quiverkey(qva, 0.85, 0.85, 5.00, label="$U_{10} = 5\,\mathrm{m/s}$")
    ax[1].set_ylabel("")
    
    # savefigure
    # if savefig:
        # fname = f"./events/{date.strftime('%Y%m%d%H')}.png"
        # fig.savefig(fname, dpi=100)



if __name__ == "__main__":


    if False:
        #
        bomm = "bomm1_its"
        date = dt.datetime(2017, 12, 21, 13, 0)
        lon, lat =-116.83, 31.82
        plot_comparison(bomm, date, lat, lon)
        plot_spectrum(date, lon, lat)

    
    if False:
        #
        year = 2018
        for month in [9,10,11,12]:
            #
            dic = load_dataset(year, month, region="GLOBAL", wind_source="ECMWF")
            for day in range(1, calendar.monthrange(year, month)[1]+1):
                #
                date = dt.datetime(year, month, day)
                fig, ax = plot_spectrum(date)
                fig.savefig(f"./iowaga_spectra/{date.strftime('%Y%m%d')}.png", dpi=100)
                plt.close(fig)
                print(f"File corresponding to {date.strftime('%Y-%m-%d')} saved")
            

    if False:
        #
        start_date = dt.datetime(2018, 11, 9)
        final_date = dt.datetime(2018, 11, 17)
        date = start_date
        while date <= final_date:
            print("Plotting ", date)
            plot_comparison(date)
            plt.close("all")
            date += dt.timedelta(hours=3)
