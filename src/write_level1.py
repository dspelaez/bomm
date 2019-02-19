#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
# vim:fdm=marker
#
# =============================================================
#  Copyright © 2018 Daniel Santiago <dpelaez@cicese.edu.mx>
#  Distributed under terms of the GNU/GPL license.
# =============================================================

"""
This module contains function to write a NetCDF4 file from the BOMM data.
"""

# --- import libs ---
import numpy as np
import datetime as dt
import netCDF4 as nc
import time
import yaml
import sys
import os
#
from read_raw_data import ReadRawData


# write variables {{{
def write_variables(metadata, sensor, grp):
    """Create NetCDF variables and attributtes for the given grp."""

    # get variables info from the metadata
    variables = metadata["sensors"][sensor]["variables"]

    # create time variable
    nctime = grp.createVariable("time", "f8", "time", fill_value=False)
    nctime.setncattr("units", global_time_units)
    nctime.setncattr("calendar", "gregorian")

    # create each variable
    for k, v in variables.items():
        #
        # check if variable is 2d or 1d
        if isinstance(v["column"], list):
            ncell = v["column"][1] - v["column"][0] + 1
            try:
                #
                # create number of cell dimension
                grp.createDimension("cell", ncell)
            except:
                pass
            #
            # create two dimensional variable 
            var = grp.createVariable(k, "f8", ("time", "cell"), fill_value=np.nan)
        else:
            #
            # create one dimensional variable 
            var = grp.createVariable(k, "f8", "time", fill_value=np.nan)
        #
        # write variable attributes
        for attr, val in v.items():
            if attr not in ["column"]:
                var.setncattr(attr, val)
# }}}

# write group {{{
def write_group(b, dataset, sensor, day, logfile):
    """Write variables and attributes associated to each group."""

    # create group related with the sensor and associate with time
    grp = dataset.createGroup(sensor)

    # get sampling frequency and seconds per file
    N  = b._getsecperfile(sensor)
    fs = b._getsampfreq(sensor)
    
    # compute the number of samples per day
    samples_per_day = int(fs * 60 * 60 * 24)
    samples_per_file = int(fs * N)

    # create time dimension and assing attributes
    grp.createDimension("time", samples_per_day)
    
    # write global attributes for each sensor
    for attr, val in b.metadata["sensors"][sensor].items():
        if attr not in ["variables", "seconds_per_file"]:
            grp.setncattr(attr, val)

    # create variables
    write_variables(b.metadata, sensor, grp)

    # loop for each data
    date, end = day, day + dt.timedelta(days=1, seconds=-N)
    i, j = 0, samples_per_file
    #
    while date <= end:

        # progress bar
        progress = 100 - 100 * (end-date).total_seconds() / (3600 * 24)
        sys.stdout.write(f" {sensor:10s}: {progress:5.1f}%  ---> {date}\r")
        sys.stdout.flush()

        # load data
        dic = b.read(sensor, date, logfile=open(logfile, "a"))

        # assign data to netctd variable
        for name, value in dic.items():

            # write variables
            if name not in ["time"]:

                # check if variables is 2d or 1d array
                if value.ndim == 1:
                    grp[name][i:j] = value
                else:
                    grp[name][i:j,:] = value

            # time variable
            else:
                grp["time"][i:j] = nc.date2num(value, global_time_units)
        
        # update counter
        i, j = j, j + samples_per_file

        # update date
        date += dt.timedelta(seconds=N) 
    
    # new line in the progress bar
    sys.stdout.write("\n")

# }}}

# write netcdf {{{
def write_netcdf(metafile):

    """This function writes a NetCDF4 file from the BOMM data.

    The function was writen to convert the BOMM raw data to a distributable
    NetCDF4 format. The functions only require the metatadata in a YAML file.
    This will write a NetCDF4 file for each day as specified in the YAML file.
    Each NetCDF4 file contains one group per buoy sensor and each group contains
    the variables specified in the YAML file as well as the metadata.

    Args:
        metafile (str): Name of the metadata YAML file.
    """

    # create instance of the bomm.ReadRowData class
    b = ReadRawData(metafile)

    # starting and final days
    day = dt.datetime.strptime(b.metadata["t_ini"], "%Y-%m-%d")
    end = dt.datetime.strptime(b.metadata["t_fin"], "%Y-%m-%d")

    # restart logfile if exist
    # logfile = os.path.splitext(metafile)[0] + "_level1.log"
    logfile = f"./log/{b.metadata['name']}_level1.log"
    with open(logfile, "w"):
        pass

    # global time units
    global global_time_units
    global_time_units = f"seconds since 1970-01-01 00:00:00"
    
    # loop for each day
    while day <= end:
        
        # netcdf filename
        fname = f"{b.basepath}/{b.bomm_name}/level1/{day.strftime('%Y%m%d')}.nc"

        # name of the group associated with each day
        with nc.Dataset(fname, "w") as dataset:

            # write global attrs
            _exclude = ["name", "basepath", "sensors", "t_ini", "t_fin",
                        "processed_variables"] 
            for gbl_name, gbl_values in b.metadata.items():
                if gbl_name not in _exclude:
                    dataset.setncattr(gbl_name, gbl_values)

            # write data for each sensor
            print("=" * 57, end="\n")
            sensors = b.metadata["sensors"].keys()
            for sensor in sensors:
                #
                # compute current time
                now = time.time()
                #
                # perform heavy task
                write_group(b, dataset, sensor, day, logfile)
                #
                # print elapsed time
                etime = time.time() - now
                print("-" * 46 + f" {etime/60:.2f} mins", end="\n")

        # update date
        day += dt.timedelta(days=1) 
# }}}



if __name__ == "__main__":
    
    # execute the code if the valid arg is passed
    if len(sys.argv) == 2:
        try:
            metafile = sys.argv[1]
            write_netcdf(metafile)
        except:
            raise Exception("An error was occurred")
    else:
        raise ValueError("Invalid number of arguments")




# === end of file ===

