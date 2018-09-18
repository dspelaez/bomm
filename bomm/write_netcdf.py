#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
# vim:fdm=marker
#
# =============================================================
#  Copyright © 2018 Daniel Santiago <dpelaez@cicese.edu.mx>
#  Distributed under terms of the GNU/GPL license.
# =============================================================

"""This module contains functions to write a NetCDF4 file from the BOMM data.

The module was writen to convert the BOMM row data to a distributable NetCDF4
format. The functions only require the metatadata in a YAML file. This will
write a NetCDF4 file for each day as specified in the YAML file. Each NetCDF4
file contains one group per buoy sensor and each group contains the variables
specified in the YAML file as well as the metadata.

Example:
    To run this module you just need the YAML filename as follows:
        
        >>> import bomm
        >>> metafile = "bomm.yml"
        >>> bomm.write_netcdf(metafile)

Author:
    Daniel Santiago <dpelaez@cicese.edu.mx>

"""

# --- import libs ---
import numpy as np
import datetime as dt
import netCDF4 as nc
import time
import yaml
import sys
#
import bomm


# write variables {{{
def write_variables(metadata, sensor, grp):
    """Create NetCDF variables and attributtes for the given grp."""

    # get variables info from the metadata
    variables = metadata["sensors"][sensor]["variables"]

    # create time variable
    grp.createVariable("time", "f8", "time")

    # create each variable
    for k, v in variables.items():
        #
        # check if variable is 2d or 1d
        if isinstance(v["column"], list):
            ncell = v["column"][1] - v["column"][0] + 1
            try:
                grp.createDimension("cell", ncell)
            except:
                pass
            var = grp.createVariable(k, "f8", ("time", "cell"))
        else:
            var = grp.createVariable(k, "f8", "time")
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

    # create time dimension
    grp.createDimension("time", samples_per_day)

    # write global attributes
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

    # create instance of the bomm.ReadRowData class
    b = bomm.ReadRawData(metafile)

    # starting and final days
    day = dt.datetime.strptime(b.metadata["date-start"], "%Y-%m-%d")
    end = dt.datetime.strptime(b.metadata["date-final"], "%Y-%m-%d")

    # restart logfile if exist
    logfile = b.metadata["name"] + ".log"
    with open(logfile, "w"):
        pass

    # global time units
    global global_time_units
    global_time_units = f"seconds since {day}"
    
    # loop for each day
    while day <= end:
        
        # netcdf filename
        fname = f"{b.basepath}/netcdf/bomm.level0.{day.strftime('%Y%m%d')}.nc"

        # name of the group associated with each day
        with nc.Dataset(fname, "w") as dataset:

            # write global attrs
            dataset.setncattr("name",       b.metadata["name"])
            dataset.setncattr("date-start", b.metadata["date-start"])
            dataset.setncattr("date-final", b.metadata["date-final"])

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

    # write netcdf
    write_netcdf(metafile="bomm.yml")


# === end of file ===

