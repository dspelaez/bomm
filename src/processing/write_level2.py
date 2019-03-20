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
import yaml
import sys
#
from processing_data import ProcessingData, rbr_data_correction
from read_raw_data import ReadRawData


# get the data from level 1 {{{
def get_data(date, number_of_minutes=30):
    """This function gets the data from the level 1 netcdf files."""
    p = ProcessingData(metafile, number_of_minutes)
    p.run(date)
    return p.r
    
# }}}

# write variables {{{
def write_variables(metadata, dataset):
    """Create NetCDF variables and attributtes for the given grp."""

    # get variables info from the metadata
    variables = metadata["processed_variables"]
    
    # compute number of times
    t_ini = dt.datetime.strptime(metadata["t_ini"], "%Y-%m-%d")
    t_fin = dt.datetime.strptime(metadata["t_fin"], "%Y-%m-%d")
    t_fin += dt.timedelta(hours=24)
    #
    samples_per_day = int(24 * 60 / number_of_minutes)
    #
    ntime = int((t_fin - t_ini).days * samples_per_day)

    # create time variable
    global global_time_units
    global_time_units = f"seconds since 1970-01-01 00:00:00"
    dataset.createDimension("time", ntime)
    nctime = dataset.createVariable("time", "f8", "time", fill_value=False)
    nctime.setncattr("units", global_time_units)
    nctime.setncattr("calendar", "gregorian")

    # create other dimensions
    other_dimensions = variables["dimensions"]
    #
    for name_dim, num_dim in other_dimensions.items():
        dataset.createDimension(name_dim, num_dim)
        dataset.createVariable(name_dim, "f8", name_dim, fill_value=False)

    # create each variable
    for k, v in variables.items():
        #
        if k not in ["dimensions"]:
            #
            # create variables with its dimensions
            dim = [x.strip() for x in v["dimensions"].split(',')]
            var = dataset.createVariable(k, "f8", dim, fill_value=np.nan)
            #
            # write variable attributes
            for attr, val in v.items():
                if k not in ["dimensions"]:
                    var.setncattr(attr, val)

# }}}

# write netcdf {{{
def write_netcdf(metafile, number_of_minutes=30):

    """This function writes a NetCDF4 file from the BOMM data at level 2.

    Args:
        metafile (str): Name of the metadata YAML file.
    """

    # read metadata file
    with open(metafile, "r") as f:
        metadata = yaml.load(f)

    # get common variables from metadata
    basepath = metadata["basepath"]
    bomm_name = metadata["name"]

    # starting and final days
    t_ini = dt.datetime.strptime(metadata["t_ini"], "%Y-%m-%d")
    t_fin = dt.datetime.strptime(metadata["t_fin"], "%Y-%m-%d")
    t_fin += dt.timedelta(hours=24)
    
    # create netcdf filename and open the file
    nm = int(number_of_minutes)
    fname = f"{basepath}/{bomm_name}/level2/{bomm_name}_level2_{nm}min.nc"
    with nc.Dataset(fname, "w") as dataset:

        # write global attrs
        _exclude = ["name", "basepath", "sensors", "t_ini", "t_fin",
                    "processed_variables"] 
        for gbl_name, gbl_values in metadata.items():
            if gbl_name not in _exclude:
                dataset.setncattr(gbl_name, gbl_values)

        # create variables
        write_variables(metadata, dataset)

        # loop for each day
        i = 0
        date = t_ini
        while date < t_fin:
            #
            print(f"Creating file corresponding to {date}\r")
            #
            # add time
            dataset["time"][i] = nc.date2num(date, global_time_units)
            #
            # get dictionary with data
            result = get_data(date, number_of_minutes)
            #
            # add the rest of the variables
            # for k in metadata["processed_variables"].keys():
            for k, v in result.items():
                try:
                    dataset[k][i,...] = v
                except:
                    pass
            #
            # update date
            date += dt.timedelta(minutes=number_of_minutes) 
            i += 1
        #
        # add the other dimensions
        # for dimensions in metadata add result[dimension]
        result = get_data(t_ini) # almost ever, the first date have all data.
        for k in metadata["processed_variables"]["dimensions"].keys():
            dataset[k][:] = result[k]

    # correction of processed data
    if True:
        rbr_data_correction(fname)
        # wstaff_data_correction(fname)
# }}}



if __name__ == "__main__":
    
    # execute the code if the valid arg is passed
    if len(sys.argv) > 1:
    
        try:
            # read metadata file
            metafile = sys.argv[1]

            # read number of minutes
            try:
                number_of_minutes = int(sys.argv[2])
            except:
                number_of_minutes = 30
            
            # write netcdf file for the given number of minutes
            write_netcdf(metafile, number_of_minutes)

        except:
            raise Exception("An error was occurred")
    else:
        raise ValueError("Invalid number of arguments")


# === end of file ===

