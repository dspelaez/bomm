#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2018 Daniel Santiago <dpelaez@cicese.edu.mx>
#
# Distributed under terms of the GNU/GPL license.

"""
This module contains utils functions to handle BOMM data
"""

# import libs
import netCDF4 as nc

# extract data from netcdf for a given date {{{
def get_data(grp, date, number_of_minutes=30, only=None):
    """Return the data corresponding to the netCDF group for a specific date.
    
    Args: TODO
    Returns:
        Dictionary containig all variables.
    
    """

    # start and final indices
    fs = grp.sampling_frequency
    N = int(fs * 24 * 3600)
    hour, minute = date.hour, date.minute

    # check number of minutes
    if number_of_minutes > N:
        raise ValueError(f"Number of minutes must be less than a day. Max={N}.")

    # start and final index
    i = int(fs*hour*3600 + fs*minute*60)
    j = i + int(fs*number_of_minutes*60)

    dic = {}
    dic["time"] = nc.num2date(grp["time"][i:j], grp["time"].units)
    if only:
        for k in only:
            dic[k] = grp[k][i:j]
    else:
        for k in grp.variables.keys():
            if k not in ["time"]:
                dic[k] = grp[k][i:j]

    return dic
# }}}
