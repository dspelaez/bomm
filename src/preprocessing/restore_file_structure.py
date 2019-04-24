#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2019 Daniel Santiago <dpelaez@cicese.edu.mx>
#
# Distributed under terms of the GNU/GPL license.

"""
This script restore the original filestructure
"""

import datetime as dt
import shutil
import os

list_of_sensors = [
        'acelerometro',
        'anemometro',
        'gps',
        'marvi',
        'maximet',
        'proceanus',
        'rbr',
        'signature',
        'vector',
        'wstaff'
        ]


basepath = "/Volumes/Boyas/bomm_database/data/"
bomm_name = "bomm3_its"

# t_ini = dt.datetime(2018, 8, 13, 18, 0)
t_ini = dt.datetime(2018,  9, 23, 21,  0)
t_fin = dt.datetime(2018, 11, 12,  7, 20)


def set_new_filename(sensor, date):
    """Returns filename based on date, sensor, and path."""
    
    if sensor in ["maximet"]:                      # <--- one-hour files
        fmt = f"/{sensor}/%Y/%m/%d/{sensor}-%y%m%d%H.csv" 
    #
    elif sensor in ["rbr", "proceanus"]:           # <--- one-day files
        fmt = f"/{sensor}/%Y/%m/{sensor}-%y%m%d.csv"
    #
    else:                                          # <--- one-minute files
        fmt = f"/{sensor}/%Y/%m/%d/%H/{sensor}-%y%m%d%H%M.csv"
    
    filename = basepath + bomm_name + \
               "/level0" + dt.datetime.strftime(date, fmt)
    return filename



def get_filename(sensor, date):
    """Returns filename based on date, sensor, and path."""
    
    # sepate location from bomm_name
    bomm, location = bomm_name.upper().split("_")

    if sensor in ["rbr", "proceanus"]:           # <--- one-day files
        fmt = f"%Y/%m/{sensor}/{location}_{bomm}_{sensor}-%y%m%d.csv"
    #
    else:                                          # <--- one-minute files
        fmt = f"%Y/%m/{sensor}/{location}_{bomm}_{sensor}-%y%m%d%H%M.csv"
    
    filename = f"/Volumes/Boyas/data/{location}/{bomm}/" + \
               dt.datetime.strftime(date, fmt)

    return filename


if __name__ == "__main__":
    
    # get current filename
    not_found = {s: [] for s in list_of_sensors}
    for s in list_of_sensors:
        
        date = t_ini
        while date <= t_fin:

            current_filename = get_filename(s, date)
            exists_file = os.path.isfile(current_filename)

            if exists_file:
                new_filename = set_new_filename(s, date)
                print(new_filename)

                # create folder
                destination_path = os.path.split(new_filename)[0]
                if not os.path.exists(destination_path):
                    os.system(f"mkdir -p {destination_path}")

                # copy the file
                shutil.copy(current_filename, new_filename)

            else:
                not_found[s] += [current_filename]

            date += dt.timedelta(minutes=10)
