#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2018 Daniel Santiago <dpelaez@cicese.edu.mx>
#
# Distributed under terms of the GNU/GPL license.

"""

"""

import glob
import csv

number_of_columns = 9
path = "/Volumes/BOMM/BOMM1-ITS/data/level0/anemometro/"
list_of_files =  glob.glob(path + "**/*.csv", recursive=True)


def fix_number_of_columns(fname, line, n):
    """Read the filename, the linenumber and the number of columns."""
    
    with open(fname, "r") as f:
        data = f.readlines()

    nans = ",NaN,NaN,NaN,NaN,NaN\n"
    row = ",".join(data[line].split(",")[:4]) + nans
    
    # replace with new data
    data[line] = row
    with open(fname, "w") as f:
        f.write("".join(data))


if __name__ == "__main__":

    # loop for each file
    outdata = []
    for fname in list_of_files:
        #
        # check for number of columns
        with open(fname, "r") as f:
            data = csv.reader(f, delimiter=',')
            #
            # loop for each row
            for irow, row in enumerate(data):
                if len(row) != number_of_columns:
                    #
                    # create file with data
                    outdata += [[fname, irow, len(row)]]
                    print(fname, irow, len(row))
    
    # perform correction for each file
    if True:
        for fname, line, n in outdata:
            fix_number_of_columns(fname, line, n)

# --- eof ---
