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

path = "/Volumes/BOMM/BOMM2-ITS/data/level0/"
list_of_files =  glob.glob(path + "**/*.csv", recursive=True)


def remove_nullbytes(fname):
    """Remove NULL bytes in the CSV file."""

    with open(fname, "r") as f:
        data = f.read()

    if data.find("\x00") != -1:
        with open(fname, "w") as f:
            f.write(data.replace("\x00", ""))
        return 1
    else:
        return 0


if __name__ == "__main__":

    # loop for each file
    counter = 0
    for fname in list_of_files:

        # remove NULL bytes if exist
        status = remove_nullbytes(fname)
        if status == 1:
            print(fname)
            counter += status

    print(f"Files with NULL bytes: {counter} out of {len(list_of_files)}")

