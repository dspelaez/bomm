#! /bin/sh
#
# make.sh
# Copyright (C) 2019 Daniel Santiago <dpelaez@cicese.edu.mx>
#
# Distributed under terms of the GNU/GPL license.
#

python src/write_level1.py "./metadata/bomm1_its.yml"
python src/write_level2.py "./metadata/bomm1_its.yml" > ./log/bomm1_its_level2.log
