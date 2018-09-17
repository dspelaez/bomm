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
File:     setup.py
Created:  2018-06-13 16:46
"""

from setuptools import setup, find_packages

setup(name = 'bomm',
      version = '0.1',
      description = 'Program to handle BOMM data',
      url = 'https://github.com/dspelaez/bomm',
      author = 'Daniel Santiago',
      author_email = 'dspelaez@gmail.com',
      license = 'GNU',
      packages = find_packages(),
      install_requires = [
          "numpy==1.14.3",
          "pandas==0.23.0",
          "PyYAML==3.13",
          "tqdm==4.25.0"
          ],
      zip_safe = False)
