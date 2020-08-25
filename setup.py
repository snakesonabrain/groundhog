#!/usr/bin/env python3

import os
from setuptools import setup, find_packages
from geotechnics.__version__ import __version__, __description__
# load the README file and use it as the long_description for PyPI
def readme():
    with open('README.md', 'r') as f:
        readme = f.read()

# package configuration - for reference see:
# https://setuptools.readthedocs.io/en/latest/setuptools.html#id9
setup(name='groundhog',
      version=__version__,
      description=__description__,
      long_description=readme(),
      url='https://github.ugent.be/labo-geotechniek/ugentgeotechnics',
      download_url='https://github.ugent.be/bstuyts/ugentgeotechnics/archive/master.zip',
      keywords=['engineering', 'geotechnical'],
      author='Bruno Stuyts',
      author_email='bruno@pro-found.be',
      license='Creative Commons BY-SA 4.0',
      packages=find_packages(),
      include_package_data=True,
      zip_safe=False,
      test_suite='nose.collector',
      tests_require=['nose'],)