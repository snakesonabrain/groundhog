#!/usr/bin/env python3

import os
from setuptools import setup, find_namespace_packages
from groundhog.__version__ import __version__, __description__, __url__, __download_url__

# load the README file and use it as the long_description for PyPI
def readme():
    with open('README.md', 'r') as f:
        return f.read()

def required_packages():
      with open('requirements.txt') as f:
            required_packages = f.read().splitlines()
      return required_packages

# package configuration - for reference see:
# https://setuptools.readthedocs.io/en/latest/setuptools.html#id9
setup(
      name='groundhog',
      version=__version__,
      description=__description__,
      long_description=readme(),
      url=__url__,
      download_url=__download_url__,
      keywords=['engineering', 'geotechnical'],
      author='Bruno Stuyts',
      author_email='bruno@pro-found.be',
      license='GNU GPLv3',
      packages=find_namespace_packages(),
      include_package_data=True,
      zip_safe=False,
      classifiers=[
            "Programming Language :: Python :: 3",
            "Operating System :: OS Independent",
      ],
      python_requires='>=3.7',
      install_requires=required_packages(),
      )