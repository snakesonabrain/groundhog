.. py_pkg documentation master file. You can adapt this file completely
   to your liking, but it should at least contain the root `toctree` 
   directive.

Welcome to GroundHog Geotechnical Libraries documentation!
====================================================================

.. image:: https://badge.fury.io/py/groundhog.svg
    :target: https://badge.fury.io/py/groundhog

.. image:: tutorials/images/groundhog_banner_wide.png
   :width: 800

This Python package contains useful functionality for supporting automated geotechnical calculations.

``groundhog`` is first and foremost aimed at education on geotechnical engineering automation with Python. It aims to support students and educators with well-developed examples of geotechnical analysis in Python.

Functionality for onshore and offshore geotechnical problems is included. This package is under constant
development so any request for additional functionality can always be submitted to the package author.

The package is developed around four pilars:

- Flexible input parameter validation: Predefined parameter ranges are defined for most functions, based on the range of soil parameters for which the function was originally developed. This validation can be overridden by the user but requires explicit definition of the modified parameter ranges;
- Multiple outputs: Groundhog functions return a Python dictionary including intermediate results or derived quantities;
- Data standardisation: Possibility to read multiple input file formats (e.g. CPT data);
- Soil profiles: Easy encoding and manipulation of soil profiles.


The package was named after the `groundhog <https://en.wikipedia.org/wiki/Groundhog/>`_, an animal
that lives in underground burrows. Moreover, the movie `Groundhog Day <https://en.wikipedia.org/wiki/Groundhog_Day_(film)/>`_ where
a reporter relives the same day again and again. The groundhog package aims to remove this repetitiveness
from your day-to-day geotechnical engineering work.


Installation requirements
-------------------------

groundhog is written for Python 3.7+. Downloading Anaconda3 is recommended for users not familiar with Python development.
Plotting functionality included in the package is built on plotly. The plotly package also needs to be installed and
is not included in the default Anaconda installation.

Installation is easily done using pip:

.. code-block:: bash

   pip install groundhog

A more comprehensive Getting Started section with detailed installation instructions is included in the documentation.

.. toctree::

   gettingstarted

Support groundhog
------------------

A lot of time and effort has gone into the creation and maintenance of groundhog.
You can support the development of groundhog by buying me a coffee.

.. raw:: html

   <a href="https://www.buymeacoffee.com/groundhog"><img src="https://img.buymeacoffee.com/button-api/?text=Buy me a coffee&emoji=&slug=groundhog&button_colour=FFDD00&font_colour=000000&font_family=Cookie&outline_colour=000000&coffee_colour=ffffff"></a>


Training, writing of non-shared company-specific software built on groundhog and consultancy are additional revenue streams for the package. Get in touch if you are interested in these services.


Tutorials
----------

.. image:: https://mybinder.org/badge_logo.svg
 :target: https://mybinder.org/v2/gh/snakesonabrain/groundhog/main

Tutorials are provided in the notebooks folder of the project. Jupyter notebooks are provided for the following examples:

- Basic use of groundhog functions
- Soil profile definition, manipulation and plotting
- PCPT data loading
- PCPT data processing
- Loading AGS data
- Axial pile capacity calculation according to Belgian practice
- ...


Function documentation
-----------------------

Detailed documentation is available on the functions and classes of the package.
This documentation is essential for using the functions in ``groundhog`` correctly as the documentation
specifies the physical meaning and the units of input and output variables.

.. toctree::
   :maxdepth: 2

   general/general_toplevel

   site_investigation/site_investigation_toplevel

   piles/piles_toplevel

   shallowfoundations/shallowfoundations_toplevel

   consolidation/consolidation_toplevel

   excavations/excavations_toplevel

   soildynamics/soildynamics_toplevel

   standards/standards_toplevel

   constitutivemodels/constitutivemodels_toplevel


Acknowledgements
-----------------

The code for the validation of function input has been adopted from the python-engineering library and was integrated
in the package to reduce the amount of dependencies.


License and usage restrictions
-------------------------------

groundhog. A general-purpose Python library for geotechnical engineering.
    Copyright (C) 2020  Bruno Stuyts

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

