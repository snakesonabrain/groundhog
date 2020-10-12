.. py_pkg documentation master file. You can adapt this file completely
   to your liking, but it should at least contain the root `toctree` 
   directive.

Welcome to GroundHog Geotechnical Libraries documentation!
====================================================================

This Python package contains useful functionality for supporting automated geotechnical calculations.

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

Tutorials
----------

.. toctree::
   :maxdepth: 2

   tutorials/soilprofiles

   tutorials/pcpt_processing

   tutorials/pile_calculation


Function documentation
-----------------------

.. toctree::
   :maxdepth: 2

   general/general_toplevel

   classification/classification_toplevel

   site_investigation/site_investigation_toplevel

   piles/piles_toplevel

   shallowfoundations/shallowfoundations_toplevel

   tunnels/tunnels_toplevel


Acknowledgements
-----------------

The code for the validation of function input has been adopted from the python-engineering library and was integrated
in the package to reduce the amount of dependencies.


License and usage restrictions
-------------------------------

This package is distributed under a Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International
(`CC BY-NC-SA 4.0 <https://creativecommons.org/licenses/by-nc-sa/4.0/>`_) license.
Any commercial usage is strictly prohibited. Contact the authors for other licensing options if you plan to use the package
for commercial work.

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

