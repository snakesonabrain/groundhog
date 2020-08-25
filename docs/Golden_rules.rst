Golden rules for groundhog functions
===============================================

groundhog is developed and maintained based on a set op 10 golden rules.
These rules are embedded in the groundhog function architecture
and should be followed by anyone wishing to contribute to the project. Most of these
rules are inherently followed if functions are constructed using the groundhog
function constructor.

1. Code once, code smart
-------------------------

The goal of groundhog is to capture a robust implementation of
the function under consideration. The function should be implemented such that
calibration are supplied as default arguments. This practice will allow users to
use the code in a variety of cases without having to make changes to the function
code.

Bad practice:
    .. code-block:: python

        def lateral_stress(vertical_stress):
            lateral_earthpressure_coefficient = 0.5
            return lateral_earthpressure_coefficient*vertical_stress

Good practice:
    .. code-block:: python

        def lateral_stress(vertical_stress, lateral_earthpressure_coefficient=0.5):
            return lateral_earthpressure_coefficient*vertical_stress

The docstring for the function should also be written with care and should contain as
a minimum:

    - A description of the calculation performed in the function
    - A description of possible caveats and pitfalls
    - A listing of all function arguments with description, units, validation ranges and defaults for optional arguments
    - A description of function outputs and their units
    - The return type of the output with a dictionary keys in case a dictionary is returned (this is the case for functions created with the groundhog function constructor)
    - The bibliographical reference to the work where the function is presented in detail

In addition to these elements, the following elements can also be supplied in the docstring:

    - Equation(s) implemented in the function
    - Figure(s) demonstrating the application of the function (See the Rule no. 10 to check which figures can be used)

2. Parameter validation, the key to robust engineering calculations
--------------------------------------------------------------------

In good engineering practice, correlations and calculation formulae are only used when
parameters are inside a certain range, usually corresponding to a well-defined calibration
range. groundhog includes parameter validation for float, integer, string, boolean
and list arguments.

groundhog implements a validation decorator which can be used to validate
parameters.

Bad practice:
    .. code-block:: python

        def my_correlation(percentage):
            if 0.0 < percentage < 100.0:
                return percentage
            else
                return None

Good practice:
    .. code-block:: python

        from groundhog.general.validation import Validator

        MY_CORRELATION = {
            'percentage':{'type': 'float','min_value':0.0,'max_value':100.0},
        }


        @Validator(MY_CORRELATION)
        def my_correlation(percentage):
            return percentage

groundhog has a tweak which allows the user to override the standard
validation parameters when calling the function. Adding an additional keyword argument
with ``__min`` or ``__max`` appended to the parameter name will override
the validation range.
For example, if calibration range is [40%, 80%], the function would be called as follows:

    .. code-block:: python

        >>>my_correlation(20.0, percentage__min=40.0, percentage__max=80.0)
        ValueError(percentage (20.0) cannot be smaller than 30.0)

groundhog allows the user to completely override parameter validation
through the ``validate`` keyword argument (``validate=False``). This
should always be used with caution and justified. Expanding validation ranges is
recommended over switching off validation.

    .. code-block:: python

        >>>my_correlation(120.0, validate=False)
        120.0

3. Units, units, units
-----------------------

The importance of using a function with correct units cannot be overstated. The unit of every
function argument needs to be documented, even if the argument is unitless. The units
of function output also need to be specified.

Bad practice:
    .. code-block:: python

        def weight(density, volume):
            return density * volume

Good practice:
    .. code-block:: python

        def weight(density, volume):
            """
            :param density: Density of the volume [kg/m3]
            :param volume: Volume of the solid body [m3]

            :rtype: Weight of the solid body [kg]

            """
            return density * volume

4. Verbosity: Providing readable code
--------------------------------------

groundhog uses verbose function names and function arguments to allow
any user to quickly examine code without having to verify which parameter name
corresponds to which physical quantity. When coding new functions,
verbose parameters

Bad practice:
    .. code-block:: python

        def w(rho, v):
            return rho * v

Good practice:
    .. code-block:: python

        def weight(density, volume):
            return density * volume

5. Function naming
-------------------

The function names need to follow a specific convention:

.. code-block:: python

        <main_output>_<main_input>_<author>

As indicated in rule 4, verbose function names should be used.
Function names should also be lowercase in accordance with PEP style guidelines.

Bad practice:
    .. code-block:: python

        def Ko_Dr_Bellotti(...):
            ...

Good practice:
    .. code-block:: python

        def lateralearthpressure_relativedensity_bellotti(...):
            ...

6. Output dictionaries, leveraging Python's awesomeness
-------------------------------------------------------

One of the really cool things about Python is that you can return dictionaries from
functions. This allows you to not only return scalar values or strings but any Python
object. For instance, if you want to return a dataframe from a gridded calculation,
you can just go ahead and do that. The dictionary associates a verbose name and unit
with an output so that end users always know which output they are working with. When
defining a function with many intermediate results, it is worthwhile returning these
intermediate results to allow checking of the calculations. This makes
groundhog functions behave less like a black box and more like a transparent
and reliable calculation tool.

Bad practice:
    .. code-block:: python

        def volume_cylinder(radius, height):
            area_base = np.pi*(radius**2.0)
            return area_base*height

Good practice:
    .. code-block:: python

        def volume_cylinder(radius, height):
            area_base = np.pi*(radius**2.0)
            return {
                'Base area [m2]': area_base,
                'Volume [m3]': area_base*height
            }

7. When all goes down the drain: Handling errors
-------------------------------------------------

groundhog functions have built in error handling which fails silently by
default. This means that if an error occurs during function execution (e.g. due to
a function argument falling outside validation ranges), ``np.nan`` is returned for
numerical values and ``None`` for strings, lists or dataframes. This prevents errors
from being raised all the time.

If the user to know the reason for errors, the keyword argument ``fail_silently`` can
be set to ``False``. This is illustrated in the example below.

    .. code-block:: python

        >>> from groundhog.correlations import sand
        >>>sand.lateralearthpressure_relativedensity_bellotti(-20.0)['Ko [-]']
        np.nan

        >>>sand.lateralearthpressure_relativedensity_bellotti(-20.0, fail_silently=False)['Ko [-]']
        ValueError(relative_density (-20.0) cannot be smaller than 20.0)


8. Unit testing, what could possibly go wrong?
-----------------------------------------------

Murphy's law applies, always, particularly in the case of software. Therefore all
groundhog functions have unit tests written against them. If you're not familiar
with unit testing, you should fill that gap in your knowledge fast! Unit testing ensures
that the function returns the expected value or that an error is raised when it should be
(e.g. when an argument is outside the validation range).

    .. code-block:: python

        import unittest
        from pyeng.geotechnical.correlations import sand

        class Test_lateralearthpressure_relativedensity_bellotti(unittest.TestCase):

            def test_values(self):
                self.assertAlmostEqual(sand.lateralearthpressure_relativedensity_bellotti(50.0)['Ko [-]'], 0.46, 2)

            def test_ranges(self):
                self.assertRaises(ValueError,sand.lateralearthpressure_relativedensity_bellotti(-50.0, fail_silently=False))

9. Sharing = Caring
--------------------

groundhog is provided to the engineering community under a Creative Commons 4.0
attribution share-alike license. Any works derived from groundhog or any functions
created using the groundhog function constructor should in turn be shared with the
community. Through the contribution of many engineers, groundhog can grow to
become an awesome tool allowing everyone, from student to grey-haired consultant to have
more confidence in their calcs.

10. Cool kids play fair
-----------------------------------

Under the terms of the Creative Commons 4.0 Attribution Share-alike license, groundhog
can be used for educational and commercial purposes. However, all derived works should be shared with the community.
Please consult the license terms for additional details. Please contact the authors if you wish to use the software
beyond the terms of the license agreement.

Playing fair allows knowledge to spread within the community.