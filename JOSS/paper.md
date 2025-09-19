---
title: 'Groundhog - A general-purpose geotechnical Python package'
tags:
  - Python
  - geotechnical
  - foundations
  - soil mechanics
authors:
  - name: Bruno Stuyts
    orcid: 0000-0002-1844-8634
    corresponding: true
    equal-contrib: true
    affiliation: "1, 2, 3" # (Multiple affiliations must be quoted)
affiliations:
 - name: Universiteit Gent (UGent), Belgium
   index: 1
 - name: Vrije Universiteit Brussel (VUB), Belgium
   index: 2
 - name: SolidGround ApS, Denmark
   index: 3
date: 06 January 2024
bibliography: paper.bib

# Optional fields if submitting to a AAS journal too, see this blog post:
# https://blog.joss.theoj.org/2018/12/a-new-collaboration-with-aas-publishing
# aas-doi: 10.3847/xxxxx <- update this with the DOI from AAS once you know it.
# aas-journal: Astrophysical Journal <- The name of the AAS journal.
---

# Summary

Geotechnical engineering is an engineering discipline in which the mechanical behaviour of soil and its impact on the performance of foundations is studied. Because of the complex constitutive behaviour of soil and the large variation in mineralogy, geological origins, stress conditions and foundation types, there is no generally applicable theory for soil behaviour and foundation design. Instead, myriad semi-empirical models and parameter correlations have been developed to describe the mechanical behaviour of soils and foundations [@Budhu:2010]. The Python package `Groundhog` was developed to allow the efficient use of these formulae in geotechnical engineering education, research and practice. 

# Statement of need

`Groundhog` is a Python package for geotechnical engineering calculations. The soil parameter correlations and foundation design models which are used in this discipline are scattered across various resources (textbooks, journal articles, recommended practices) and engineers spend a significant amount of time implementing them in calculation tools (typically Microsoft Excel). This is not an ideal situation, with a lot of repeated work and implementation mistakes often going undetected. Moreover, certain geotechnical calculations are calibrated for specific ranges of the input parameters. Checking that the inputs are within these ranges is often omitted in an Excel implementation.

To overcome these shortcomings, `Groundhog` provides a robust implementation of geotechnical functions with in-built parameter validation and [extensive documentation](https://groundhog.readthedocs.io/). Each input parameter is described, providing the expected units and the default validation range. The validation ranges can be overriden by the user with specific keyword arguments, making adjustments to the validation ranges explicitly visible.

Because geotechnical functions can return multiple outputs (e.g. intermediate calculation results), the output of `Groundhog` functions are Python dictionaries. The users can select the relevant outputs for their calculations by addressing the appropriate dictionary key. `Groundhog` functions are also unit-tested to ensure they return the expected results.

In addition to geotechnical functions, selected geotechnical workflows are encoded in an object-oriented manner. Processing of data from the cone penetration test (CPT) and the standard penetration tests (SPT) is a recurring task and the steps in the processing workflow are implemented in the `PCPTProcessing` and `SPTProcessing` classes respectively. The manipulation of stratigraphic profiles describing the various layers in the subsoil, is made possible with the `SoilProfile` class. Soil parameter visualisation and interactive parameter selection is made possible in the `LogPlot` class (using the `Plotly` [@plotly] plotting backend) and the `LogPlotMatplotlib` class (using `Matplotlib` [@Hunter:2007] as plotting backend).

The package implements various methods for basic foundation design taught in undergraduate and graduate course (e.g. shallow foundation capacity on sand and clay, axial pile resistance in sand clay, one-dimensional consolidation).

`Groundhog` is under continuous development and allows the geotechnical community to focus on a single robust and well-documented set of calculation tools.

# Applications in research

The geotechnical parameter correlations and geotechnical workflow automation tools have enabled research in offshore wind geotechnical engineering on the following topics:

- Natural frequency analysis of offshore wind turbine structures across an entire offshore wind farm [@fallais:2022]. `Groundhog` was used to define the stratigraphic profiles and perform the geotechnical parameter selection. The automation offered by `Groundhog` allowed all data to be processed in a reasonable time and with a focus on quality.
- Back-analysis of bending moment measurements on offshore wind turbine monopiles [@stuyts:2024]. `Groundhog` was used to define the stratigraphic profiles and perform the geotechnical parameter selection.
- Evaluation of CPT-based correlations for shear wave velocity estimation for North Sea soils [@stuytsshear:2024]. All correlations evaluated in the study were implemented in `Groundhog` and a newly developed correlation was shared with the community through implementation in the package.
- `Groundhog` is also being used in research on seismic inversion. Amplitude vs Offset (AVO) inversion allows inversion for bulk modulus, shear modulus and density. Groundtruth geotechnical data is processed with `Groundhog` and where direct measurements of the parameters are not available, the package is used to estimate them from other data sources.

# Applications in geotechnical engineering education

`Groundhog` is being used in undergraduate and graduate courses at Ghent University. The package allows students to spend more time on the parameter selection process and get more insight into the underlying data for the semi-empirical correlations. The effect of parameter variations can also easily be studied while reducing the time spent on implementing equations.

A dedicated two-day training course has also been set up to teach Python and `Groundhog` to geotechnical engineers seeking to get started with scripting. This course has been delivered to Systra in Dubai (April 2024) and RINA in Milan (October 2024).

# Acknowledgements

We acknowledge contributions from Berk Demir for providing the first examples of Streamlit applications with `Groundhog`.

# References

