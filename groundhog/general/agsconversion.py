#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = 'Bruno Stuyts'

# Native Python packages
import warnings
import re
from io import StringIO

# 3rd party packages
import pandas as pd

GROUP_NAMES = {
    'PROJ': 'Project Information',
    'ABBR': 'Abbreviation Definitions',
    'DICT': 'User Defined Groups and Headings',
    'FILE': 'Associated Files',
    'TRAN': 'Data File Transmission Information / Data Status',
    'TYPE': 'Definition of Data Types',
    'UNIT': 'Definition of Units',
    'CLSS': 'Classification tests',
    'CONG': 'Consolidation Tests - General',
    'CONS': 'Consolidation Tests - Data',
    'CORE': 'Coring Information',
    'GEOL': 'Field Geological Descriptions',
    'GRAG': 'Particle Size Distribution Analysis - General',
    'GRAT': 'Particle Size Distribution Analysis - Data',
    'SCPG': "Static Cone Penetration Tests - General",
    'SCPT': 'Static Cone Penetration Tests - Data',
    'SCPP': 'Static Cone Penetration Tests - Derived Parameters',
    'LOCA': 'Location Details',
    'DETL': 'Stratum Detail Descriptions',
    'SAMP': 'Sample Information',
    'GCHM': 'Geotechnical Chemistry Testing',
    'LDEN': 'Density tests',
    'LLPL': 'Liquid and Plastic Limit Tests',
    'LNMC': 'Water/Moisture Content Tests',
    'LPDN': 'Particle Density Tests',
    'LPEN': 'Laboratory Hand Penetrometer Tests',
    'TREG': 'Triaxial Tests - Effective Stress - General',
    'TRET': 'Triaxial Tests - Effective Stress - Data',
    'TRIG': 'Triaxial Tests - Total Stress - General',
    'TRIT': 'Triaxial Tests - Total Stress - Data',
    'GRAD': 'Particle size distribution data',
    'RELD': 'Relative density test',
    'STCN': 'Static cone penetration test'
}

AGS_TABLES = {
    'PROJ': {
        'PROJ_ID': 'Project identifier',
        'PROJ_NAME': 'Project title',
        'PROJ_LOC': 'Location of site',
        'PROJ_CLNT': 'Client name',
        'PROJ_CONT': 'Contractors name',
        'PROJ_ENG': 'Project Engineer',
        'PROJ_MEMO': 'General project comments',
        'FILE_FSET': 'Associated file reference'
    },
    'ABBR': {
        'ABBR_HDNG': 'Field heading in group',
        'ABBR_CODE': 'Abbreviation used',
        'ABBR_DESC': 'Description of abbreviation',
        'ABBR_LIST': 'Source of abbreviation',
        'ABBR_REM': 'Remarks',
        'FILE_FSET': 'Associated file reference'
    },
    'DICT': {
        'DICT_TYPE': 'Flag to indicate definition is a GROUP or HEADING',
        'DICT_GRP': 'Group name',
        'DICT_HDNG': 'Heading name',
        'DICT_STAT': 'Heading status KEY, REQUIRED or OTHER',
        'DICT_DTYP': 'Type of data and format',
        'DICT_DESC': 'Description',
        'DICT_UNIT': 'Units',
        'DICT_EXMP': 'Example',
        'DICT_PGRP': 'Parent group name',
        'DICT_REM': 'Remarks',
        'FILE_FSET': 'Associated file reference'
    },
    'FILE': {
        'FILE_FSET': 'File set reference',
        'FILE_NAME': 'File name',
        'FILE_DESC': 'Description of content',
        'FILE_TYPE': 'File type',
        'FILE_PROG': 'Parent program and version number',
        'FILE_DOCT': 'Document type',
        'FILE_DATE': 'File date',
        'FILE_REM': 'Comments on file'
    },
    'TRAN': {
        'TRAN_ISNO': 'Issue sequence reference',
        'TRAN_DATE': 'Date of production of data file',
        'TRAN_PROD': 'Data file producer',
        'TRAN_STAT': 'Status of data within submission',
        'TRAN_DESC': 'Description of data transferred',
        'TRAN_AGS': 'AGS Edition Reference',
        'TRAN_RECV': 'Data file recipient',
        'TRAN_DLIM': 'Record Link data type Delimiter',
        'TRAN_RCON': 'Concatenator',
        'TRAN_REM': 'Remarks',
        'FILE_FSET': 'Associated file reference'
    },
    'TYPE': {
        'TYPE_TYPE': 'Data type code',
        'TYPE_DESC': 'Description',
        'FILE_FSET': 'Associated file reference'
    },
    'UNIT': {
        'UNIT_UNIT': 'Unit',
        'UNIT_DESC': 'Description',
        'UNIT_REM': 'Remarks',
        'FILE_FSET': 'Associated file reference'
    },
    'CONG': {
        'LOCA_ID': 'Location identifier',
        'SAMP_TOP': 'Depth to top of sample',
        'SAMP_REF': 'Sample reference',
        'SAMP_TYPE': 'Sample type',
        'SAMP_ID': 'Sample unique global identifier',
        'SPEC_REF': 'Specimen reference',
        'SPEC_DPTH': 'Depth to top of test specimen',
        'SPEC_DESC': 'Specimen description',
        'SPEC_PREP': 'Details of specimen preparation including time between preparation and testing',
        'CONG_TYPE': 'Type of consolidation test',
        'CONG_COND': 'Sample condition',
        'CONG_SDIA': 'Test specimen diameter',
        'CONG_HIGT': 'Test specimen height',
        'CONG_MCI': 'Initial water / moisture content',
        'CONG_MCF': 'Final water / moisture content',
        'CONG_BDEN': 'Initial bulk density',
        'CONG_DDEN': 'Initial dry density',
        'CONG_PDEN': 'Particle density with prefix # if value assumed',
        'CONG_SATR': 'Initial degree of saturation',
        'CONG_SPRS': 'Swelling pressure',
        'CONG_SATH': 'Height change of specimen on saturation, or flooding as percentage of original height',
        'CONG_IVR': 'Initial voids ratio',
        'CONG_REM': 'Remarks',
        'CONG_METH': 'Test method',
        'CONG_LAB' : 'Name of testing laboratory/organisation',
        'CONG_CRED': 'Accrediting body and reference number',
        'TEST_STAT': 'Test status',
        'FILE_FSET': 'Associated file reference'
    },
    'CONS': {
        'LOCA_ID': 'Location identifier',
        'SAMP_TOP': 'Depth to top of sample',
        'SAMP_REF': 'Sample reference',
        'SAMP_TYPE': 'Sample type',
        'SAMP_ID': 'Sample unique global identifier',
        'SPEC_REF': 'Specimen reference',
        'SPEC_DPTH': 'Depth to top of test specimen',
        'CONS_INCN': 'Oedometer stress increment',
        'CONS_IVR': 'Voids ratio at start of increment',
        'CONS_INCF': 'Stress at end of stress increment/decrement',
        'CONS_INCE': 'Voids ratio at end of stress increment',
        'CONS_INMV': 'Reported coefficient of volume compressibility over stress increment',
        'CONS_INSC': 'Coefficient of secondary compression over stress increment',
        'CONS_CVRT': 'Coefficient of consolidation over stress increment (root-time method)',
        'CONS_CVLG': 'Coefficient of consolidation over stress increment (log time method)',
        'CONS_TEMP': 'Average temperature over stress increment',
        'CONS_REM': 'Remarks',
        'FILE_FSET': 'Associated file reference'
    },
    'CORE': {
        'LOCA_ID': 'Location identifier',
        'CORE_TOP': 'Depth to top of core run',
        'CORE_BASE': 'Depth to base of core run',
        'CORE_PREC': 'Percentage of core recovered in core run(TCR)',
        'CORE_SREC': 'Percentage of solid core recovered in core run(SCR)',
        'CORE_RQD': 'Rock Quality Designation for core run(RQD)',
        'CORE_DIAM': 'Core diameter',
        'CORE_DURN': 'Time taken to drill core run',
        'CORE_REM': 'Remarks',
        'FILE_FSET': 'Associated file reference'
    },
    'GEOL': {
        'LOCA_ID': 'Location identifier',
        'GEOL_TOP': 'Depth to the top of stratum',
        'GEOL_BASE': 'Depth to the base of stratum',
        'GEOL_DESC': 'General description of stratum',
        'GEOL_LEG': 'Legend code',
        'GEOL_GEOL': 'Geology code',
        'GEOL_GEO2': 'Secondary geology code',
        'GEOL_STAT': 'Stratum reference shown on trial pit or traverse sketch',
        'GEOL_BGS': 'BGS Lexicon code',
        'GEOL_FORM': 'Geological formation or stratum name',
        'GEOL_REM': 'Remarks',
        'FILE_FSET': 'Associated file reference'
    },
    'GRAG': {
        'LOCA_ID': 'Location identifier',
        'SAMP_TOP': 'Depth to top of sample',
        'SAMP_REF': 'Sample reference',
        'SAMP_TYPE': 'Sample type',
        'SAMP_ID': 'Sample unique global identifier',
        'SPEC_REF': 'Specimen reference',
        'SPEC_DPTH': 'Depth to top of test specimen',
        'SPEC_DESC': 'Specimen description',
        'SPEC_PREP': 'Details of specimen preparation including time between preparation and testing',
        'GRAG_UC': 'Uniformity coefficient D60 / D10',
        'GRAG_VCRE': 'Percentage of material tested greater than 63 mm(cobbles)',
        'GRAG_GRAV': 'Percentage of material tested in range 63mm to 2mm (gravel)',
        'GRAG_SAND': 'Percentage of material tested in range 2mm to 63um (sand)',
        'GRAG_SILT': 'Percentage of material tested in range 63um to 2um (silt)',
        'GRAG_CLAY': 'Percentage of material tested less than 2um (clay)',
        'GRAG_FINE': 'Percentage less than 63um',
        'GRAG_REM': 'Remarks including commentary on effect of specimen disturbance on test result',
        'GRAG_METH': 'Test method',
        'GRAG_LAB': 'Name of testing laboratory / organisation',
        'GRAG_CRED': 'Accrediting body and reference number (when appropriate)',
        'TEST_STAT': 'Test status',
        'FILE_FSET': 'Associated file reference (eg equipment calibrations)'
    },
    'GRAT': {
        'LOCA_ID': 'Location identifier',
        'SAMP_TOP': 'Depth to top of sample',
        'SAMP_REF': 'Sample reference',
        'SAMP_TYPE': 'Sample type',
        'SAMP_ID': 'Sample unique global identifier',
        'SPEC_REF': 'Specimen reference',
        'SPEC_DPTH': 'Depth to top of test specimen',
        'GRAT_SIZE': 'Sieve or particle size',
        'GRAT_PERP': 'Percentage passing / finer than GRAT_SIZE',
        'GRAT_TYPE': 'Test type',
        'GRAT_REM': 'Remarks',
        'FILE_FSET': 'Associated file reference (eg test result sheets)'
    },
    'SCPG': {
        'LOCA_ID': 'Location identifier',
        'SCPG_TESN': 'Test reference or push number',
        'SCPG_TYPE': 'Cone test type',
        'SCPG_REF': 'Cone reference',
        'SCPG_CSA': 'Surface area of cone tip',
        'SCPG_RATE':  'Nominal rate of penetration of the cone',
        'SCPG_FILT': 'Type of filter material used',
        'SCPG_FRIC': 'Friction reducer used',
        'SCPG_WAT': 'Groundwater level at time of test',
        'SCPG_WATA':  'Origin of water level in SCPG_WAT',
        'SCPG_REM': 'Comments on testing and basis of any interpretated parameters included in SCPT and SCPP',
        'SCPG_ENV': 'Details of weather and environmental conditions during test',
        'SCPG_CONT': 'Subcontractors name',
        'SCPG_METH': 'Standard followed for testing',
        'SCPG_CRED': 'Accrediting body and reference number (when appropriate)',
        'SCPG_CAR': 'Cone area ratio used to calculate qt',
        'SCPG_SLAR': 'Sleeve area ratio used to calculate ft',
        'FILE_FSET': 'Associated file reference (eg cone calibration records)'
    },
    'SCPT': {
        'LOCA_ID': 'Location identifier',
        'SCPG_TESN': 'Test reference or push number',
        'SCPT_DPTH': 'Depth of result',
        'SCPT_RES': 'Cone resistance (qc)',
        'SCPT_FRES': 'Local unit side friction resistance (fs)',
        'SCPT_PWP1': 'Face porewater pressure (u1)',
        'SCPT_PWP2': 'Shoulder porewater pressure (u2)',
        'SCPT_PWP3': 'Top of sleeve porewater pressure',
        'SCPT_CON': 'Conductivity',
        'SCPT_TEMP': 'Temperature',
        'SCPT_PH': 'pH reading',
        'SCPT_SLP1': 'Slope indicator no. 1',
        'SCPT_SLP2': 'Slope indicator no. 2',
        'SCPT_REDX': 'Redox potential reading',
        'SCPT_MAGT': 'Magnetic flux - Total (calculated)',
        'SCPT_MAGX': 'Magnetic flux - X',
        'SCPT_MAGY': 'Magnetic flux - Y',
        'SCPT_MAGZ': 'Magnetic flux - Z',
        'SCPT_SMP': 'Soil moisture',
        'SCPT_NGAM': 'Natural gamma radiation',
        'SCPT_REM': 'Remarks',
        'SCPT_FRR': 'Friction ratio (Rf)',
        'SCPT_QT': 'Corrected cone resistance (qt) piezocone only',
        'SCPT_FT': 'Corrected sleeve resistance (ft) piezocone only',
        'SCPT_QE': 'Effective cone resistance (qe) piezocone only',
        'SCPT_BDEN': 'Bulk density of material (measured or assumed)',
        'SCPT_CPO': 'Total vertical stress (based on SCPT_BDEN)',
        'SCPT_CPOD': 'Effective vertical stress (calculated from SCPT_CPO and SCPT_ISPP or SCPG_WAT)',
        'SCPT_QNET': 'Net cone resistance (qn)',
        'SCPT_FRRC': 'Corrected friction ratio (Rf) piezocone only',
        'SCPT_EXPP': 'Excess pore pressure (u-uo) piezocone only',
        'SCPT_BQ': 'Pore pressure ratio (Bq) piezocone only',
        'SCPT_ISPP': 'In situ pore pressure (uo) (measured or assumed where not simple hydrostatic based on SCPG_WAT)',
        'SCPT_NQT': 'Normalised cone resistance (Qt)',
        'SCPT_NFR': 'Normalised friction ratio (Fr)',
        'FILE_FSET': 'Associated file reference (eg raw field data)'
    },
    'SCPP': {
        'LOCA_ID': 'Location identifier',
        'SCPG_TESN': 'Test reference or push number',
        'SCPP_TOP': 'Depth to top of layer',
        'SCPP_BASE': 'Depth to base of layer',
        'SCPP_REF': 'Interpretation reference',
        'SCPP_REM': 'Remarks',
        'SCPP_CSBT': 'Interpreted Soil Type',
        'SCPP_CSU': 'Undrained Shear Strength (Su); fine soils only',
        'SCPP_CRD': 'Relative density (Dr); coarse soils only',
        'SCPP_CPHI': 'Internal Friction Angle; coarse soils only',
        'SCPP_CIC': 'Soil Behaviour Type Index (Ic)',
        'SCPP_CSPT': 'Equivalent SPT N60 value',
        'FILE_FSET': 'Associated file reference'
    },
    'LOCA': {
        'LOCA_ID': 'Location identifier',
        'LOCA_TYPE': 'Type of activity',
        'LOCA_STAT': 'Status of information relating to this position',
        'LOCA_NATE': 'National Grid Easting of location or start of traverse',
        'LOCA_NATN': 'National Grid Northing of location or start of traverse',
        'LOCA_GREF': 'National grid referencing system used',
        'LOCA_GL': 'Ground level relative to datum of location or start of traverse',
        'LOCA_REM': 'General remarks',
        'LOCA_FDEP': 'Final depth',
        'LOCA_STAR': 'Date of start of activity',
        'LOCA_PURP': 'Purpose of activity at this location',
        'LOCA_TERM': 'Reason for activity termination',
        'LOCA_ENDD': 'End date of activity',
        'LOCA_LETT': 'OSGB letter grid reference',
        'LOCA_LOCX': 'Local grid x co-ordinate or start of traverse',
        'LOCA_LOCY': 'Local grid y co-ordinate or start of traverse',
        'LOCA_LOCZ': 'Level or start of traverse to local datum',
        'LOCA_LREF': 'Local grid referencing system used',
        'LOCA_DATM': 'Local datum referencing system used',
        'LOCA_ETRV': 'National Grid Easting of end of traverse',
        'LOCA_NTRV': 'National Grid Northing of end of traverse',
        'LOCA_LTRV': 'Ground level relative to datum of end of traverse',
        'LOCA_XTRL': 'Local grid easting of end of traverse',
        'LOCA_YTRL': 'Local grid northing of end of traverse',
        'LOCA_ZTRL': 'Local elevation of end of traverse',
        'LOCA_LAT': 'Latitude of location or start of traverse',
        'LOCA_LON': 'Longitude of location or start of traverse',
        'LOCA_ELAT': 'Latitude of end of traverse',
        'LOCA_ELON': 'Longitude of end of traverse',
        'LOCA_LLZ': 'Projection Format',
        'LOCA_LOCM': 'Method of location',
        'LOCA_LOCA': 'Site location sub division (within project) code or description',
        'LOCA_CLST': 'Investigation phase grouping code or description',
        'LOCA_ALID': 'Alignment Identifier',
        'LOCA_OFFS': 'Offset',
        'LOCA_CNGE': 'Chainage',
        'LOCA_TRAN': 'Reference to or details of algorithm used to calculate local grid reference, local ground levels or chainage',
        'FILE_FSET': 'Associated file reference (e.g. boring or pitting instructions, location photographs)',
        'LOCA_NATD': 'National Datum Referencing System used',
        'LOCA_ORID': 'Original Hole ID',
        'LOCA_ORJO': 'Original Job Reference',
        'LOCA_ORCO': 'Originating Company'
    },
    'DETL': {
        'LOCA_ID': 'Location identifier',
        'DETL_TOP': 'Depth to top of detail description',
        'DETL_BASE': 'Depth to base of detail description',
        'DETL_DESC': 'Detail description',
        'DETL_REM': 'Remarks',
        'FILE_FSET': 'Associated file reference'
    },
    'SAMP': {
        'LOCA_ID': 'Location identifier',
        'SAMP_TOP': 'Depth to top of sample',
        'SAMP_REF': 'Sample reference',
        'SAMP_TYPE': 'Sample type',
        'SAMP_ID': 'Sample unique identifier',
        'SAMP_BASE': 'Depth to base of sample',
        'SAMP_DTIM': 'Date and time sample taken',
        'SAMP_UBLO': 'Number of blows required to drive sampler',
        'SAMP_CONT': 'Sample container',
        'SAMP_PREP': 'Details of sample preparation at time of sampling',
        'SAMP_SDIA': 'Sample diameter',
        'SAMP_WDEP': 'Depth to water below ground surface at time of sampling',
        'SAMP_RECV': 'Percentage of sample recovered',
        'SAMP_TECH': 'Sampling technique/method',
        'SAMP_MATX': 'Sample matrix',
        'SAMP_TYPC': 'Sample QA type (Normal, blank or spike)',
        'SAMP_WHO': 'Samplers initials or name',
        'SAMP_WHY': 'Reason for sampling',
        'SAMP_REM': 'Sample remarks',
        'SAMP_DESC': 'Sample/specimen description',
        'SAMP_DESD': 'Date sample described',
        'SAMP_LOG': 'Person responsible for sample/specimen description',
        'SAMP_COND': 'Condition and representativeness of sample',
        'SAMP_CLSS': 'Sample classification as required by EN ISO 14688-1',
        'SAMP_BAR': 'Barometric pressure at time of sampling',
        'SAMP_TEMP': 'Sample temperature at time of sampling',
        'SAMP_PRES': 'Gas pressure (above barometric)',
        'SAMP_FLOW': 'Gas flow rate',
        'SAMP_ETIM': 'Date and time sampling completed',
        'SAMP_DURN': 'Sampling duration',
        'SAMP_CAPT': 'Caption used to describe sample',
        'SAMP_LINK': 'Sample record link',
        'GEOL_STAT': 'Stratum reference shown on trial pit or traverse sketch',
        'FILE_FSET': 'Associated file reference',
        'SAMP_RECL': 'Length of sample recovered'
    },
    'GCHM': {
        'LOCA_ID': 'Location identifier',
        'SAMP_TOP': 'Depth to top of sample',
        'SAMP_REF': 'Sample reference',
        'SAMP_TYPE': 'Sample type',
        'SAMP_ID': 'Sample unique global identifier',
        'SPEC_REF': 'Specimen reference',
        'SPEC_DPTH': 'Depth to top of test specimen',
        'GCHM_CODE': 'Determinand',
        'GCHM_METH': 'Test method',
        'GCHM_TTYP': 'Test type',
        'GCHM_RESL': 'Test result',
        'GCHM_UNIT': 'Test result units',
        'GCHM_NAME': 'Client/laboratory preferred name of determinand',
        'SPEC_DESC': 'Specimen description',
        'SPEC_PREP': 'Details of specimen preparation including time between preparation and testing',
        'GCHM_REM': 'Remarks on test',
        'GCHM_LAB': 'Name of testing laboratory/organisation',
        'GCHM_CRED': 'Accrediting body and reference number (when appropriate)',
        'TEST_STAT': 'Test status',
        'FILE_FSET': 'Associated file reference',
        'GCHM_RTXT': 'Reported test result',
        'GCHM_DLM': 'Limit of detection'
    },
    'LDEN': {
        'LOCA_ID': 'Location identifier',
        'SAMP_TOP': 'Depth to top of sample',
        'SAMP_REF': 'Sample reference',
        'SAMP_TYPE': 'Sample type',
        'SAMP_ID': 'Sample unique global identifier',
        'SPEC_REF': 'Specimen reference',
        'SPEC_DPTH': 'Depth to top of test specimen',
        'SPEC_DESC': 'Specimen description',
        'SPEC_PREP': 'Details of specimen preparation including time between preparation and testing',
        'LDEN_TYPE': 'Type of test performed',
        'LDEN_COND': 'Sample condition',
        'LDEN_SMTY': 'Type of sample',
        'LDEN_MC': 'Water/moisture content',
        'LDEN_BDEN': 'Bulk density',
        'LDEN_DDEN': 'Dry density',
        'LDEN_REM': 'Remarks',
        'LDEN_METH': 'Test method',
        'LDEN_LAB': 'Name of testing laboratory/organisation',
        'LDEN_CRED': 'Accrediting body and reference number (when appropriate)',
        'TEST_STAT': 'Test status',
        'FILE_FSET': 'Associated file reference (eg test result sheets)'
    },
    'LLPL': {
        'LOCA_ID': 'Location identifier',
        'SAMP_TOP': 'Depth to top of sample',
        'SAMP_REF': 'Sample reference',
        'SAMP_TYPE': 'Sample type',
        'SAMP_ID': 'Sample unique global identifier',
        'SPEC_REF': 'Specimen reference',
        'SPEC_DPTH': 'Depth to top of test specimen',
        'SPEC_DESC': 'Specimen description',
        'SPEC_PREP': 'Details of specimen preparation including time between preparation and testing',
        'LLPL_LL': 'Liquid limit',
        'LLPL_PL': 'Plastic limit',
        'LLPL_PI': 'Plasticity Index',
        'LLPL_425': 'Percentage passing 425μm sieve',
        'LLPL_PREP': 'Method of preparation',
        'LLPL_STAB': 'Amount of stabiliser added',
        'LLPL_STYP': 'Type of stabiliser added',
        'LLPL_REM': 'Remarks',
        'LLPL_METH': 'Test method',
        'LLPL_LAB': 'Name of testing laboratory/organisation',
        'LLPL_CRED': 'Accrediting body and reference number (when appropriate)',
        'TEST_STAT': 'Test status',
        'FILE_FSET': 'Associated file reference (eg test result sheets)'
    },
    'LNMC': {
        'LOCA_ID': 'Location identifier',
        'SAMP_TOP': 'Depth to top of sample',
        'SAMP_REF': 'Sample reference',
        'SAMP_TYPE': 'Sample type',
        'SAMP_ID': 'Sample unique global identifier',
        'SPEC_REF': 'Specimen reference',
        'SPEC_DPTH': 'Depth to top of test specimen',
        'SPEC_DESC': 'Specimen description',
        'SPEC_PREP': 'Details of specimen preparation including time between preparation and testing',
        'LNMC_MC': 'Water/moisture content',
        'LNMC_TEMP': 'Temperature sample dried at',
        'LNMC_STAB': 'Amount of stabiliser added',
        'LNMC_STYP': 'Type of stabiliser added',
        'LNMC_ISNT': 'Is test result assumed to be a natural water/moisture content',
        'LNMC_COMM': 'Reason water/moisture content is assumed to be other than natural',
        'LNMC_REM': 'Remarks',
        'LNMC_METH': 'Test method',
        'LNMC_LAB': 'Name of testing laboratory/organisation',
        'LNMC_CRED': 'Accrediting body and reference number (when appropriate)',
        'TEST_STAT': 'Test status',
        'FILE_FSET': 'Associated file reference (eg test result sheets)'
    },
    'LPDN': {
        'LOCA_ID': 'Location identifier',
        'SAMP_TOP': 'Depth to top of sample',
        'SAMP_REF': 'Sample reference',
        'SAMP_TYPE': 'Sample type',
        'SAMP_ID': 'Sample unique global identifier',
        'SPEC_REF': 'Specimen reference',
        'SPEC_DPTH': 'Depth to top of test specimen',
        'SPEC_DESC': 'Specimen description',
        'SPEC_PREP': 'Details of specimen preparation including time between preparation and testing',
        'LPDN_PDEN': 'Particle density with prefix # if value assumed',
        'LPDN_TYPE': 'Type of test',
        'LPDN_REM': 'Remarks',
        'LPDN_METH': 'Test method',
        'LPDN_LAB': 'Name of testing laboratory/organisation',
        'LPDN_CRED': 'Accrediting body and reference number (when appropriate)',
        'TEST_STAT': 'Test status',
        'FILE_FSET': 'Associated file reference (eg test result sheets)'
    },
    'LPEN': {
        'LOCA_ID': 'Location identifier',
        'SAMP_TOP': 'Depth to top of sample',
        'SAMP_REF': 'Sample reference',
        'SAMP_TYPE': 'Sample type',
        'SAMP_ID': 'Sample unique global identifier',
        'SPEC_REF': 'Specimen reference',
        'SPEC_DPTH': 'Depth to top of test specimen',
        'SPEC_DESC': 'Specimen description',
        'SPEC_PREP': 'Details of specimen preparation including time between preparation and testing',
        'LPEN_PPEN': 'Hand penetrometer undrained shear strength',
        'LPEN_MC': 'Water/moisture content local to test, if measured',
        'LPEN_REM': 'Remarks',
        'LPEN_METH': 'Test method',
        'LPEN_LAB': 'Name of testing laboratory/organisation',
        'LPEN_CRED': 'Accrediting body and reference number (when appropriate)',
        'TEST_STAT': 'Test status',
        'FILE_FSET': 'Associated file reference (eg test result sheet)'
    },
    'TREG': {
        'LOCA_ID': 'Location identifier',
        'SAMP_TOP': 'Depth to top of sample',
        'SAMP_REF': 'Sample reference',
        'SAMP_TYPE': 'Sample type',
        'SAMP_ID': 'Sample unique global identifier',
        'SPEC_REF': 'Specimen reference',
        'SPEC_DPTH': 'Depth to top of test specimen',
        'SPEC_DESC': 'Specimen description',
        'SPEC_PREP': 'Details of specimen preparation including time between preparation and testing',
        'TREG_TYPE': 'Test type',
        'TREG_COND': 'Sample condition',
        'TREG_COH': 'Cohesion intercept associated with TREG_PHI',
        'TREG_PHI': 'Angle of friction for effective shear strength triaxial test',
        'TREG_FCR': 'Failure criterion',
        'TREG_REM': 'Remarks including commentary on effect of specimen disturbance on test result',
        'TREG_METH': 'Test method',
        'TREG_LAB': 'Name of testing laboratory/organisation',
        'TREG_CRED': 'Accrediting body and reference number (when appropriate)',
        'TEST_STAT': 'Test status',
        'FILE_FSET': 'Associated file reference (eg equipment calibrations)'
    },
    'TRET': {
        'LOCA_ID': 'Location identifier',
        'SAMP_TOP': 'Depth to top of sample',
        'SAMP_REF': 'Sample reference',
        'SAMP_TYPE': 'Sample type',
        'SAMP_ID': 'Sample unique global identifier',
        'SPEC_REF': 'Specimen reference',
        'SPEC_DPTH': 'Depth to top of test specimen',
        'TRET_TESN': 'Triaxial test/stage number',
        'TRET_SDIA': 'Specimen diameter',
        'TRET_LEN': 'Specimen length',
        'TRET_IMC': 'Specimen initial water/moisture content',
        'TRET_FMC': 'Specimen final water/moisture content',
        'TRET_BDEN': 'Initial bulk density',
        'TRET_DDEN': 'Initial dry density',
        'TRET_SAT': 'Method of saturation',
        'TRET_CONS': 'Details of consolidation stage',
        'TRET_CONP': 'Effective stress at end of consolidation/ start of shear stage',
        'TRET_CELL': 'Total cell pressure during shearing stage',
        'TRET_PWPI': 'Porewater pressure at start of shear stage',
        'TRET_STRR': 'Rate of axial strain during shear',
        'TRET_STRN': 'Axial strain at failure',
        'TRET_DEVF': 'Deviator stress at failure',
        'TRET_PWPF': 'Porewater pressure at failure',
        'TRET_STV': 'Volumetric strain at failure (drained only)',
        'TRET_MODE': 'Mode of failure',
        'TRET_REM': 'Comments',
        'FILE_FSET': 'Associated file reference (eg test result sheets)'
    },
    'TRIG': {
        'LOCA_ID': 'Location identifier',
        'SAMP_TOP': 'Depth to top of sample',
        'SAMP_REF': 'Sample reference',
        'SAMP_TYPE': 'Sample type',
        'SAMP_ID': 'Sample unique global identifier',
        'SPEC_REF': 'Specimen reference',
        'SPEC_DPTH': 'Depth to top of test specimen',
        'SPEC_DESC': 'Specimen description',
        'SPEC_PREP': 'Details of specimen preparation including time between preparation and testing',
        'TRIG_TYPE': 'Test type',
        'TRIG_COND': 'Sample condition',
        'TRIG_REM': 'Remarks including commentary on effect of specimen disturbance on test result',
        'TRIG_METH': 'Test method',
        'TRIG_LAB': 'Name of testing laboratory/organisation',
        'TRIG_CRED': 'Accrediting body and reference number (when appropriate)',
        'TEST_STAT': 'Test status',
        'FILE_FSET': 'Associated file reference (eg equipment calibrations)'
    },
    'TRIT': {
        'LOCA_ID': 'Location identifier',
        'SAMP_TOP': 'Depth to top of sample',
        'SAMP_REF': 'Sample reference',
        'SAMP_TYPE': 'Sample type',
        'SAMP_ID': 'Sample unique global identifier',
        'SPEC_REF': 'Specimen reference',
        'SPEC_DPTH': 'Depth to top of test specimen',
        'TRIT_TESN': 'Triaxial test/stage reference',
        'TRIT_SDIA': 'Specimen diameter',
        'TRIT_SLEN': 'Specimen length',
        'TRIT_IMC': 'Specimen initial water/moisture content',
        'TRIT_FMC': 'Specimen final water/moisture content',
        'TRIT_CELL': 'Total cell pressure',
        'TRIT_DEVF': 'Corrected deviator stress at failure',
        'TRIT_BDEN': 'Initial bulk density',
        'TRIT_DDEN': 'Initial dry density',
        'TRIT_STRN': 'Axial strain at failure',
        'TRIT_CU': 'Undrained Shear Strength at failure',
        'TRIT_MODE': 'Mode of failure',
        'TRIT_REM': 'Comments',
        'FILE_FSET': 'Associated file reference (eg test result sheets)'
    },
    'CLSS': {
        'HOLE_ID': 'Location identifier',
        'SAMP_TOP': 'Depth to top of sample',
        'SAMP_REF': 'Sample reference',
        'SAMP_TYPE': 'Sample type',
        'SPEC_REF': 'Specimen reference number',
        'SPEC_DPTH': 'Specimen depth',
        'CLSS_NMC': 'Natural moisture content',
        'CLSS_LL': 'Liquid limit',
        'CLSS_PL': 'Plastic limit',
        'CLSS_BDEN': 'Bulk density',
        'CLSS_DDEN': 'Dry density',
        'CLSS_PD': 'Particle density',
        'CLSS_425': 'Percentage passing 425 micron sieve',
        'CLSS_PREP': 'Method of preparation',
        'CLSS_SLIM': 'Shrinkage limit',
        'CLSS_LS': 'Linear shrinkage',
        'CLSS_HVP': 'Hand vane undrained shear strength (peak)',
        'CLSS_HVR': 'Hand vane undrained shear strength (remoulded)',
        'CLSS_PPEN': 'Pocket penetrometer undrained shear strength',
        'CLSS_VNPK': 'Laboratory vane undrained shear strength (peak)',
        'CLSS_VNRM': 'Laboratory vane undrained shear strength (remoulded)',
        '?CLSS_REM': 'Notes on classification testing',
        '?FILE_FSET': 'Associated file reference'
    },
    'GRAD': {
        'HOLE_ID': 'Location identifier',
        'SAMP_TOP': 'Depth to top of sample',
        'SAMP_REF': 'Sample reference',
        'SAMP_TYPE': 'Sample type',
        'SPEC_REF': 'Specimen reference',
        'SPEC_DPTH': 'Specimen Depth',
        'GRAD_SIZE': 'Sieve or particle size',
        'GRAD_PERP': 'Percentage passing',
        'GRAD_TYPE': 'Grading analysis test type' 
    },
    'RELD': {
        'HOLE_ID': 'Location identifier',
        'SAMP_TOP': 'Depth to top of sample',
        'SAMP_REF': 'Sample reference',
        'SAMP_TYPE': 'Sample type',
        'SPEC_REF': 'Specimen reference',
        'SPEC_DPTH': 'Specimen depth',
        'RELD_REM': 'Method of test',
        'RELD_DMAX': 'Maximum dry density',
        'RELD_375': 'Percentage weight percent of sample retained on 37.5mm sieve',
        'RELD_Ø63': 'Percentage weight percent of sample retained on 6.3mm sieve',
        'RELD_Ø2Ø': 'Percentage weight percent of sample retained on 2mm sieve',
        'RELD_DMIN': 'Minimum dry density' 
    },
    'STCN': {
        'HOLE_ID': 'Location identifier',
        'STCN_DPTH': 'Depth',
        'STCN_TYP': 'Cone test type', 
        'STCN_REF': 'Cone ID',
        'STCN_FORC': 'Axial force',
        'STCN_FRIC': 'Sleeve force',
        'STCN_RES': 'Cone resistance',
        'STCN_FRES': 'Sleeve friction',
        'STCN_PWP1': 'Pore water pressure 1',
        'STCN_PWP2': 'Pore water pressure 2',
        'STCN_PWP3': 'Pore water pressure 3',
        'STCN_CON': 'Conductivity',
        'STCN_TEMP': 'Temperature',
        'STCN_PH': 'pH',
        'STCN_SLP1': 'Slope indicator 1',
        'STCN_SLP2': 'Slope indicator 2',
        'STCN_REDX': 'Redox potential',
        'STCN_FFD': 'Fluorescense',
        'STCN_PMT': 'Photo-multiplier tube reading',
        'STCN_PID':  'Photo ionization detector reading',
        'STCN_FID': 'Flame ionization detector reading',
        'FILE_FSET': 'Associated file reference'
    }
}

# Use common shorthands for geotechnical parameters
AGS_TABLES_SHORTHANDS = {
    'CONG': {
        'SPEC_DPTH': 'Depth',
        'CONG_SDIA': 'Diameter',
        'CONG_HIGT': 'Height',
        'CONG_MCI': 'w0',
        'CONG_MCF': 'wf',
        'CONG_BDEN': 'gamma_tot_0',
        'CONG_DDEN': 'gamma_d_0',
        'CONG_PDEN': 'gamma_grains',
        'CONG_SATR': 'S0',
        'CONG_SATH': 'Delta H',
        'CONG_IVR': 'e0',
    },
    'CONS': {
        'SPEC_DPTH': 'Depth',
        'CONS_INCN': 'Delta p',
        'CONS_IVR': 'e_i',
        'CONS_INCF': 'p_i+1',
        'CONS_INCE': 'e_i+1',
        'CONS_INMV': 'mv',
        'CONS_INSC': 'Cs',
        'CONS_CVRT': 'cv_rt',
        'CONS_CVLG': 'cv_lt'
    },
    'CORE': {
        'CORE_TOP': 'Depth from',
        'CORE_BASE': 'Depth to',
        'CORE_PREC': 'TCR',
        'CORE_SREC': 'SCR',
        'CORE_RQD': 'RQD',
        'CORE_DIAM': 'Diameter'
    },
    'GEOL': {
        'GEOL_TOP': 'Depth from',
        'GEOL_BASE': 'Depth to',
        'GEOL_DESC': 'Description'
    },
    'GRAG': {
        'SPEC_DPTH': 'Depth',
        'SPEC_DESC': 'Description',
        'GRAG_UC': 'Cu',
        'GRAG_VCRE': '% cobbles',
        'GRAG_GRAV': '% gravel',
        'GRAG_SAND': '% sand',
        'GRAG_SILT': '% silt',
        'GRAG_CLAY': '% clay',
        'GRAG_FINE': '% fines',
    },
    'GRAT': {
        'SPEC_DPTH': 'Depth',
        'GRAT_SIZE': 'd',
        'GRAT_PERP': '% passing'
    },
    'SCPG': {
        'SCPG_CSA': 'Ac',
        'SCPG_WAT': 'Groundwater level',
        'SCPG_CAR': 'Area ratio',
        'SCPG_SLAR': 'Sleeve area ratio'
    },
    'SCPT': {
        'SCPT_DPTH': 'Depth',
        'SCPT_RES': 'qc',
        'SCPT_FRES': 'fs',
        'SCPT_PWP1': 'u1',
        'SCPT_PWP2': 'u2',
        'SCPT_PWP3': 'u3',
        'SCPT_FRR': 'Rf',
        'SCPT_QT': 'qt',
        'SCPT_FT': 'ft',
        'SCPT_QE': 'qe',
        'SCPT_BDEN': 'gamma_total',
        'SCPT_CPO': 'sigma_vo',
        'SCPT_CPOD': 'sigma_vo_eff',
        'SCPT_QNET': 'qn',
        'SCPT_FRRC': 'Rf corrected',
        'SCPT_EXPP': 'u-u0',
        'SCPT_BQ': 'Bq',
        'SCPT_ISPP': 'u0',
        'SCPT_NQT': 'Qt',
        'SCPT_NFR': 'Fr'
    },
    'SCPP': {
        'SCPP_TOP': 'Depth from',
        'SCPP_BASE': 'Depth to',
        'SCPP_CSBT': 'Soil Type',
        'SCPP_CSU': 'Su',
        'SCPP_CRD': 'Dr',
        'SCPP_CPHI': 'Friction Angle',
        'SCPP_CIC': 'Ic',
        'SCPP_CSPT': 'N60'
    },
    'LOCA': {
        'LOCA_NATE': 'Easting',
        'LOCA_NATN': 'Northing',
        'LOCA_GREF': 'Coordinate system',
        'LOCA_GL': 'Ground level',
        'LOCA_LETT': 'OSGB reference',
        'LOCA_LOCX': 'Local grid x',
        'LOCA_LOCY': 'Local grid y',
        'LOCA_LOCZ': 'Local level',
        'LOCA_LREF': 'Local grid',
        'LOCA_DATM': 'Local datum',
        'LOCA_LAT': 'Latitude',
        'LOCA_LON': 'Longitude',
    },
    'DETL': {
        'DETL_TOP': 'Depth from',
        'DETL_BASE': 'Depth to',
        'DETL_DESC': 'Description'
    },
    'SAMP': {
        'SAMP_TOP': 'Depth from',
        'SAMP_ID': 'Sample ID',
        'SAMP_BASE': 'Depth to',
        'SAMP_UBLO': 'No blows',
        'SAMP_SDIA': 'Sample diameter',
        'SAMP_RECV': 'Recovery',
        'SAMP_TECH': 'Sample technique',
        'SAMP_DESC': 'Description',
        'SAMP_DESD': 'Date described',
        'SAMP_LOG': 'Logged by',
        'SAMP_CLSS': 'Classification',
        'SAMP_CAPT': 'Caption',
        'SAMP_RECL': 'Recovered length'
    },
    'GCHM': {
        'SAMP_ID': 'Sample ID',
        'SPEC_DPTH': 'Depth',
    },
    'LDEN': {
        'SAMP_ID': 'Sample ID',
        'SPEC_REF': 'Specimen ID',
        'SPEC_DPTH': 'Depth',
        'SPEC_DESC': 'Description',
        'SPEC_PREP': 'Preparation',
        'LDEN_TYPE': 'Test type',
        'LDEN_SMTY': 'Specimen type',
        'LDEN_MC': 'Water content',
    },
    'LLPL': {
        'SAMP_ID': 'Sample ID',
        'SPEC_DPTH': 'Depth',
        'SPEC_DESC': 'Description',
        'LLPL_LL': 'LL',
        'LLPL_PL': 'PL',
        'LLPL_PI': 'PI'
    },
    'LNMC': {
        'SAMP_ID': 'Sample ID',
        'SPEC_DPTH': 'Depth',
        'SPEC_DESC': 'Description',
        'LNMC_MC': 'Water content',
        'LNMC_TEMP': 'Temperature oven'
    },
    'LPDN': {
        'SAMP_ID': 'Sample ID',
        'SPEC_DPTH': 'Depth',
        'SPEC_DESC': 'Description',
        'LPDN_PDEN': 'Particle density',
        'LPDN_TYPE': 'Test type'
    },
    'LPEN': {
        'SAMP_ID': 'Sample ID',
        'SPEC_DPTH': 'Depth',
        'SPEC_DESC': 'Description',
        'LPEN_PPEN': 'Su',
        'LPEN_MC': 'Water content'
    },
    'TREG': {
        'SAMP_ID': 'Sample ID',
        'SPEC_DPTH': 'Depth',
        'SPEC_DESC': 'Description',
        'TREG_COH': 'Cohesion',
        'TREG_PHI': 'Friction angle',
        'TREG_REM': 'Remarks'
    },
    'TRET': {
        'SAMP_ID': 'Sample ID',
        'SPEC_DPTH': 'Depth',
        'TRET_TESN': 'Stage number',
        'TRET_SDIA': 'Specimen diameter',
        'TRET_LEN': 'Specimen length',
        'TRET_IMC': 'w0',
        'TRET_FMC': 'wf',
        'TRET_BDEN': 'gamma_total_0',
        'TRET_DDEN': 'gamma_d_0',
        'TRET_CONP': 'sigma_eff_0',
        'TRET_CELL': 'Cell pressure',
        'TRET_PWPI': 'u0',
        'TRET_STRR': 'Strain rate',
        'TRET_STRN': 'epsilon_a_f',
        'TRET_DEVF': 'q_f',
        'TRET_PWPF': 'u_f',
        'TRET_STV': 'epsilon_v_f',
        'TRET_MODE': 'Failure mode'
    },
    'TRIG': {
        'SAMP_ID': 'Sample ID',
        'SPEC_DPTH': 'Depth',
        'SPEC_DESC': 'Description',
        'TRIG_TYPE': 'Test type',
        'TRIG_REM': 'Remarks',
        'TRIG_METH': 'Test method'
    },
    'TRIT': {
        'SAMP_ID': 'Sample ID',
        'SPEC_DPTH': 'Depth',
        'TRIT_TESN': 'Test stage',
        'TRIT_SDIA': 'Specimen diameter',
        'TRIT_SLEN': 'Specimen length',
        'TRIT_IMC': 'w0',
        'TRIT_FMC': 'wf',
        'TRIT_CELL': 'Total cell pressure',
        'TRIT_DEVF': 'q_f',
        'TRIT_BDEN': 'gamma_total_0',
        'TRIT_DDEN': 'gamma_d_0',
        'TRIT_STRN': 'epsilon_a_f',
        'TRIT_CU': 'Su',
        'TRIT_MODE': 'Failure mode'
    },
    'CLSS': {
        'HOLE_ID': 'Location ID',
        'SAMP_TOP': 'Sample top',
        'SAMP_REF': 'Sample reference',
        'SAMP_TYPE': 'Sample type',
        'SPEC_REF': 'Specimen reference',
        'SPEC_DPTH': 'Depth',
        'CLSS_NMC': 'w',
        'CLSS_LL': 'LL',
        'CLSS_PL': 'PL',
        'CLSS_BDEN': 'Bulk density',
        'CLSS_DDEN': 'Dry density',
        'CLSS_PD': 'Particle density',
        'CLSS_425': 'Percentage passing 425 micron sieve',
        'CLSS_PREP': 'Preparation',
        'CLSS_SLIM': 'Shrinkage limit',
        'CLSS_LS': 'Linear shrinkage',
        'CLSS_HVP': 'Hand vane Su (peak)',
        'CLSS_HVR': 'Hand vane Su (remoulded)',
        'CLSS_PPEN': 'PP Su',
        'CLSS_VNPK': 'LV Su (peak)',
        'CLSS_VNRM': 'LV Su (remoulded)',
        'CLSS_REM': 'Notes',
        'FILE_FSET': 'File'
    },
    'GRAD': {
        'HOLE_ID': 'Location ID',
        'SAMP_TOP': 'Depth to top of sample',
        'SAMP_REF': 'Sample reference',
        'SAMP_TYPE': 'Sample type',
        'SPEC_REF': 'Specimen reference',
        'SPEC_DPTH': 'Depth',
        'GRAD_SIZE': 'Size',
        'GRAD_PERP': 'Percentage passing',
        'GRAD_TYPE': 'Test type' 
    },
    'STCN': {
        'HOLE_ID': 'Location identifier',
        'STCN_DPTH': 'z',
        'STCN_RES': 'qc',
        'STCN_FRES': 'fs',
        'STCN_PWP1': 'u1',
        'STCN_PWP2': 'u2',
        'STCN_PWP3': 'u3'
    }
}


class AGSConverter(object):

    def __init__(self, path, encoding='utf8', errors='replace', removedoublequotes=True,
        removeheadinglinebreaks=True, agsformat="4",**kwargs):
        """
        Initializes an AGS conversion object using the path to the AGS file.
        The AGS file needs to properly formatted with at least one blank line between each group.
        Each group should have four lines before the data starts:
            - A line with the group name;
            - A line with column headers;
            - A line with the units of the values in the columns;
            - A line with the data type of the columns

        The functionality is developed for AGS4.x files but support for AGS3.1 files is also available
        using ``agsformat="3.1"`` as optional keyword argument.

        :param path: Path to the AGS 4.0 file
        :param encoding: Encoding of the file (default=utf-8)
        :param errors: Specify file reading behaviour in case of encoding errors
        :param removedoublequotes: Boolean determining whether doublequotes need to be removed after file loading (default=True)
        :param removeheadinglinebreaks: Boolean determining whether line breaks in heading rows need to be removed after file loading (default=True)
        :param agsformat: Format of the AGS file (default=``"4"``). AGS 3.1 (``"3.1"``) is also available
        """
        self.path = path
        self.agsformat = agsformat
        with open(path, "r", encoding=encoding, errors=errors) as file_handle:
            self.rawtextstring = file_handle.read()
        if removeheadinglinebreaks:
            self.remove_heading_linebreaks()
        if removedoublequotes:
            self.remove_doublequotes(**kwargs)
        
        self.extract_groupnames()

    def remove_doublequotes(self, replace_by=""):
        """
        Remove double quotes which are not preceded by a comma ("") and replace by a the value defined in ``replace_by`` and a
        single quote.
        This is done because the ``read_csv`` function of Pandas will be used in a later
        stage and there can be errors when reading double quotes.
        Such expressions are common when coordinates in ° ' " format are included in the ags file.

        :param replace_by: String to replace the first quote of the double quote with
        :return:
        """
        self.textstring = re.sub(r'[^,]""', r'%s"' % replace_by, self.rawtextstring)

        self.raw_dataframe = pd.DataFrame(re.split(r'\n', self.textstring), columns=['AGS lines'])

    def remove_heading_linebreaks(self):
        """
        Removes line breaks in header rows which would prevent further AGS parsing
        If a comma is followed by a line break (``,\n``), it is replaced by a comma without line break
        """
        self.rawtextstring = self.rawtextstring.replace(',\n', ",")

    def extract_groupnames(self):
        """
        Scans the AGS file and extracts all group names
        :return: Sets the attribute ``groupnames`` of the ``AGSConverter`` object
        """
        if self.agsformat == "4":
            self.groupnames = re.findall(r'"GROUP","(?P<groupname>.+)"', self.textstring)
        elif self.agsformat == "3.1":
            self.groupnames = re.findall('\"\*\*(?P<groupname>.+)\"', self.textstring)
        else:
            raise ValueError("AGS format %s not recognised. Use '4' or '3.1' for currently supported formats")

    @staticmethod
    def convert_ags_headers(df, agsformat):
        """
        Converts the headers of an AGS-based dataframes from the three rows in the AGS to a single column header.
        Numerical data is also converted into the correct datatype.
        :param df: Dataframe with the group data
        :return: Dataframe with updated headers
        """
        # Rename the headers
        new_headers = []
        datatypes = dict()
        if agsformat == "3.1":
            if df[df.columns[0]].iloc[0] == "<UNITS>":
                pass
            else:
                return df
        for i, original_header in enumerate(df.columns):
            if str(df.loc[0, original_header]) == 'nan':
                new_name = "%s" % (original_header)
            else:
                new_name = "%s [%s]" % (original_header, df.loc[0, original_header])
            new_headers.append(new_name)
            if agsformat == '4':
                if ("DP" in df.loc[1, original_header]) or ("SF" in df.loc[1, original_header]):
                    datatypes[new_name] = 'float'
        df.columns = new_headers
        if agsformat == '4':
            df = df[2:].reset_index(drop=True).astype(datatypes)
        elif agsformat == "3.1":
            df = df[1:].reset_index(drop=True)
            for _col in df.columns:
                try:
                    df[_col] = pd.to_numeric(df[_col])
                except:
                    pass
        else:
            raise ValueError("AGS format %s not recognised. Use '4' or '3.1' for currently supported formats")
        return df

    def convert_ags_group(self, groupname, verbose_keys=False, additional_keys=dict(), use_shorthands=False,
                          drop_heading_col=True, **kwargs):
        """
        Isolate the data for a certain group and convert it to a Pandas dataframe.

        :param groupname: Name of the group to be converted
        :param verbose_keys: Boolean determining whether AGS code keys or their verbose equivalents are used. Conversion happens using the dictionaries in tables.py (default=False for AGS code keys)
        :param additional_keys: Additional custom keys used in dataframe column name conversion
        :param use_shorthands: Boolean determining whether shorthand codes should be used. If True, a first pass is done using these.

        :return: Returns a dataframe with the requested data
        """

        # Read the textstring into a pandas dataframe
        # Define where the data for the selected group starts
        if self.agsformat == "4":
            _start_index = self.raw_dataframe[
                self.raw_dataframe['AGS lines'].str.startswith(r'"GROUP","%s"' % groupname)].index[0]
        elif self.agsformat == "3.1":
            _start_index = self.raw_dataframe[
                self.raw_dataframe['AGS lines'].str.startswith(r'"**%s' % groupname)].index[0]
        else:
            raise ValueError("AGS format %s not recognised. Use '4' or '3.1' for currently supported formats")

        # Define where the data for the selected group ends
        empty_rows = self.raw_dataframe[
            (self.raw_dataframe['AGS lines'] == "\r\n") |
            (self.raw_dataframe['AGS lines'] == "") |
            (self.raw_dataframe['AGS lines'] == "\n")]

        # Forsee the case where the file does not end with a blank line
        try:
            _end_index = empty_rows[empty_rows.index > _start_index].index[0]
        except:
            _end_index = self.raw_dataframe.__len__()

        # Read only the data for the group
        _group_data = pd.read_csv(
            StringIO(self.textstring),
            skiprows=_start_index + 1,
            nrows=_end_index - _start_index - 2,
            **kwargs)

        # Remove * from header names in AGS3.1
        if self.agsformat == "3.1":
            coldict = dict()
            for _col in _group_data.columns:
                coldict[_col] = _col[1:]
            _group_data.rename(columns=coldict, inplace=True)

        # Converted to verbose column keys if necessary
        if verbose_keys:
            try:
                if use_shorthands:
                    _group_data.rename(
                        columns=AGS_TABLES_SHORTHANDS[groupname], inplace=True)
                conversion_scheme = {**AGS_TABLES[groupname], **additional_keys}
                _group_data.rename(
                    columns=conversion_scheme, inplace=True)
            except Exception as err:
                warnings.warn("Verbose names for group %s not found. AGS column names kept - %s" % (groupname, str(err)))

        # Convert the headers using the convert_ags_headers function (only for AGS 4.x format)
        _group_data = self.convert_ags_headers(_group_data, agsformat=self.agsformat)

        # Drop the HEADING [UNIT] column if required
        if drop_heading_col:
            try:
                _group_data.drop(['HEADING [UNIT]'], axis=1, inplace=True)
            except Exception as err:
                warnings.warn(
                    "Error during dropping of HEADING [UNIT] column - %s" % (str(err)))

        return _group_data

    def create_dataframes(self, selectedgroups=None, verbose_keys=False, use_shorthands=False, drop_heading_col=True,
                          **kwargs):
        """
        Create a dictionary with Pandas dataframes for each groupname.
        The groups can be finetuned through the ``selectedgroups`` argument.

        :param selectedgroups: List of groupnames to limit the conversion to (default is None leading to all groups being converted)
        :param verbose_keys: Boolean determining whether AGS code keys or their verbose equivalents are used. Conversion happens using the dictionaries in tables.py (default=False for AGS code keys)
        :param use_shorthands: Boolean determining whether shorthand codes should be used. If True, a first pass is done using these.
        :param drop_heading_col: Boolean determining is the column ``HEADING [UNIT]`` should be dropped (default=True)
        :return: Sets the ``data`` attribute for the ``AGSConverter`` object
        """
        self.data = dict()

        if selectedgroups is None:
            selectedgroups = self.groupnames

        # Create a dictionary with additional keys for verbosity
        additional_keys = dict()
        if verbose_keys:
            try:
                dict_data = self.convert_ags_group(groupname='DICT')
                additional_keys = dict()
                for i, row in dict_data.iterrows():
                    additional_keys[row['DICT_HDNG']] = row['DICT_DESC']
            except:
                pass

        for _groupname in selectedgroups:
            try:
                self.data[_groupname] = self.convert_ags_group(
                    groupname=_groupname, verbose_keys=verbose_keys,
                    use_shorthands=use_shorthands,
                    additional_keys=additional_keys,
                    drop_heading_col=drop_heading_col,
                    **kwargs)
            except Exception as err:
                warnings.warn("Group %s could not be converted - %s" % (_groupname, str(err)))


