#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = 'Bruno Stuyts'

# Native Python packages
import os
import warnings

# 3rd party packages
import pandas as pd

# Project imports


def read_ags(file_path, groupname, combine_headers=True, includes_type=True):
    """
    Reads AGS data from a file and extracts the data for a given groupname into a DataFrame.
    All data with type "DP" is converted to float format

    :param file_path: Path (absolute or relative to the ags file)
    :param groupname: Name of the AGS group exactly as it is written in the AGS file
    :param combine_headers: Boolean determining whether the units are included in the header or not
    :param includes_type: Boolean determining whether a TYPE is included with the data
    :return: Dataframe with data for the given group
    """

    try:
        with open(file_path) as f:
            # Read the file as raw data
            ags_raw = pd.DataFrame(f.readlines(), columns=["AGS lines"])
    except Exception as err:
        raise IOError("Error during reading of file %s - %s" % (file_path, str(err)))

    # Find the index where the data for the given groupname starts
    try:
        if includes_type:
            start_index = ags_raw[ags_raw['AGS lines'].str.startswith(r'"GROUP","%s"' % groupname)].index[0]
        else:
            start_index = ags_raw[ags_raw['AGS lines'].str.startswith(r'"%s"' % groupname)].index[0]
    except:
        raise ValueError("Groupname not found in AGS file")

    try:
        # Find the index of the row where the data ends
        empty_rows = ags_raw[ags_raw['AGS lines'] == "\n"]
        if empty_rows.__len__() == 0:
            raise ValueError("AGS file not correctly structured, no, empty rows between groups")
        end_index = empty_rows[empty_rows.index > start_index].index[0]
    except:
        warnings.warn("No empty row detected below group, reading to the end of the file")
        end_index = ags_raw.index[-1]

    # Read the data as csv
    try:
        group_data = pd.read_csv(file_path, skiprows=start_index + 1, nrows=end_index - start_index-2)
    except Exception as err:
        raise ValueError("Error during reading of group data - %s" % str(err))

    # Convert column names and datatypes of numerical columns
    try:
        if includes_type:
            new_headers = []
            datatypes = dict()
            for i, original_header in enumerate(group_data.columns):
                if combine_headers:
                    new_name = "%s [%s]" % (original_header, group_data.loc[0, original_header])
                else:
                    new_name = original_header
                new_headers.append(new_name)
                if "DP" in group_data.loc[1, original_header]:
                    datatypes[new_name] = 'float'
            group_data.columns = new_headers
            group_data = group_data[2:].reset_index(drop=True).astype(datatypes)
        else:
            group_data = group_data[1:].reset_index(drop=True)
    except Exception as err:
        raise ValueError(
            "Error during the modification of column headers, check AGS file format - %s" % str(err))

    return group_data
