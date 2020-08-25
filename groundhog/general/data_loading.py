#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = 'Bruno Stuyts'

# Django and native Python packages

# 3rd party packages
import pandas as pd
# Project imports


def load_ags(path, groupname, **kwargs):
    """
    Generates a Pandas dataframe from an .ags file for a given group name.
    The function is based on the AGS 4.0 standard for geotechnical data transfer

    :param path: Path to the file
    :param groupname: AGS 4.0 group name
    :return: A pandas dataframe with the AGS data
    """

    # Read the raw AGS data
    with open(path, 'rb') as f:
        _lines = list(map(lambda _x: _x.decode('utf-8', 'backslashreplace'), f.readlines()))
        ags_raw = pd.DataFrame(_lines, columns=["AGS lines"])

    # Define where the data for the selected group starts
    _start_index = ags_raw[ags_raw['AGS lines'].str.startswith(r'"GROUP","%s"' % groupname)].index[0]

    # Define where the data for the selected group ends
    empty_rows = ags_raw[(ags_raw['AGS lines'] == "\r\n") | (ags_raw['AGS lines'] == "\n")]
    _end_index = empty_rows[empty_rows.index > _start_index].index[0]

    # Read only the data for the group
    _group_data = pd.read_csv(path, skiprows=_start_index + 1, nrows=_end_index - _start_index - 2, **kwargs)

    # Rename the headers
    new_headers = []
    datatypes = dict()
    for i, original_header in enumerate(_group_data.columns):
        if str(_group_data.loc[0, original_header]) == 'nan':
            new_name = "%s" % (original_header)
        else:
            new_name = "%s [%s]" % (original_header, _group_data.loc[0, original_header])
        new_headers.append(new_name)
        if "DP" in _group_data.loc[1, original_header]:
            datatypes[new_name] = 'float'
    _group_data.columns = new_headers

    _group_data = _group_data[2:].reset_index(drop=True).astype(datatypes)

    return _group_data