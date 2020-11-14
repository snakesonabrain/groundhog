#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = 'Bruno Stuyts'

# Native Python packages


# 3rd party packages
import numpy as np
import pandas as pd

# Project imports


def map_depth_properties(target_df, layering_df, target_z_key=None,
                         layering_zfrom_key=None, layering_zto_key=None):
    """
    Maps properties defined in a dataframe with layers to a dataframe with nodal depth positions.

    Note that numerical parameters in the layering dataframe should contain a unit between square brackets (e.g. 'Friction angle [deg]').
    String parameters don't have the square brackets (e.g. 'Soil type'). Do not use square brackets anywhere else in the column keys.

    Note that if a node of the target dataframe corresponds to a layer change, the properties of the layer layer below the selected node are assigned.

    :param target_df: Pandas dataframe to which the properties need to be mapped
    :param layering_df: Pandas dataframe with the layering definition
    :param target_z_key: Depth key in the target dataframe. If unspecified, 'z [m]' is used
    :param layering_zfrom_key: Start depth key in the layering dataframe. If unspecified, 'z from [m]' is used
    :param layering_zto_key: End depth key in the layering dataframe. If unspecified, 'z to [m]' is used
    :return:
    """
    # Check for depth keys
    if target_z_key is None:
        target_z_key = "z [m]"
    if layering_zfrom_key is None:
        layering_zfrom_key = "z from [m]"
    if layering_zto_key is None:
        layering_zto_key = "z to [m]"

    # Validation on presence of depth keys
    if target_z_key not in target_df.columns:
        raise ValueError("Required key for depth is not in the target dataframe (default is 'z [m]')")
    if (layering_zto_key not in layering_df.columns) or (layering_zfrom_key not in layering_df.columns):
        raise ValueError("Required key for start and/or end depth not in the layering dataframe"
                         " (default is 'z from [m]' and 'z to [m]'")
    # Validation on depth limits of layering dataframe and target dataframe
    if layering_df[layering_zfrom_key].min() > target_df[target_z_key].min():
        raise ValueError("Minimum depth for the layering (%.2fm) is greater "
                         "than minimum depth for the target dataframe (%.2fm)" % (
                layering_df[layering_zfrom_key].min(), target_df[target_z_key].min()
        ))
    if layering_df[layering_zto_key].max() < target_df[target_z_key].max():
        raise ValueError("Maximum depth for the layering (%.2fm) is smaller "
                         "than maximum depth for the target dataframe (%.2fm)" % (
                layering_df[layering_zto_key].max(), target_df[target_z_key].max()
        ))
    # Reset the index of the target dataframe to ensure correct calculation
    target_df.reset_index(drop=True, inplace=True)

    merged_z = np.insert(
        np.array(layering_df[layering_zto_key]),
        np.arange(len(layering_df[layering_zfrom_key])),
        np.array(layering_df[layering_zfrom_key]))

    for key in layering_df.columns:
        if key == layering_zfrom_key or key == layering_zto_key:
            pass
        else:
            if "[" in key and "]" in key:
                # Mapping for numerical parameter
                if "from" in key:
                    # Create merged property array for linearly increasing properties
                    if key.replace("from", "to") not in layering_df.columns:
                        raise ValueError("%s has no corresponding %s key." % (
                            key, key.replace("from", "to")
                        ))
                    target_key = key.replace("from ", "")
                    from_key = key
                    to_key = key.replace("from", "to")
                elif "to" in key:
                    pass
                else:
                    # Create merged property array for constant properties
                    target_key = key
                    from_key = key
                    to_key = key
                # Perform the actual mapping
                merged_prop = np.insert(
                    np.array(layering_df[to_key]),
                    np.arange(len(layering_df[from_key])),
                    np.array(layering_df[from_key]))

                target_df[target_key] = np.interp(target_df[target_z_key], merged_z, merged_prop)

            else:
                # Mapping for string parameter
                target_df[key] = list(map(lambda _z: layering_df[
                    (layering_df[layering_zfrom_key] <= _z) & (layering_df[layering_zto_key] >= _z)][key].iloc[-1],
                                          target_df[target_z_key]))

    return target_df


def merge_two_dicts(x, y):
    """
    Merges two dictionaries
    :param x: First dictionary
    :param y: Second dictionary
    :return: Updated dictionary
    """
    z = x.copy()   # start with x's keys and values
    z.update(y)    # modifies z with y's keys and values & returns None
    return z

def reverse_dict(input_dict):
    """
    Turn dictionary keys into values and values into keys
    :param input_dict: Dictionary with just 1 level
    :return: Dictionary with keys turned into values and vice-versa
    """
    return {y:x for x,y in input_dict.items()}


def offsets(startpoint, endpoint, point):
    """
    Calculates the offset between a point and a line joining a given start- and endpoint.
    The offset from the projected point to the start and end point is also calculated.
    Through analytical calculations, the position of the point is also determined.

    :param startpoint: Tuple with x and y coordinates of the starting point
    :param endpoint: Tuple with x and y coordinates of the end point
    :param point: Point for which the offset to the section needs to be computed

    :returns: Dictionary with the following keys:

        - 'offset start to point': Distance between start point and point of interest
        - 'offset end to point': Distance between end point and point of interest
        - 'offset to line': Offset between point and the line joining start and end point
        - 'offset to start projected': Offset from the start point (negative is before the start point)
        - 'offset to end projected': Offset from the end point (negative is behing the end point)
        - 'angle start [deg]': Angle between line joining start and end point and line joining point and start point
        - 'angle end [deg]': Angle between line joining start and end point and line joining point and end point
        - 'before start': Boolean determining if point lies before the start point
        - 'behind end': Boolean determining if point lies behind the end point

    """

    if endpoint[0] != startpoint[0]:
        a = (endpoint[1] - startpoint[1]) / (endpoint[0] - startpoint[0])
    else:
        a = 1e9
    b = -1
    c = startpoint[1] - a * startpoint[0]

    vector_1 = np.array([endpoint[0] - startpoint[0], endpoint[1] - startpoint[1]])
    vector_2 = np.array([point[0] - startpoint[0], point[1] - startpoint[1]])
    vector_3 = np.array([point[0] - endpoint[0], point[1] - endpoint[1]])

    unit_vector_1 = vector_1 / np.linalg.norm(vector_1)
    unit_vector_2 = vector_2 / np.linalg.norm(vector_2)
    unit_vector_3 = vector_3 / np.linalg.norm(vector_3)

    dot_product_start = np.dot(unit_vector_1, unit_vector_2)
    dot_product_end = np.dot(unit_vector_1, unit_vector_3)
    angle_start = np.degrees(np.arccos(dot_product_start))
    angle_end = np.degrees(np.arccos(dot_product_end))
    if angle_start > 90:
        before_start = True
    else:
        before_start = False

    if angle_end > 90:
        behind_end = False
    else:
        behind_end = True

    offset_start = np.linalg.norm(vector_2)
    offset_end = np.linalg.norm(vector_3)
    offset_to_line = np.abs(a * point[0] + b * point[1] + c) / np.sqrt(a ** 2 + b ** 2)
    offset_to_start = np.sqrt(offset_start ** 2 - offset_to_line ** 2)
    offset_to_end = np.sqrt(offset_end ** 2 - offset_to_line ** 2)

    if before_start:
        offset_to_start = -1 * offset_to_start
    if behind_end:
        offset_to_end = -1 * offset_to_end

    return {
        'offset start to point': offset_start,
        'offset end to point': offset_end,
        'offset to line': offset_to_line,
        'offset to start projected': offset_to_start,
        'offset to end projected': offset_to_end,
        'angle start [deg]': angle_start,
        'angle end [deg]': angle_end,
        'before start': before_start,
        'behind end': behind_end
    }