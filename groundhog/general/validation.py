#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = 'Bruno Stuyts'

import re
import numpy as np
from functools import wraps, partial
import inspect
from copy import deepcopy
from collections import OrderedDict
import warnings


def validate_float(var_name,value,min_value=None,max_value=None):
    """
    Validates whether a variable can be used as a floating point number and whether it is within specified bounds
    If a value equals one of the bounds, the validation passes
    """
    try:
        float(value)
    except Exception as err:
        raise TypeError("%s (%s) is not a floating point number - %s" % (var_name,str(value),str(err)))
        
    if min_value!=None and value<min_value:
        raise ValueError("%s (%s) cannot be smaller than %s" % (var_name,str(value),str(min_value)))
        
    if max_value!=None and value>max_value:
        raise ValueError("%s (%s) cannot be greater than %s" % (var_name,str(value),str(max_value)))
    
    return True
    
def validate_integer(var_name,value,min_value=None,max_value=None):
    """
    Validates whether a variable can be used as an integer and whether it is within specified bounds
    If a value equals one of the bounds, the validation passes
    """
    try:
        if int(value)==value:
            pass
        else:
            raise TypeError("Value can be converted to integer (%s) but converted integer does not equal %s" % (str(int(value)),str(value)))
    except Exception as err:
        raise TypeError("%s (%s) is not an integer number - %s" % (var_name,str(value),str(err)))
        
    if min_value!=None and value<min_value:
        raise ValueError("%s (%s) cannot be smaller than %s" % (var_name,str(value),str(min_value)))
        
    if max_value!=None and value>max_value:
        raise ValueError("%s (%s) cannot be greater than %s" % (var_name,str(value),str(max_value)))
    
    return True

def validate_boolean(var_name,value):
    """
    Validates whether a variable can be used as a boolean
    """
    try:
        if bool(value)==value:
            pass
        else:
            raise TypeError("Value can be converted to boolean (%s) but converted boolean does not equal %s" % (str(bool(value)),str(value)))
    except Exception as err:
        raise TypeError("%s (%s) is not a boolean - %s" % (var_name,str(value),str(err)))
    
    return True
    
def validate_string(var_name,value,options=None,regex=None):
    """
    Validates whether a variable can be used as a string.
    The routine also allows checking whether the string is in a list of strings
    or whether it matches a specific regex pattern
    """
    try:
        if str(value)==value:
            pass
        else:
            raise TypeError("Value can be converted to string (%s) but converted string does not equal %s" % (str(value),str(value)))
    except Exception as err:
        raise TypeError("%s (%s) is not a string - %s" % (var_name,str(value),str(err)))
    
    if options!=None and value not in options:
        raise ValueError("%s (%s) not included in list of allowable strings (%s)" % (var_name,str(value),str(options)))
        
    if regex!=None and not bool(re.match(re.compile(regex), value)):
        raise ValueError("%s (%s) does not match the required string format (%s)" % (var_name,str(value),str(regex)))
    
    return True
   
def validate_list(var_name,value,elementtype=None,order=None,unique=None,empty_allowed=None):
    """
    Validates whether a list contains numbers. It allows checking whether these numbers are ascending or descending
    and whether non-unique values exist 
    """
    try:
        if type(value)==np.ndarray:
            value=list(value)

        if list(value)==value or tuple(value)==value:
            pass
        else:
            raise TypeError("Value can be converted to list (%s) but converted list does not equal %s" % (str(value),str(value)))
    except Exception as err:
        raise TypeError("%s (%s) is not a list or tuple - %s" % (var_name,str(value),str(err)))
    
    if elementtype!=None:
        try:
            for i,el in enumerate(value):
                if elementtype=="float":
                    validate_float(var_name,el)
                elif elementtype=="string":
                    validate_string(var_name,el)
                elif elementtype=="int":
                    validate_integer(var_name,el)
                elif elementtype=="boolean":
                    validate_boolean(var_name,el)
                else:
                    raise ValueError("Unspecified elementtype")
        except Exception as err:
            raise ValueError("Invalid element type for %s, %s required" % (str(el),elementtype))
    
    if order=='ascending':
        try:
            if sorted(value)==value and (np.nan not in value):
                pass
            else:
                raise ValueError("List %s is not ascending" % str(value))
        except Exception as err:
            raise ValueError("%s" % str(err))
    elif order=='descending':
        try:
            if sorted(value)==list(reversed(value)) and (np.nan not in value):
                pass
            else:
                raise ValueError("List %s is not descending" % str(value))
        except Exception as err:
            raise ValueError("%s" % str(err))
    elif order is None:
        pass # Nothing happens when order is not specified
    else:
        raise ValueError("Incorrect string for list order")
    
    if unique==True:
        if len(value) > len(set(value)):
            raise ValueError("%s (%s) contains non-unique elements" % (var_name,str(value)))
    elif unique is None or unique==False:
        pass # Nothing happens when unique is None or unspecified
    else:
        raise ValueError("Validation parameter unique must be boolean")
        
    if empty_allowed==False:
        if len(value)==0:
            raise ValueError("Empty lists are not allowed")
        
    return True

def map_args(method,var,*args,**kwargs):
    
    """
    Constructs a data structure with all parameters, their values and the validation parameters
    which need to be used during validation.
    
    :param method: The function for which validation will be applied
    :param var: The validation data structure, entered as argument of the function decorator
    :param args: function arguments
    :param kwargs: function keyword arguments
    
    :returns dictionary var_validation which is a copy of the validation data structure
             it is possible to override __min and __max arguments
    """
    try:
        # Construct a data structure with all function arguments, defaults are used
        # Remove self for validators applied to class methods
        parameter_names = [parameter.name for parameter in inspect.signature(method).parameters.values() \
                           if ((parameter.kind == parameter.POSITIONAL_OR_KEYWORD) and (parameter.name!='self'))]
        all_vars = OrderedDict.fromkeys(parameter_names)

        args = tuple(x for x in args if isinstance(x, (int, float, str, bool, complex, list, tuple, np.ndarray)))

        for parameter in inspect.signature(method).parameters.values():
            if str(parameter) != 'self':
                if not isinstance(parameter.default, type):
                    all_vars[parameter.name] = parameter.default

        for key, value in kwargs.items():
            if key in parameter_names:
                all_vars[key] = value

        for i, arg in enumerate(args):
            all_vars[list(all_vars.keys())[i]] = args[i]

        var_validation = deepcopy(var)
        
        for key in kwargs.keys():
            # Modification of min and max ranges with override
            # To be changed for not permanent override of min and max
            if key.endswith('__min'):
                var_validation[key.replace('__min','')]['min_value'] = kwargs[key]
            elif key.endswith('__max'):
                var_validation[key.replace('__max','')]['max_value'] = kwargs[key]
            # Bind the actual function arguments, this is required because the defaults are otherwise used 
            else:
                all_vars[key]=kwargs[key]
        
        # Add the value used at runtime to the validation data structure, except for raises_errors and validate
        # which are used elsewhere and not in the validation routine
        for key in all_vars.keys():
            try:
                var_validation[key]['value'] = all_vars[key]
            except:
                pass
        
        return var_validation
    except Exception as err:
        raise ValueError("Error during mapping of validation parameters to function parameters - %s" % str(err))


class Validator(object):
    """
    The Validator has the following features

        - Automatic handling of validation errors
        - Automatic handling of function output upon errors
        - Possibility to override the default validation dictionary with custom validation

    """

    def __init__(self, validationspec, outputonerrorspec):
        self.validationspec = validationspec
        self.outputonerror = outputonerrorspec

    def __call__(self, fn):
        @wraps(fn)
        def decorated(*args, **kwargs):

            try:
                validate = kwargs['validate']
            except:
                validate = None

            try:
                fail_silently = kwargs['fail_silently']
            except:
                fail_silently = True

            try:
                validation_params = kwargs['customvalidation']
            except:
                validation_params = self.validationspec

            try:
                output_for_errors = kwargs['customerroroutput']
            except:
                output_for_errors = self.outputonerror

            if validate or validate is None:
                # Execute validation
                try:
                    var_validation = map_args(fn, validation_params, *args, **kwargs)

                    for v in var_validation.keys():

                        if var_validation[v]['type'] == 'float':
                            validate_float(v, var_validation[v]['value'],
                                           var_validation[v]['min_value'],
                                           var_validation[v]['max_value'])
                        elif var_validation[v]['type'] == 'int':
                            validate_integer(v, var_validation[v]['value'],
                                             var_validation[v]['min_value'],
                                             var_validation[v]['max_value'])
                        elif var_validation[v]['type'] == 'string':
                            validate_string(v, var_validation[v]['value'],
                                            options=var_validation[v]['options'],
                                            regex=var_validation[v]['regex'])
                        elif var_validation[v]['type'] == 'bool':
                            validate_boolean(v, var_validation[v]['value'])
                        elif var_validation[v]['type'] == 'list':
                            validate_list(v, var_validation[v]['value'],
                                          var_validation[v]['elementtype'],
                                          var_validation[v]['order'],
                                          var_validation[v]['unique'],
                                          var_validation[v]['empty_allowed'])

                except Exception as err:
                    warnings.warn(str(err))
                    if fail_silently:
                        return output_for_errors
                    else:
                        raise
            else:
                # No validation
                pass

            try:
                result = fn(*args, **kwargs)
                return result
            except:
                if fail_silently:
                    return output_for_errors
                else:
                    raise

        return decorated

def check_layer_overlap(df, raise_error=True, z_from_key=None, z_to_key=None):
    """
    Checks possible overlap on a dataframe
    :param df: Dataframe with keys 'z from [m]' and 'z to [m]'. Other keys can be used but then the arguments `z_from_key` and `z_to_key` need to be provided.
    :param raise_error: Boolean determining whether an error needs to be raised or whether a warning is sufficient (default behaviour is to raise an error warning)
    :param z_from_key: Key for start depth of the layer
    :param z_to_key: Key for end depth of the layer
    :return: Default behaviour: raises a warning if there are overlaps or gaps.
    """
    # Reset the index first
    df.reset_index(drop=True, inplace=True)

    # Set keys for top and bottom depths of layers
    if z_from_key is None:
        z_from_key = "z from [m]"

    if z_to_key is None:
        z_to_key = "z to [m]"

    for i, row in df.iterrows():
        if i > 0:
            if row[z_from_key] > df.loc[i-1, z_to_key]:
                if raise_error:
                    raise ValueError("A gap exists between layer %i and %i" % (i-1, i))
                else:
                    warnings.warn("A gap exists between layer %i and %i" % (i-1, i))
            elif row[z_from_key] < df.loc[i-1, z_to_key]:
                if raise_error:
                    raise ValueError("Overlap exists between layer %i and %i" % (i-1, i))
                else:
                    warnings.warn("Overlap exists between layer %i and %i" % (i-1, i))