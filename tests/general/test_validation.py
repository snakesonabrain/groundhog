#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = 'Bruno Stuyts'

# Native Python packages
import unittest
import os

# 3rd party packages
import pandas as pd
import numpy as np

# Project imports
from groundhog.general.validation import *


VALIDATION_DATA = {
    'a': {'type':'float','min_value':0.0,'max_value':1.0},
    'b': {'type':'string','options':('bruno','stuyts'),'regex':None},
    'c': {'type':'float','min_value':None,'max_value':None},
    'd': {'type':'list','elementtype':'float','order':'ascending','unique':True,'empty_allowed':True}
}

class Test_validate_float(unittest.TestCase):

    def test_nonfloat(self):
        example_string = "abcd"
        self.assertRaises(TypeError,validate_float,"example_string",example_string)
        example_dict = {'1': 2.5, '2': 'abc'}
        self.assertRaises(TypeError,validate_float,"example_dict",example_dict)
        example_tuple = (1.0,2.0,3.0)
        self.assertRaises(TypeError,validate_float,"example_tuple",example_tuple)
        example_list = [1.0,2.0,3.0]
        self.assertRaises(TypeError,validate_float,"example_list",example_list)

    def test_nan(self):
        self.assertEqual(validate_float("example_float",np.nan),True)

    def test_float(self):
        example_float = 1.1
        self.assertEqual(validate_float("example_float",example_float),True)

    def test_min(self):
        value = 5.0
        min_value = 10.0
        self.assertRaises(ValueError,validate_float,"example_float",value,min_value=min_value)

    def test_max(self):
        value = 10.0
        max_value = 5.0
        self.assertRaises(ValueError,validate_float,"example_float",value,max_value=max_value)

    def test_range(self):
        value = 10.0
        max_value = 5.0
        min_value = 1.0
        self.assertRaises(ValueError,validate_float,"example_float",value,min_value=min_value,max_value=max_value)
        value = 0
        self.assertRaises(ValueError,validate_float,"example_float",value,min_value=min_value,max_value=max_value)


class Test_validate_integer(unittest.TestCase):

    def test_noninteger(self):
        example_string = "abcd"
        self.assertRaises(TypeError,validate_integer,"example_string",example_string)
        example_dict = {'1': 2.5, '2': 'abc'}
        self.assertRaises(TypeError,validate_integer,"example_dict",example_dict)
        example_tuple = (1.0,2.0,3.0)
        self.assertRaises(TypeError,validate_integer,"example_tuple",example_tuple)
        example_list = [1.0,2.0,3.0]
        self.assertRaises(TypeError,validate_integer,"example_list",example_list)
        example_nonint = 1.2
        self.assertRaises(TypeError,validate_integer,"example_nonint",example_nonint)

    def test_integer(self):
        example_int = 1
        self.assertEqual(validate_integer("example_int",example_int),True)

    def test_nan(self):
        self.assertEqual(validate_float("example_float",np.nan),True)

    def test_min(self):
        value = 5
        min_value = 10
        self.assertRaises(ValueError,validate_integer,"example_integer",value,min_value=min_value)

    def test_max(self):
        value = 10
        max_value = 5
        self.assertRaises(ValueError,validate_integer,"example_integer",value,max_value=max_value)

    def test_range(self):
        value = 10
        max_value = 5
        min_value = 1
        self.assertRaises(ValueError,validate_integer,"example_integer",value,min_value=min_value,max_value=max_value)
        value = 0
        self.assertRaises(ValueError,validate_integer,"example_integer",value,min_value=min_value,max_value=max_value)


class Test_validate_boolean(unittest.TestCase):

    def test_nonboolean(self):
        example_string = "abcd"
        self.assertRaises(TypeError,validate_boolean,"example_string",example_string)
        example_dict = {'1': 2.5, '2': 'abc'}
        self.assertRaises(TypeError,validate_boolean,"example_dict",example_dict)
        example_tuple = (1.0,2.0,3.0)
        self.assertRaises(TypeError,validate_boolean,"example_tuple",example_tuple)
        example_list = [1.0,2.0,3.0]
        self.assertRaises(TypeError,validate_boolean,"example_list",example_list)
        example_float = 1.2
        self.assertRaises(TypeError,validate_boolean,"example_nonint",example_float)

    def test_boolean(self):
        example_boolean = True
        self.assertEqual(validate_boolean("example_boolean",example_boolean),True)


class Test_validate_string(unittest.TestCase):

    def test_nonstring(self):
        example_int = 10
        self.assertRaises(TypeError,validate_string,"example_int",example_int)
        example_dict = {'1': 2.5, '2': 'abc'}
        self.assertRaises(TypeError,validate_string,"example_dict",example_dict)
        example_tuple = (1.0,2.0,3.0)
        self.assertRaises(TypeError,validate_string,"example_tuple",example_tuple)
        example_list = [1.0,2.0,3.0]
        self.assertRaises(TypeError,validate_string,"example_list",example_list)
        example_float = 1.2
        self.assertRaises(TypeError,validate_string,"example_nonint",example_float)

    def test_string(self):
        example_string = "Bruno Stuyts"
        self.assertEqual(validate_string("example_string",example_string),True)

    def test_options(self):
        example_list = ('first','second','third')
        example_string = "fourth"
        self.assertRaises(ValueError,validate_string,"example_string",example_string,options=example_list)
        example_list = ['first','second','third']
        example_string = "fourth"
        self.assertRaises(ValueError,validate_string,"example_string",example_string,options=example_list)
        example_string = 'third'
        self.assertEqual(validate_string("example_string",example_string,options=example_list),True)

    def test_regex(self):
        example_regex = "^[a-z]+"
        example_string = "bruno"
        self.assertEqual(validate_string("example_string",example_string,regex=example_regex),True)
        example_string = "Bruno"
        self.assertRaises(ValueError,validate_string,"example_string",example_string,regex=example_regex)
        example_string = "123"
        self.assertRaises(ValueError,validate_string,"example_string",example_string,regex=example_regex)


class Test_validate_list(unittest.TestCase):

    def test_nonlist(self):
        example_int = 10
        self.assertRaises(TypeError,validate_list,"example_int",example_int)
        example_dict = {'1': 2.5, '2': 'abc'}
        self.assertRaises(TypeError,validate_list,"example_dict",example_dict)
        example_string = "bruno"
        self.assertRaises(TypeError,validate_list,"example_string",example_string)
        example_float = 1.2
        self.assertRaises(TypeError,validate_list,"example_float",example_float)

    def test_list(self):
        example_list = [1.0,2.0,3.0]
        self.assertEqual(validate_list("example_list",example_list),True)
        example_tuple = (1.0,2.0,3.0)
        self.assertEqual(validate_list("example_tuple",example_tuple),True)

    def test_elementtype(self):
        example_list = [1.0,2.0,3.0]
        self.assertEqual(validate_list("example_list",example_list,elementtype="float"),True)
        example_list = [1.0,"a",3.0]
        self.assertRaises(ValueError,validate_list,"example_list",example_list,elementtype="float")
        example_list = [1.0,2.2,3.0]
        self.assertRaises(ValueError,validate_list,"example_list",example_list,elementtype="int")
        example_list = [1,2,3]
        self.assertEqual(validate_list("example_list",example_list,elementtype="int"),True)
        example_list = ['a','b','c']
        self.assertEqual(validate_list("example_list",example_list,elementtype="string"),True)
        example_list = ['a',2.2,'c']
        self.assertRaises(ValueError,validate_list,"example_list",example_list,elementtype="string")

    def test_ascending(self):
        example_list = [1.0,2.0,3.0]
        self.assertEqual(validate_list("example_list",example_list,order="ascending"),True)
        example_list = [1.0,3.0,2.0]
        self.assertRaises(ValueError,validate_list,"example_list",example_list,order="ascending")
        example_list = [np.nan,3.0,2.0]
        self.assertRaises(ValueError,validate_list,"example_list",example_list,order="ascending")
        example_list = [3.0,np.nan,2.0]
        self.assertRaises(ValueError,validate_list,"example_list",example_list,order="ascending")

    def test_descending(self):
        example_list = [3.0,2.0,1.0]
        self.assertEqual(validate_list("example_list",example_list,order="descending"),True)
        example_list = [2.0,3.0,1.0]
        self.assertRaises(ValueError,validate_list,"example_list",example_list,order="descending")
        example_list = [3.0,2.0,np.nan]
        self.assertRaises(ValueError,validate_list,"example_list",example_list,order="descending")
        example_list = [3.0,np.nan,2.0,1.0,]
        self.assertRaises(ValueError,validate_list,"example_list",example_list,order="descending")

    def test_unique(self):
        example_list = [3.0,2.0,1.0]
        self.assertEqual(validate_list("example_list",example_list,unique=True),True)
        example_list = [3.0,3.0,1.0]
        self.assertRaises(ValueError,validate_list,"example_list",example_list,unique=True)

    def test_empty(self):
        example_list = []
        self.assertEqual(validate_list("example_list",example_list),True)
        self.assertRaises(ValueError,validate_list,"example_list",example_list,empty_allowed=False)


class Test_map_args(unittest.TestCase):

    def setUp(self):
        self.validation_data = {
            'a': {'type':'float','min_value':0.0,'max_value':1.0},
            'b': {'type':'string','options':None,'regex':None},
            'c': {'type':'float','min_value':None,'max_value':None},
        }

        def test_func(a, b, c=1.0):
            pass
        self.test_func = test_func

    def test_mapping(self):
        mapped_data = map_args(self.test_func,self.validation_data,0.5,'bruno')
        self.assertEqual(mapped_data['a']['type'],'float')
        self.assertEqual(mapped_data['a']['value'],0.5)
        self.assertEqual(mapped_data['c']['value'],1.0)

    def test_override(self):
        mapped_data = map_args(self.test_func,self.validation_data,0.5,'bruno',a__min=None)
        self.assertEqual(mapped_data['a']['min_value'],None)
        mapped_data = map_args(self.test_func,self.validation_data,0.5,'bruno',a__min=-10.0,a__max=10.0)
        self.assertEqual(mapped_data['a']['min_value'],-10.0)
        self.assertEqual(mapped_data['a']['max_value'],10.0)


class Test_validate_new(unittest.TestCase):

    def setUp(self):
        self.ERROR_DICT = {'value': np.nan}

        self.CUSTOM_ERROR_DICT = {'value': 0.0}

        self.CUSTOM_VALIDATION = {
            'a': {'type':'float','min_value':0.0,'max_value':3.0},
            'b': {'type':'string','options':('bruno','stuyts','hendrik'),'regex':None},
            'c': {'type':'float','min_value':None,'max_value':None},
            'd': {'type':'list','elementtype':'float','order':'ascending','unique':True,'empty_allowed':True}
        }

        @Validator(VALIDATION_DATA, self.ERROR_DICT)
        def test_validated_func(a, b, c=1.0, d=[], **kwargs):
            return {'value': True}
        self.test_validated_func = test_validated_func

        @Validator(VALIDATION_DATA, self.ERROR_DICT)
        def test_fail_silentfunc(a, b, c=1.0, d=[], fail_silently=True, **kwargs):
            errorval = 1.0 / 0.0
            return {'value': errorval}
        self.test_fail_silentfunc = test_fail_silentfunc

    def test_validate_errors(self):
        self.assertRaises(ValueError,self.test_validated_func,2.0,'bruno', fail_silently=False)
        self.assertRaises(
            ValueError,self.test_validated_func,4.0,'bruno', customvalidation=self.CUSTOM_VALIDATION,
            fail_silently = False)
        self.assertRaises(Exception,self.test_validated_func,0.5,1.0, fail_silently=False)
        self.assertRaises(ValueError,self.test_validated_func,0.5,'hendrik', fail_silently=False)
        self.assertRaises(ValueError,self.test_validated_func,0.5,'bruno',c__min=2.0, fail_silently=False)
        self.assertRaises(ValueError,self.test_validated_func,0.5,'bruno',c__max=0.0, fail_silently=False)
        self.assertRaises(ValueError,self.test_validated_func,0.5,'bruno',d=[1.0,'b',3.0], fail_silently=False)
        self.assertRaises(ValueError,self.test_validated_func,0.5,'bruno',d=[1.0,5.0,3.0], fail_silently=False)
        self.assertRaises(ValueError,self.test_validated_func,0.5,'bruno',d=[1.0,1.0,3.0], fail_silently=False)

    def test_validate_correct(self):
        self.assertTrue(self.test_validated_func(0.5,'bruno')['value'])
        self.assertTrue(self.test_validated_func(1.5,'bruno', customvalidation=self.CUSTOM_VALIDATION)['value'])
        self.assertTrue(self.test_validated_func(0.5,'bruno',c__min=0.0,c__max=2.0)['value'])
        self.assertTrue(self.test_validated_func(0.5,'bruno',c__min=2.0,c__max=3.0,validate=False)['value'])
        self.assertTrue(self.test_validated_func(0.5,1.0,validate=False)['value'])

    def test_fail_silent(self):
        self.assertRaises(Exception,self.test_fail_silentfunc,0.0,'bruno',fail_silently=False)
        self.assertTrue(np.isnan(self.test_fail_silentfunc(0.0,'bruno')['value']))
        self.assertEqual(self.test_fail_silentfunc(0.0,'bruno', customerroroutput=self.CUSTOM_ERROR_DICT)['value'], 0.0)

    def test_additional_call(self):

        def external_func(**kwargs):
            # External function calling the function test_validated_func
            result = self.test_validated_func(**kwargs)
            return result['value']

        def external_func_failure(**kwargs):
            # External function calling the function test_fail_silentfunc
            result = self.test_fail_silentfunc(**kwargs)
            return result['value']

        @Validator(VALIDATION_DATA, self.ERROR_DICT)
        def double_decorated_func(a, b, c=1.0, d=[], **kwargs):
            # An external function also decorated with the validation decorator. Note that arguments and keyword
            # arguments used by the function need to be passes explicitly to the inside function since.
            # These are not in kwargs
            result = self.test_validated_func(a, b, c=c, d=d, **kwargs)
            return result['value']

        self.assertTrue(external_func(a=0.5, b='bruno'))
        self.assertRaises(ValueError, external_func, a=4.0, b='bruno', fail_silently=False)
        self.assertTrue(external_func(a=1.5, b='bruno', customvalidation=self.CUSTOM_VALIDATION))
        self.assertTrue(external_func(validate=False, a=25000.0, b='testje'))
        self.assertEqual(external_func_failure(a=0.0, b='bruno', customerroroutput=self.CUSTOM_ERROR_DICT),
                         0.0)
        self.assertTrue(double_decorated_func(a=0.5, b='bruno'))
        self.assertRaises(ValueError, double_decorated_func, a=4.0, b='bruno', fail_silently=False)
        self.assertTrue(double_decorated_func(a=1.5, b='bruno', customvalidation=self.CUSTOM_VALIDATION))


class Test_check_layer_overlap(unittest.TestCase):

    def setUp(self):
        self.df_no_overlap = pd.DataFrame({
            'z from [m]': np.array([0, 4, 10]),
            'z to [m]': np.array([4, 10, 20])
        })
        self.df_gaps = pd.DataFrame({
            'z from [m]': np.array([0, 5, 10]),
            'z to [m]': np.array([4, 10, 20])
        })
        self.df_overlap = pd.DataFrame({
            'z from [m]': np.array([0, 4, 10]),
            'z to [m]': np.array([5, 10, 20])
        })
        self.df_otherkeys = pd.DataFrame({
            'Depth from [m]': np.array([0, 4, 10]),
            'Depth to [m]': np.array([4, 10, 20])
        })

    def test_check_layer_overlap(self):
        check_layer_overlap(self.df_no_overlap)
        check_layer_overlap(self.df_otherkeys, z_from_key="Depth from [m]", z_to_key="Depth to [m]")
        # An error should be raised
        self.assertRaises(ValueError, check_layer_overlap, df=self.df_gaps)
        self.assertRaises(ValueError, check_layer_overlap, df=self.df_overlap)
        check_layer_overlap(self.df_gaps, raise_error=False)
        check_layer_overlap(self.df_overlap, raise_error=False)
