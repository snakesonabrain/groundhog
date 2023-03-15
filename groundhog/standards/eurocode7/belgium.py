#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = 'Bruno Stuyts'

# Native Python packages

# 3rd party libraries
import numpy as np


CHARACTERISTIC_VALUES_NATIONALANNEX = {
    'Matig gepakt grind': {
        'Bulk unit weight above water table [kN/m3]': 18,
        'Bulk unit weight below water table [kN/m3]': 20,
        'Friction angle [deg]': 35,
  'Effective cohesion [kPa]': 0,
  'Undrained shear strength [kPa]': np.nan},
 'Dicht gepakt grind': {'Bulk unit weight above water table [kN/m3]': 19,
  'Bulk unit weight below water table [kN/m3]': 21,
  'Friction angle [deg]': 40,
  'Effective cohesion [kPa]': 0,
  'Undrained shear strength [kPa]': np.nan},
 'Matig gepakt leem- of kleihoudend grind': {'Bulk unit weight above water table [kN/m3]': 19,
  'Bulk unit weight below water table [kN/m3]': 21,
  'Friction angle [deg]': 32,
  'Effective cohesion [kPa]': 0,
  'Undrained shear strength [kPa]': np.nan},
 'Dicht gepakt leem- of kleihoudend grind': {'Bulk unit weight above water table [kN/m3]': 20,
  'Bulk unit weight below water table [kN/m3]': 22,
  'Friction angle [deg]': 37,
  'Effective cohesion [kPa]': 0,
  'Undrained shear strength [kPa]': np.nan},
 'Los gepakt zand': {'Bulk unit weight above water table [kN/m3]': 16,
  'Bulk unit weight below water table [kN/m3]': 18,
  'Friction angle [deg]': 27,
  'Effective cohesion [kPa]': 0,
  'Undrained shear strength [kPa]': np.nan},
 'Matig gepakt zand': {'Bulk unit weight above water table [kN/m3]': 17,
  'Bulk unit weight below water table [kN/m3]': 19,
  'Friction angle [deg]': 30,
  'Effective cohesion [kPa]': 0,
  'Undrained shear strength [kPa]': np.nan},
 'Dicht gepakt zand': {'Bulk unit weight above water table [kN/m3]': 18,
  'Bulk unit weight below water table [kN/m3]': 20,
  'Friction angle [deg]': 32,
  'Effective cohesion [kPa]': 0,
  'Undrained shear strength [kPa]': np.nan},
 'Zeer dicht gepakt zand': {'Bulk unit weight above water table [kN/m3]': 19,
  'Bulk unit weight below water table [kN/m3]': 20,
  'Friction angle [deg]': 35,
  'Effective cohesion [kPa]': 0,
  'Undrained shear strength [kPa]': np.nan},
 'Los gepakt leem- of kleihoudend zand': {'Bulk unit weight above water table [kN/m3]': 16,
  'Bulk unit weight below water table [kN/m3]': 18,
  'Friction angle [deg]': 25,
  'Effective cohesion [kPa]': 0,
  'Undrained shear strength [kPa]': np.nan},
 'Matig gepakt leem- of kleihoudend zand': {'Bulk unit weight above water table [kN/m3]': 17,
  'Bulk unit weight below water table [kN/m3]': 19,
  'Friction angle [deg]': 27,
  'Effective cohesion [kPa]': 0,
  'Undrained shear strength [kPa]': np.nan},
 'Dicht gepakt leem- of kleihoudend zand': {'Bulk unit weight above water table [kN/m3]': 18,
  'Bulk unit weight below water table [kN/m3]': 20,
  'Friction angle [deg]': 30,
  'Effective cohesion [kPa]': 0,
  'Undrained shear strength [kPa]': np.nan},
 'Zeer dicht gepakt leem- of kleihoudend zand': {'Bulk unit weight above water table [kN/m3]': 19,
  'Bulk unit weight below water table [kN/m3]': 20,
  'Friction angle [deg]': 32,
  'Effective cohesion [kPa]': 0,
  'Undrained shear strength [kPa]': np.nan},
 'Weinig vast leem': {'Bulk unit weight above water table [kN/m3]': 16,
  'Bulk unit weight below water table [kN/m3]': 16,
  'Friction angle [deg]': 22,
  'Effective cohesion [kPa]': 0,
  'Undrained shear strength [kPa]': 10},
 'Matig vast leem': {'Bulk unit weight above water table [kN/m3]': 17,
  'Bulk unit weight below water table [kN/m3]': 17,
  'Friction angle [deg]': 22,
  'Effective cohesion [kPa]': 2,
  'Undrained shear strength [kPa]': 25},
 'Vrij vast leem': {'Bulk unit weight above water table [kN/m3]': 18,
  'Bulk unit weight below water table [kN/m3]': 18,
  'Friction angle [deg]': 22,
  'Effective cohesion [kPa]': 4,
  'Undrained shear strength [kPa]': 50},
 'Vast leem': {'Bulk unit weight above water table [kN/m3]': 19,
  'Bulk unit weight below water table [kN/m3]': 19,
  'Friction angle [deg]': 22,
  'Effective cohesion [kPa]': 8,
  'Undrained shear strength [kPa]': 100},
 'Weinig vast zandhoudend leem': {'Bulk unit weight above water table [kN/m3]': 16,
  'Bulk unit weight below water table [kN/m3]': 16,
  'Friction angle [deg]': 25,
  'Effective cohesion [kPa]': 0,
  'Undrained shear strength [kPa]': 10},
 'Matig vast zandhoudend leem': {'Bulk unit weight above water table [kN/m3]': 17,
  'Bulk unit weight below water table [kN/m3]': 17,
  'Friction angle [deg]': 25,
  'Effective cohesion [kPa]': 2,
  'Undrained shear strength [kPa]': 25},
 'Vrij vast zandhoudend leem': {'Bulk unit weight above water table [kN/m3]': 18,
  'Bulk unit weight below water table [kN/m3]': 18,
  'Friction angle [deg]': 25,
  'Effective cohesion [kPa]': 4,
  'Undrained shear strength [kPa]': 50},
 'Vast zandhoudend leem': {'Bulk unit weight above water table [kN/m3]': 19,
  'Bulk unit weight below water table [kN/m3]': 19,
  'Friction angle [deg]': 25,
  'Effective cohesion [kPa]': 8,
  'Undrained shear strength [kPa]': 100},
 'Weinig vast klei': {'Bulk unit weight above water table [kN/m3]': 16,
  'Bulk unit weight below water table [kN/m3]': 16,
  'Friction angle [deg]': 20,
  'Effective cohesion [kPa]': 2,
  'Undrained shear strength [kPa]': 20},
 'Matig vast klei': {'Bulk unit weight above water table [kN/m3]': 17,
  'Bulk unit weight below water table [kN/m3]': 17,
  'Friction angle [deg]': 20,
  'Effective cohesion [kPa]': 4,
  'Undrained shear strength [kPa]': 50},
 'Vrij vast klei': {'Bulk unit weight above water table [kN/m3]': 18,
  'Bulk unit weight below water table [kN/m3]': 18,
  'Friction angle [deg]': 20,
  'Effective cohesion [kPa]': 8,
  'Undrained shear strength [kPa]': 100},
 'Vast klei': {'Bulk unit weight above water table [kN/m3]': 19,
  'Bulk unit weight below water table [kN/m3]': 19,
  'Friction angle [deg]': 20,
  'Effective cohesion [kPa]': 15,
  'Undrained shear strength [kPa]': 200},
 'Weinig vast zandhoudend klei': {'Bulk unit weight above water table [kN/m3]': 16,
  'Bulk unit weight below water table [kN/m3]': 16,
  'Friction angle [deg]': 22,
  'Effective cohesion [kPa]': 2,
  'Undrained shear strength [kPa]': 20},
 'Matig vast zandhoudend klei': {'Bulk unit weight above water table [kN/m3]': 17,
  'Bulk unit weight below water table [kN/m3]': 17,
  'Friction angle [deg]': 22,
  'Effective cohesion [kPa]': 4,
  'Undrained shear strength [kPa]': 50},
 'Vrij vast zandhoudend klei': {'Bulk unit weight above water table [kN/m3]': 18,
  'Bulk unit weight below water table [kN/m3]': 18,
  'Friction angle [deg]': 22,
  'Effective cohesion [kPa]': 8,
  'Undrained shear strength [kPa]': 100},
 'Vast zandhoudend klei': {'Bulk unit weight above water table [kN/m3]': 19,
  'Bulk unit weight below water table [kN/m3]': 19,
  'Friction angle [deg]': 22,
  'Effective cohesion [kPa]': 15,
  'Undrained shear strength [kPa]': 200},
 'Weinig vast veen': {'Bulk unit weight above water table [kN/m3]': 10,
  'Bulk unit weight below water table [kN/m3]': 10,
  'Friction angle [deg]': 15,
  'Effective cohesion [kPa]': 2,
  'Undrained shear strength [kPa]': 10},
 'Matig vast veen': {'Bulk unit weight above water table [kN/m3]': 12,
  'Bulk unit weight below water table [kN/m3]': 12,
  'Friction angle [deg]': 15,
  'Effective cohesion [kPa]': 5,
  'Undrained shear strength [kPa]': 20},
 'Vast veen': {'Bulk unit weight above water table [kN/m3]': 14,
  'Bulk unit weight below water table [kN/m3]': 14,
  'Friction angle [deg]': 15,
  'Effective cohesion [kPa]': 10,
  'Undrained shear strength [kPa]': 40}}

SOILTYPE_COLOR_NATIONALANNEX_MATPLOTLIB = {
    'Matig gepakt grind': '#d3d3d380',
    'Dicht gepakt grind': '#d3d3d3',
    'Matig gepakt leem- of kleihoudend grind': '#a9a9a980',
    'Dicht gepakt leem- of kleihoudend grind': '#a9a9a9',
    'Los gepakt zand': '#fff33966',
    'Matig gepakt zand': '#fff33999',
    'Dicht gepakt zand': '#fff339cc',
    'Zeer dicht gepakt zand': '#fff339',
    'Los gepakt leem- of kleihoudend zand': '#d5ca2b66',
    'Matig gepakt leem- of kleihoudend zand': '#d5ca2b99',
    'Dicht gepakt leem- of kleihoudend zand': '#d5ca2bcc',
    'Zeer dicht gepakt leem- of kleihoudend zand': '#d5ca2b',
    'Weinig vast leem': '#d4c17666',
    'Matig vast leem': '#d4c17699',
    'Vrij vast leem': '#d4c176cc',
    'Vast leem': '#d4c176',
    'Weinig vast zandhoudend leem': '#f2d76a66',
    'Matig vast zandhoudend leem': '#f2d76a99',
    'Vrij vast zandhoudend leem': '#f2d76acc',
    'Vast zandhoudend leem': '#f2d76a',
    'Weinig vast klei': '#ce870d66',
    'Matig vast klei': '#ce870d99',
    'Vrij vast klei': '#ce870dcc',
    'Vast klei': '#ce870d',
    'Weinig vast zandhoudend klei': '#ffac1e66',
    'Matig vast zandhoudend klei': '#ffac1e99',
    'Vrij vast zandhoudend klei': '#ffac1ecc',
    'Vast zandhoudend klei': '#ffac1e',
    'Weinig vast veen': '#5fb32799',
    'Matig vast veen': '#5fb327cc',
    'Vast veen': '#5fb327'}

SOILTYPE_COLOR_NATIONALANNEX_PLOTLY = {
    'Matig gepakt grind': 'rgba(211,211,211,0.5)',
    'Dicht gepakt grind': 'rgba(211,211,211,1.0)',
    'Matig gepakt leem- of kleihoudend grind': 'rgba(169,169,169,0.5)',
    'Dicht gepakt leem- of kleihoudend grind': 'rgba(169,169,169,1.0)',
    'Los gepakt zand': 'rgba(255,243,57,0.4)',
    'Matig gepakt zand': 'rgba(255,243,57,0.6)',
    'Dicht gepakt zand': 'rgba(255,243,57,0.8)',
    'Zeer dicht gepakt zand': 'rgba(255,243,57,1.0)',
    'Los gepakt leem- of kleihoudend zand': 'rgba(213,202,43,0.4)',
    'Matig gepakt leem- of kleihoudend zand': 'rgba(213,202,43,0.6)',
    'Dicht gepakt leem- of kleihoudend zand': 'rgba(213,202,43,0.8)',
    'Zeer dicht gepakt leem- of kleihoudend zand': 'rgba(213,202,43,1.0)',
    'Weinig vast leem': 'rgba(212,193,118,0.4)',
    'Matig vast leem': 'rgba(212,193,118,0.6)',
    'Vrij vast leem': 'rgba(212,193,118,0.8)',
    'Vast leem': 'rgba(212,193,118,1.0)',
    'Weinig vast zandhoudend leem': 'rgba(242,215,106,0.4)',
    'Matig vast zandhoudend leem': 'rgba(242,215,106,0.6)',
    'Vrij vast zandhoudend leem': 'rgba(242,215,106,0.8)',
    'Vast zandhoudend leem': 'rgba(242,215,106,1.0)',
    'Weinig vast klei': 'rgba(206,135,13,0.4)',
    'Matig vast klei': 'rgba(206,135,13,0.6)',
    'Vrij vast klei': 'rgba(206,135,13,0.8)',
    'Vast klei': 'rgba(206,135,13,1.0)',
    'Weinig vast zandhoudend klei': 'rgba(255,172,30,0.4)',
    'Matig vast zandhoudend klei': 'rgba(255,172,30,0.6)',
    'Vrij vast zandhoudend klei': 'rgba(255,172,30,0.8)',
    'Vast zandhoudend klei': 'rgba(255,172,30,1.0)',
    'Weinig vast veen': 'rgba(95,179,39,0.6)',
    'Matig vast veen': 'rgba(95,179,39,0.8)',
    'Vast veen': 'rgba(95,179,39,1.0)'
}