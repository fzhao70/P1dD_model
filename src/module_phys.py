# -*- coding: utf-8 -*-
"""
1-D Diffusion Advection model
Module Name : Core

A PDE Solver with crank-nicolson

Fanghe @ gatech MoSE 3230

Version:
    + python => 3.5
    + Anaconda recommend
USTC-AEMOL
Gatech-Apollo
"""

import numpy as np
import random as rand

def Dilution(c, time, modal, TTL, Dilution_type = 'Force'):
    """
    This function defined how to behave Dilution

    Args:
    ---------
    c : numpy array
    time : integer
    modal : integer
    TTL : integer
    Dilution_type : String
                    Option :'Force', 'Smooth', 'Stochastic', 'None'

    Returns:
    ---------
    c : numpy array
    """
    # Dilution
    # Force Dilution
    if Dilution_type == 'Force':
        # top of troposphere = 15000m ~ 30 lev
        c[modal, time + 1, 0:TTL] = 0
        if time == 0:
            c[modal, time, 0:TTL] = 0
    # Stochastic Force Dilution
    if Dilution_type == 'Stochastic':
        rand.seed(time)
        if rand.random() > 0.1:
            c[modal, time + 1, 0:TTL] = 0
    # Smooth Dilution
    if Dilution_type == 'Smooth':
        for i in range(0,TTL - 2):
            c[modal, time + 1, TTL - i] = c[modal, time + 1, TTL - i + 1] * 0.001
    # None Dilution
    if Dilution_type == 'None':
        c[modal, time + 1, :] = c[modal, time + 1, :] * 1.0

    return c

def Sedimentation(c, time, modal, level, Sid_type = 'Concentration'):
    """
    Vertical Sedimentation

    Args:
    ----------
    c : numpy array
    time : integer
    modal : integer
    level : integer
    Sid_type : String
               Option : 'Concentration', 'Height'

    Returns:
    ----------
    c : numpy array
    """
    if Sid_type == 'Concentration':
        for i in range(1, level):
            delta_c_vertical = c[modal, time + 1, i] - c[modal, time + 1, i - 1]
            v_coeff = 0.6
            if delta_c_vertical > 0:
                c[modal, time + 1, i - 1] = c[modal, time + 1, i - 1] + delta_c_vertical * v_coeff
                c[modal, time + 1, i] = c[modal, time + 1, i] - delta_c_vertical * v_coeff
        del i

    if Sid_type == 'Height':
        for i in range(1, level):
            delta_c_vertical = c[modal, time + 1, i] - i
            v_coeff = 0.07
            if delta_c_vertical > 0:
                c[modal, time + 1, i - 1] = c[modal, time + 1, i - 1] + i * v_coeff
                c[modal, time + 1, i] = c[modal, time + 1, i] - i * v_coeff
        del i

    return c

def H_Dissipation(c ,time, modal, level, dis_rate):
    """
    Horizontal dissipation

    Args:
    ----------
    c : numpy array
    time : integer
    modal : integer
    level : integer
    dis_rate : numpy with size (3,)

    Returns:
    ----------
    c : numpy array
    """
    for i in range(level):
        if c[modal, time + 1, i] > 0:
            h_flux =  v[time + 1, i] * dis_rate[modal]
            if c[modal, time + 1, i] - h_flux > 0:
                c[modal, time + 1, i] = c[modal, time + 1, i] - h_flux
            else:
                c[modal, time + 1, i] = 0
        if c[modal, time + 1, i] < 0:
            c[modal, time + 1, i] = 0

    return c

def V_Transportation(c, time, modal, level):
    """
    Vertical Transportation

    Args:
    ----------
    c : numpy array
    time : integer
    modal : integer
    level : integer

    Returns:
    ----------
    c : numpy array
    """
    for i in range(1, level - 1):
        delta_c_up = np.absolute(c[modal, time + 1, i + 1] - c[modal, time + 1, i])
        delta_c_down = np.absolute(c[modal, time + 1, i - 1] - c[modal, time + 1, i])
        vertical_budget = delta_c_up * -1 * w[time + 1, i + 1] + delta_c_down * w[time + 1, i]
        c[modal, time + 1, i] = c[modal, time + 1, i] + 0.2 * vertical_budget

    return c
