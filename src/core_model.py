# -*- coding: utf-8 -*-
"""
1-D model
Module Name :Core
Only Diffusion included
A PDE Solver

Fanghe @ gatech MoSE 3229

Version:
    + python => 3.5
    + Anaconda recommend
USTC-AEMOL
Gatech-Apollo
"""

import numpy as np
import random as rand 

def pde_solver(time_step, level, c, v, TTL = 30, Dilution_type = 'Force', H_Dissipate = True):
    """
    This is a pde_solver for solving diffusion equation

    use Crank-Nicolson method
    webpage: georg.io/2013/12/Crank_Nicolson

     C_t = D * C_zz + f(z)
          ||
           V
     A * U^(n+1) = B * U^(n) + f^(n)

     From PDE to Linear equation system
    
    Args :
    -------------
    time_step, level : integer
    c_aerosol : concentration of aerosol [time_step, level]
    v : horizontal [level], mainly Zonal Wind
    TTL : troposphere_top_lev default is 25 
    Dilution_type :options : 'Force', 'Smooth', 'Stochastic'
    H_Dissipate : Default is True
    
    Return :
    -------------
    concentration in arrays
    """
    assert (time_step > 0 and level > 0), "Initial level should be positive"
    #Multiply v to fix the years of wind
    for i in range(int(time_step / 12)):
        v = np.concatenate((v, v), axis = 0)

    # sigma_c = D * delta_t / 2 * (delta_x)^2
    # delta_t = 1 month delta_x = 500 meter
    # True Value : sigma_c = 2.5e-4
    sigma_c =  2.5e-4
    dis_rate = 1.2

    #~~~~~~~Init Part
    # Init A
    A = np.diagflat([-sigma_c for i in range(level - 1)], -1)
    A = A + np.diagflat([-sigma_c for i in range(level - 1)], 1)
    A = A + np.diagflat([1 + sigma_c] + [1 + 2 * sigma_c for i in range(level - 2)] + [1 + sigma_c], 0)
    # Init B
    B = np.diagflat([sigma_c for i in range(level - 1)], -1)
    B = B + np.diagflat([sigma_c for i in range(level - 1)], 1)
    B = B + np.diagflat([1 - sigma_c] + [1 - 2 * sigma_c for i in range(level - 2)] + [1 - sigma_c], 0)
    #u for dissipate process

    #~~~~~~~Solve Part
    #Solve use np linalg module
    for time in range(0, time_step - 1):
        #Different way to induce the dissipate , f is a lambda function about elements
        #c[time + 1, :] = np.linalg.solve(A, B.dot(c[time, :]) + list(map(f, c[time,:])))
        #c[time + 1, :] = np.linalg.solve(A, B.dot(c[time, :]) + dis_rate * sigma_c * v)
        c[time + 1, :] = np.linalg.solve(A, B.dot(c[time, :]))

    #~~~~~~~Physical Process Part
        # Force Horizontal dissipation
        if H_Dissipate == True:
            for i in range(80):
                if c[time + 1, i] > 0:
                    c[time + 1, i] = c[time + 1, i] - v[time, i] * dis_rate
                if c[time + 1, i] < 0:
                    c[time + 1, i] = 0
        # Force Dilution
        if Dilution_type == 'Force':
            # top of troposphere = 15000m ~ 30 lev
            c[time + 1, 0:TTL] = 0
        # Stochastic Force Dilution
        if Dilution_type == 'Stochastic':
            rand.seed(time)
            if rand.random() > 0.1:
                c[time + 1, 0:TTL] = 0
        # Smooth Dilution
        if Dilution_type == 'Smooth':
            for i in range(0,TTL - 2):
                c[time + 1, TTL - i] = c[time + 1, TTL - i + 1] * 0.001
        if Dilution_type == 'None':
            c[time + 1, :] = c[time + 1, :] * 1.0
            

    return c
