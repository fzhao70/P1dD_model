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

def pde_solver(time_step, level, c, v, w, TTL = 25, \
                Dilution_type = 'Force', H_Dissipate = True, V_Sedimentation = False, V_Transportation = False):
    """
    This is a pde_solver for solving diffusion equation

    use Crank-Nicolson method
    webpage: georg.io/2013/12/Crank_Nicolson

     C_t = D * C_zz - w * C_z + f(z)
          ||
           V
     A * U^(n+1) = B * U^(n) + f^(n)

     From PDE to Linear equation system
    
    Args :
    -------------
    time_step, level : integer
    c_aerosol : concentration of aerosol [time_step, level]
    v : horizontal [level], mainly Zonal Wind
    w : vertical [level + 1]
    TTL : troposphere_top_lev default is 25 
    Dilution_type :options : 'Force', 'Smooth', 'Stochastic'
    H_Dissipate : Default is True
    
    Return :
    -------------
    concentration in arrays
    """
    assert (time_step > 0 and level > 0), "Initial level should be positive"


    # sigma_c = D * delta_t / 2 * (delta_x)^2
    # delta_t = 1 month delta_x = 500 meter
    # True Value : sigma_c = 2.5e-4
    modal_number = 3
    D_zz = np.full((modal_number), 1e-6)
    delta_z = 500
    delta_t = 1
    # Unit transist from m/s to m/month
    dis_rate = np.full((modal_number), 30 * 86400)

    #Multiply v to fix the years of wind
    for i in range(int(time_step / 12)):
        v = np.concatenate((v, v), axis = 0)
    for i in range(int(time_step / 12)):
        w = np.concatenate((w, w), axis = 0)
    #Different Modal
    for modal in range(modal_number):
        #~~~~~~~Solve Part
        for time in range(0, time_step - 1):
        # Init A
            A = np.diagflat([-1 * delta_z * delta_t * w[time, i] - D_zz[modal] * delta_t  for i in range(level - 1)], -1)
            A = A + np.diagflat([delta_z * delta_t * w[time, i + 1] for i in range(level - 1)], 1)
            A = A + np.diagflat(\
            [-2 * delta_z * delta_t * w[time, 0] + D_zz[modal] * delta_t - 2 * (delta_z)**2]\
            + [-2 * D_zz[modal] * delta_t - 2 * (delta_z)**2 for i in range(level - 2)]\
            + [D_zz[modal] * delta_t - 2 * (delta_z)**2 + delta_z * delta_t * w[time, 80]], 0)

            # Init B
            B = np.diagflat([D_zz[modal] * delta_t for i in range(level - 1)], -1)
            B = B + np.diagflat([2 * D_zz[modal] * delta_t  for i in range(level - 1)], 1)
            B = B + np.diagflat(\
            [-1 * D_zz[modal] * delta_t - 2 * (delta_z)**2] \
            + [-2 * D_zz[modal] * delta_t - 2 * (delta_z)**2 for i in range(level - 2)] \
            + [-2 * (delta_z)**2], 0)

            #u for dissipate process
            #Solve use np linalg module
            c[modal, time + 1, :] = np.linalg.solve(A, B.dot(c[modal, time, :]))

        #~~~~~~~Physical Process Part
            # Force Horizontal dissipation
            if H_Dissipate == True:
                for i in range(80):
                    if c[modal, time + 1, i] > 0:
                        c[modal, time + 1, i] = c[modal, time + 1, i] - v[time, i] * dis_rate[modal]
                    if c[modal, time + 1, i] < 0:
                        c[modal, time + 1, i] = 0

            # Vertical Sedimentation
            if V_Sedimentation == True:
                for i in range(1, 80):
                    delta_c = np.absolute(c[modal, time + 1, i - 1] - c[modal, time + 1, i])
                    c[modal, time + 1, i - 1] = c[modal, time + 1, i - 1] + delta_c * 0.8
                    c[modal, time + 1, i] = c[modal, time + 1, i] - delta_c * 0.8

            # Vertical Transportation
            if V_Transportation == True:
                for i in range(1, 79):
                    delta_c_up = np.absolute(c[modal, time + 1, i + 1] - c[modal, time + 1, i])
                    delta_c_down = np.absolute(c[modal, time + 1, i - 1] - c[modal, time + 1, i])
                    vertical_budget = delta_c_up * -1 * w[time + 1, i + 1] + delta_c_down * w[time + 1, i]
                    c[modal, time + 1, i] = c[modal, time + 1, i] + 0.2 * vertical_budget

            # Dilution
            # Force Dilution
            if Dilution_type == 'Force':
                # top of troposphere = 15000m ~ 30 lev
                c[modal, time + 1, 0:TTL] = 0
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

        c_total = np.sum(c, axis = 0)

    return c
