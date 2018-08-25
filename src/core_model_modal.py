# -*- coding: utf-8 -*-
"""
1-D model
Module Name : Core
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
                Dilution_type = 'Force', H_Dissipate = False, V_Sedimentation = 'Concentration', V_Transportation = False, Diffusion_Coef = "Huang"):
    """
    This is a pde_solver for solving diffusion equation

    use Crank-Nicolson method
    webpage: georg.io/2013/12/Crank_Nicolson
             en.wikipedia.org/wiki/Crank%E2%80%93Nicolson_method

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
    exceed_flag = 0

    modal_number = 3
    delta_z = 1 #units : level
    delta_t = 1 #units : month
    delta_y = 1

    #Concentration in Reservoir
    c_exchange = np.zeros((modal_number, time_step, level))

    if Diffusion_Coef == "Huang":
        #Vertical Eddy Diffusion Coefficient D_zz
        #Huang(1979) parameterization method
        #b : coefficient
        #n : power of the formula
        #D_zz = b * (level * 500)**n
        b = 1e-1
        n = 0.4
        D_zz_generator = lambda z : b * (z**n)
        level_height = np.linspace(0,80,80)
        D_zz = D_zz_generator(level_height)
        for modal in range(modal_number):
            D_zz = np.vstack((D_zz, D_zz_generator(level_height)))

    if Diffusion_Coef == "Constant":
        print("Constant")
        D_zz = np.full((modal_number, level), 1e-1)

    #Concatenate to expand wind into many years
    for i in range(int(time_step / 12)):
        v = np.concatenate((v, v), axis = 0)
    for i in range(int(time_step / 12)):
        w = np.concatenate((w, w), axis = 0)

    #Different Modal
    for modal in range(modal_number):
        #~~~~~~~Solve Part
        for time in range(0, time_step - 1):
            cross_term = lambda x : (w[time, x] * delta_z * delta_t)/2 
        # Init A ------>  time + 1
            A =     np.diagflat([-1 * D_zz[modal, i] * delta_t - cross_term(i) for i in range(level - 1)], -1)
            A = A + np.diagflat([-1 * D_zz[modal, i] * delta_t + cross_term(i) for i in range(level - 1)], 1)
            A = A + np.diagflat([D_zz[modal, 0] * delta_t + 2 * (delta_z)**2 - cross_term(0)] \
            + [2 * D_zz[modal, i] * delta_t + 2 * (delta_z)**2 for i in range(level - 2)] \
            + [D_zz[modal, level - 1] * delta_t + 2 * (delta_z)**2 + cross_term(level - 1)], 0)

        # Init B ------> time
            B =     np.diagflat([D_zz[modal, i] * delta_t + cross_term(i) for i in range(level - 1)], -1)
            B = B + np.diagflat([D_zz[modal, i] * delta_t - cross_term(i) for i in range(level - 1)], 1)
            B = B + np.diagflat([-1 * D_zz[modal, 0] * delta_t + 2 * (delta_z)**2 + cross_term(0)] \
            + [-2 * D_zz[modal, i] * delta_t + 2 * (delta_z)**2 for i in range(level - 2)] \
            + [-1 * D_zz[modal, level - 1] * delta_t + 2 * (delta_z)**2 - cross_term(level - 1)], 0)

            #U for dissipate process
            #Exchange Aerosol with a zero-initial reservoirs
            #Solve use numpy linalg module
            delta_c = c[modal, time, :] - c_exchange[modal, time, :]
            if all(delta_c >= 0):
                c_new = np.linalg.solve(A, B.dot(c[modal, time, :]) - 2 * delta_t * (delta_z)**2 * v[time, :] * delta_c / delta_y)
                if any(c_new < 0) :
                    c[modal, time + 1, :] = 0 * c[modal, time + 1, :]
                else :
                    c[modal, time + 1, :] = c_new
                c_exchange[modal, time + 1, :] = (c_exchange[modal, time, :] + delta_c * v[time, :] / delta_y)
            else:
                c[modal, time + 1, :] = np.linalg.solve(A, B.dot(c[modal, time, :]))

            del A, B

        #~~~~~~~Physical Process Part
            # Horizontal dissipation
            if H_Dissipate == True:
                # Unit transist from m/s to m/month
                dis_rate = np.full((modal_number), 1)
                for i in range(level):
                    if c[modal, time + 1, i] > 0:
                        h_flux =  v[time + 1, i] * dis_rate[modal]
                        if c[modal, time + 1, i] - h_flux > 0:
                            c[modal, time + 1, i] = c[modal, time + 1, i] - h_flux
                        else:
                            c[modal, time + 1, i] = 0
                    if c[modal, time + 1, i] < 0:
                        c[modal, time + 1, i] = 0

            # Vertical Sedimentation
            if V_Sedimentation == 'Concentration':
                for i in range(1, level):
                    delta_c_vertical = c[modal, time + 1, i] - c[modal, time + 1, i - 1]
                    v_coeff = 0.6
                    if delta_c_vertical > 0:
                        c[modal, time + 1, i - 1] = c[modal, time + 1, i - 1] + delta_c_vertical * v_coeff
                        c[modal, time + 1, i] = c[modal, time + 1, i] - delta_c_vertical * v_coeff
                del i

            if V_Sedimentation == 'Height':
                for i in range(1, level):
                    delta_c_vertical = c[modal, time + 1, i] - i
                    v_coeff = 0.07
                    if delta_c_vertical > 0:
                        c[modal, time + 1, i - 1] = c[modal, time + 1, i - 1] + i * v_coeff
                        c[modal, time + 1, i] = c[modal, time + 1, i] - i * v_coeff
                del i

            # Vertical Transportation
            if V_Transportation == True:
                for i in range(1, level - 1):
                    delta_c_up = np.absolute(c[modal, time + 1, i + 1] - c[modal, time + 1, i])
                    delta_c_down = np.absolute(c[modal, time + 1, i - 1] - c[modal, time + 1, i])
                    vertical_budget = delta_c_up * -1 * w[time + 1, i + 1] + delta_c_down * w[time + 1, i]
                    c[modal, time + 1, i] = c[modal, time + 1, i] + 0.2 * vertical_budget

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

            #Restriction : To prevent the infinite diverge of c
            #exceed_cond = any(c[modal, time, :] > level * 100 * modal) or \
            #any(c[modal, time, :] < 0)
            exceed_cond = any(c[modal, time, :] < 0)
            if exceed_cond:
                exceed_flag = 1
                break

        #Check to avoid the singularity
        if exceed_flag == 1:
            #c[modal, time, :] = 0 * c[modal, time, :]
            print("Warning : Data has exceed until " + str(time))
            break

           
    c_total = np.sum(c, axis = 0)

    return c_total
