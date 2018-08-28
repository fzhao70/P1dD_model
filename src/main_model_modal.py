# -*- coding: utf-8 -*-
"""
1-D Diffusion Advection model

Module Name : Main
This is entrance of the whole program

Fanghe @ gatech MoSE 3229

Version:
    + python => 3.5
    + Anaconda recommend
USTC-AEMOL
Gatech-Apollo
"""

import numpy as np
import matplotlib.pyplot as plt
import time
import configparser
import os
#import user module
import init_model
import graph_model
import core_model_modal
    
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Main Procedure Start
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
if __name__ == '__main__':

    #--------------------
    #Initial vars part
    print("Initial Part start")
    time0 = time.time()
    #--------------------
    config = configparser.ConfigParser()
    config.read("namelist.ini")
    #Section : model
    TTL = int(config.get("vars", "TopTroposphereLayer"))
    time_step = int(config.get("vars", "time_step"))
    level = int(config.get("vars", "level"))
    modal_number = int(config.get("vars", "modal_number"))
    c_max = int(config.get("vars", "c_max"))
    #Section : switch
    Dilution = config.get("switch", "Dilution")
    Sedimentation = config.get("switch", "Sedimentation")
    Diffusion = config.get("switch" ,"Diffusion")
    Dissipate = bool(int(config.get("switch", "Dissipate")))
    Transport = bool(int(config.get("switch", "Transport")))


    #Initial Condition
    c_0 = init_model.initial_vars(time_step, sigma = 0.25, mu = 38, level = level)
    c_in = np.zeros((modal_number, time_step, level))
    c_max = 1e2
    for i in range(modal_number):
        c_in[i, :, :] = c_max * c_0 # units : mol
    # Set stratosphere horizon vertical wind
    # w : 1e-2 ~ 1e-7
    # v : 1    ~ 1e-2
    v, w = init_model.read_in()

    #Mt Pinatubo Eruption occurs in June
    v = np.concatenate((v[6:, :], v[0:6, :]), axis = 0)
    w = np.concatenate((w[6:, :], w[0:6, :]), axis = 0)
    
    print("Initial Part Succeed")
    time_used = time.time() - time0 
    print("Initial Process uses : " + str(time_used) + "s")
    #------------------
    # Computation part
    print("Computational Part Start")
    time0 = time.time()
    #------------------

    c_out = core_model_modal.pde_solver(time_step, level, c_in, v, w, TTL = TTL)
    
    print("Computational Part Succeed")
    time_used = time.time() - time0 
    print("Computational Process uses : " + str(time_used) + "s")
    #------------------
    # Output part
    #------------------

    c_out = np.transpose(c_out[:, :])
    graph_model.graph_output(c_out, time_step, fig_type = 'contour')
