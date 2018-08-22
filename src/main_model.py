# -*- coding: utf-8 -*-
"""
1-D model
Module Name : Main
Only Diffusion included

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
#import user fuction
import init_model
import core_model

    
def graph_output(data, time_step, fig_type = "contour"):
    """
    This function is mean to initialize vars and give them IC

    Args:
    ---------------
    data : data for plot ,2-d Only

    Return:
    ---------------
    Nan

    """

    assert(data.shape),"NO DATA"
    if fig_type == "contour":
        plt.style.context('Solarize_Light2')
        plt.contourf(data)
        plt.xlabel('Time')
        plt.ylabel('level')
        plt.colorbar()
        plt.xlim((0,time_step))
        plt.ylim((20,79))
        #plt.yticks(np.linspace(0,80,80),np.linspace(0,40000,80).astype(int))
        plt.title('Time Series of Atlitude')
        plt.show()
    if fig_type == "pixel":
        plt.imshow(data ,interpolation='nearest')
        plt.gca().invert_yaxis()
        plt.title('Time Series of Atlitude')
        plt.xlim((0,time_step))
        plt.ylim((20,79))
        plt.show()

    return 0

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Main Procedure Start
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
if __name__ == '__main__':

    #--------------------
    #Initial vars part
    #--------------------
    print("Initial Part start")
    time0 = time.time()
    #Initial Vars
    TTL = 25
    time_step = 60
    level = 80
    modal_number = 3
    #Initial Condition
    c_0 = init_model.initial_vars(time_step, sigma = 0.12, mu = 38, level = level)
    c_in = np.zeros((modal_number, time_step, level))
    for i in range(modal_number):
        c_in[i, :, :] = c_0
    # Set stratosphere horizon wind
    v, w = init_model.read_in()
    v = np.absolute(v)
    v[:, 0:TTL] = 0
    v = np.concatenate((v[8:, :], v[0:8, :]), axis = 0)
    w = np.concatenate((w[8:, :], w[0:8, :]), axis = 0)

    
    print("Initial Part Succeed")
    time_used = time.time() - time0 
    print("Initial Process uses : " + str(time_used) + "s")
    #------------------
    # Computation part
    #------------------
    print("Computational Part Start")
    time0 = time.time()
    c_out = core_model_modal.pde_solver(time_step, level, c_in, v, w)
    print("Computational Part Succeed")
    time_used = time.time() - time0 
    print("Computational Process uses : " + str(time_used) + "s")

    #------------------
    # Output part
    #------------------
    c_out = np.transpose(c_out)
    graph_output(c_out, time_step)
