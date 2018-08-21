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
#import user fuction
import init_model
import core_model
import core_model_modal

    
def graph_output(data, fig_type = "contour"):
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
        plt.xlim((0,40))
        plt.ylim((20,79))
        #plt.yticks(np.linspace(0,80,80),np.linspace(0,40000,80).astype(int))
        plt.title('Time Series of Atlitude')
        plt.show()
    if fig_type == "pixel":
        plt.imshow(data ,interpolation='nearest')
        plt.gca().invert_yaxis()
        plt.title('Time Series of Atlitude')
        plt.xlim((0,100))
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
    #Initial Vars
    TTL = 30
    time_step = 60
    level = 80
    #Initial Condition
    c_in = init_model.initial_vars(time_step, sigma = 0.2, mu = 38, level = 80)
    # Set stratosphere horizon wind
    v = np.absolute(init_model.read_in())
    #v[:, 0:TTL] = 0
    #v_after = v[8:, :]
    #v_before = v[0:8, :]
    #v = np.concatenate((v_after, v_before), axis = 0)

    
    print("Initial Part Succeed")
    #------------------
    # Computation part
    #------------------
    print("Computational Part Start")
    c_out = core_model.pde_solver(time_step, level, c_in, v)
    print("Computational Part Succeed")

    #------------------
    # Output part
    #------------------
    c_out = np.transpose(c_out)
    graph_output(c_out)
