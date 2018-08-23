# -*- coding: utf-8 -*-
"""
1-D model
Module Name : Graph
Graph module for 2-D data

Fanghe @ gatech MoSE 3229

Version:
    + python => 3.5
    + Anaconda recommend
USTC-AEMOL
Gatech-Apollo
"""

import numpy as np
import matplotlib.pyplot as plt

    
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
        plt.xlim((0,time_step - 1))
        plt.ylim((20,79))
        #plt.yticks(np.linspace(0,80,80),np.linspace(0,40000,80).astype(int))
        plt.title('Time Series of Atlitude')
        plt.show()
    if fig_type == "pixel":
        plt.imshow(data ,interpolation='nearest')
        plt.gca().invert_yaxis()
        plt.title('Time Series of Atlitude')
        plt.xlim((0,time_step - 1))
        plt.ylim((20,79))
        plt.show()

    return 0
