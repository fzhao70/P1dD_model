# -*- coding: utf-8 -*-
"""
1-D model
Module Name : init_model
Module function : read_in
Read-in Part
Simple Data-Assimilation

ATTENTION:
UNDERSTAND THE DATA BEFORE YOU USE IT ----NCAR

Fanghe @ gatech MoSE 3229

Version:
    + python => 3.5
    + Anaconda recommend
    + py netCDF4

USTC-AEMOL
Gatech-Apollo
"""

from scipy.interpolate import interp1d
import numpy as np
import functools as ft
import netCDF4 as nc
import matplotlib.pyplot as plt

def read_in(file_name = "ECMWF_2000.nc"):
    """
    Function 1: read data as input
    Read reanalysis data to initialize V field

    Args:
    ---------
    file_name : a string default is "ECMWF_2001.nc"

    Return:
    ---------
    v : 2-d array for initialize v
    """
    #Private Args Configuration
    std_height = np.linspace(0, 40000, 80)

    #Open File and Retrieve Data from it
    #Pay attention:
    #1. netCDF4 can not use with...as....: to open the file
    #2. Do not close the file before u use the variables
    #3. Retrieve you data you exactly need
    Reanalysis_fid = nc.Dataset(file_name, "r")

    level = np.array(Reanalysis_fid.variables['level'][:])
    v = np.array(Reanalysis_fid.variables['v'][:, :, :, :])
    lat = Reanalysis_fid.variables['latitude'][:]
    z = np.array(Reanalysis_fid.variables['z'][:, : ,: ,:])
    w = np.array(Reanalysis_fid.variables['w'][:, : ,: ,:])

    #Latitude Filter
    #Only use boundary stripe wind
    #Attention to the boundary condition of the wind
    cond_up = np.logical_and((lat < 16) ,(lat > 14))
    cond_down = np.logical_and((lat < -14) ,(lat > -16))
    cond = np.logical_or(cond_up, cond_down)
    lat_index = np.where(cond)
    lat_index_up = np.where(cond_up)
    lat_index_down = np.where(cond_down)

    v_up = v[:, :, lat_index_up, :]
    v_down = v[:, :, lat_index_down, :]
    #v_up = np.where(v_up > 0, v_up, 0)
    #v_down = np.where(v_down < 0, v_down, 0)
    v = np.average(np.concatenate((v_up, v_down), axis = 2), 2)
    #v = np.where(v > 0, v, 0)

    z = z[:, :, lat_index, :]
    z = np.average(z, 2)
    w = w[:, :, lat_index, :]
    w = np.average(w, 2)
    #Longitude Average
    #After Selection dimension will add one dummy dimension
    z = np.average(z, 3)
    w = np.average(w, 3)
    v = np.average(v, 3)
    #Latitude Average
    v = np.average(v, 2)
    w = np.average(w, 2)
    z = np.average(z, 2)
    z = z / 9.8

    #Attention w_0 is on the vertex
    #
    #  -----w_0-----    top w_0 = 0
    #       ...
    #  -----w_0-----    3
    #    v_0, c_0     2
    #  -----w_0-----    2
    #    v_0, c_0     1
    #  -----w_0-----    1
    #    v_0, c_0     0       std_height
    #  -----w_0-----    0     std_vertex
    #
    #
    std_vertex_list = list(range(750, 40750, 500))
    std_vertex_list.insert(0, 0)
    std_vertex = np.array(std_vertex_list)

    v_0 = np.zeros((12, len(std_height)))
    w_0 = np.zeros((12, len(std_vertex)))
    #Interpolate z to Standard Level
    #Below the Stratosphere is all setted to zero
    for month in range(12):
        f_v = interp1d(z[month,:], v[month,:])
        f_w = interp1d(z[month,:], w[month,:])
        cond = lambda x : x > min(z[month,:])
        v_temp = np.zeros(len(std_height))
        v_temp[-len(list(filter(cond, std_height))): -1] = f_v(std_height[-len(list(filter(cond, std_height))) : -1])
        w_temp = np.zeros(len(std_vertex))
        w_temp[-len(list(filter(cond, std_vertex))): -1] = f_w(std_vertex[-len(list(filter(cond, std_vertex))) : -1])
        v_0[month, :] = v_temp
        w_0[month, :] = w_temp

    #Close file
    Reanalysis_fid.close()

    w_0 = np.absolute(w_0)
    v_0 = np.absolute(v_0)
    v_0 = v_0 * 0.7
    w_0 = w_0 * 0.7

    return v_0, w_0

def initial_vars(time_step, sigma = 0.20, mu = 38, level = 80, TTL = 25):
    """
    This function is mean to initialize vars and give them IC

    Args:
    ---------------
    Sigma : Control the range of the value
    Mu : centre of distribution
    level : the max level of the model, default is 80
    time_step : the total time steps of this model

    Return:
    ---------------
    C: 2-d array with Normal distribution

    """
    #troposphere_top_lev

    normal_dis = lambda x : (1 / (sigma * np.sqrt(2 * np.pi))) * \
            np.exp(-1 * (x - mu)**2 / 2 * sigma**2)
    c_0 = normal_dis(np.linspace(0, 79, 80))
    c = np.zeros([time_step, level])
    c[0, :] = c_0
    c[0, 0:TTL] = 0

    return c

if __name__ == '__main__':
    """
    This part is JUST FOR TEST
    """
    dum = read_in()
    print("End")
