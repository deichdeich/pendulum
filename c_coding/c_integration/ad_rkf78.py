""" This is a Python wrapper for the ad_rkf78.c integrator.
"""

import ctypes
import os
import numpy as np
from numpy.ctypeslib import ndpointer


os.system('gcc -Wall -I/usr/local/include -c ad_rkf78.c -o ad_rkf78.o')
os.system('gcc -L/usr/local/lib ad_rkf78.o -lgslcblas -lgsl -o ad_rkf78.e')

integrator = ctypes.CDLL('./ad_rkf78.e')

pendot = integrator.double_pendulum_eom
intfunc = integrator.rkf78


def rkf78(t0 = 0,
          tmax = 10,
          nsteps = 1000,
          y0 = np.array([3*np.pi/5, 0, np.pi, 0], dtype = np.float64),
          h_init = 0.1,
          tol = 0.001,
          make_section = 0,
          poincare_condition = np.array([0, 1], dtype = np.float64),
          poincare_tolerance = 0.0001):
    
    if nsteps/tmax <= 10:
        raise ValueError("You need more time steps or a shorter time upper-limit")
    
    t0 = ctypes.c_double(t0)
    tmax = ctypes.c_double(tmax)
    
    h = ctypes.c_double(h_init)
    hnext = ctypes.c_double()
    hnext_pt = ctypes.byref(hnext)
    
    y0_pt = y0.ctypes.data_as(ctypes.POINTER(ctypes.c_double))

    tol = ctypes.c_double(tol)

    ms = ctypes.c_int(make_section)
    pc_pt = poincare_condition.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    p_tol = ctypes.c_double(poincare_tolerance)

    hist = np.zeros((nsteps, 5), dtype = np.float64)
    
    hist_pt = hist.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    nsteps = ctypes.c_int(nsteps)

    a = intfunc(pendot,
                t0,
                tmax,
                y0_pt,
                nsteps,
                hist_pt,
                tol,
                h,
                ms,
                pc_pt,
                p_tol)
    
    if make_section:
        nz = np.nonzero(hist)
        hist = hist[nz[0],:]

    return(hist)
