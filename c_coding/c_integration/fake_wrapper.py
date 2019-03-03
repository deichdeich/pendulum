import ctypes
import numpy as np
import os

from numpy.ctypeslib import ndpointer

# compile c program
os.system('gcc -shared -Wl,-install_name,integrator -o integrator.so -fPIC f_integrator.c')

# load the c program
integrator = ctypes.CDLL('./integrator.so')



def RK4(t0, N, dt, init_state):
    
    
    integrator.integrate_rk4.argtypes = [ctypes.c_double,
                                        ctypes.c_int,
                                        ctypes.c_double,
                                        ndpointer(ctypes.c_double),
                                        ndpointer(ctypes.c_double)]
    
    history = np.zeros((2, N), dtype = np.float64)
    
    integrator.integrate_rk4(t0, N, dt, init_state, history)
    return(history)