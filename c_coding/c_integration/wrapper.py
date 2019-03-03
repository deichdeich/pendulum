import ctypes
import numpy as np
import os

from numpy.ctypeslib import ndpointer

# compile c program
os.system('gcc -shared -Wl,-install_name,integrator -o integrator.so -fPIC integrator.c')

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



                   
def RKF78(start, end, h, tol, init_state, N_per_call = 10000):
    
    h_pointer = (ctypes.c_double)()
    ctypes.cast(h_pointer, ctypes.POINTER(ctypes.c_double))

    integrator.integrate_rkf78.argtypes = [ctypes.c_double,
                                           ctypes.c_double,
                                           ctypes.c_double,
                                           h_pointer,
                                           ctypes.c_double,
                                           ctypes.c_int,
                                           ndpointer(ctypes.c_double),
                                           ndpointer(ctypes.c_double)]
    full_history = np.zeros((2, N), dtype = np.float64)
    
    while((full_history[-1][0] - end) < tolerance):
        history = np.zeros((2, N), dtype = np.float64)
        integrator.integrate_rkf78(start,
                                   end,
                                   h,
                                   h_pointer,
                                   tol,
                                   N_per_call,
                                   init_state,
                                   history)
        full_history = _copy_history(full_history, history)
    
    return(full_history)
        

def _copy_history(full_history, new_history):
    full_len = len(full_history)
    new_len = len(new_history)
    new_full = np.zeros((2, new_len + full_len), dtype=np.float64)
    new_full[:, full_len] = full_history
    new_full[:, full_len:new_len] = new_history
    return(new_full)
    
    
    
    
    
    
    
    
    