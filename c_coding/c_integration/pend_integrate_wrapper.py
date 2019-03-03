import ctypes
import os

from numpy.ctypeslib import ndpointer

os.system('gcc -shared -Wl,-install_name,integrator -o integrator.so -fPIC integrator.c')

integrator = ctypes.CDLL('./integrator.so')

sinfunc = integrator.f1
func2 = integrator.f2
func3 = integrator.f3
feldfunc = integrator.fehlberg
intfunc = integrator.integrate_rkf78



def rkf78(x = 0, xmax = np.pi, y0 = 0, N_x = 1000, h_init = 0.1, tol = 0.001):
    x = ctypes.c_double(x)
    xmax = ctypes.c_double(xmax)
    
    h = ctypes.c_double(h_init)
    hnext = ctypes.c_double()
    hnext_pt = ctypes.byref(hnext)
    
    y0 = ctypes.c_double(y0)
    
    outvec = np.zeros(2, dtype = np.float64)
    outvec_pt = outvec.ctypes.data_as(ctypes.POINTER(ctypes.c_double))

    tol = ctypes.c_double(tol)

    hist = np.zeros((N_x, 2), dtype = np.float64)
    hist_pt = hist.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    
    xvals = np.zeros(N_x, dtype = np.float64)
    xvals_pt = xvals.ctypes.data_as(ctypes.POINTER(ctypes.c_double))


    a = intfunc(func2, x, xmax, y0, N_x, hist_pt, tol, h)
    #a = feldfunc(sinfunc, outvec_pt, x, h, xmax, hnext_pt, tol)

    return(hist)