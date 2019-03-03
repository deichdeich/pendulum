import ctypes
import numpy as np
import os

# compile c program
os.system('gcc -shared -Wl,-install_name,testlib -o testlib.so -fPIC ctypes_test.c')

# load the c program
testlib = ctypes.CDLL('./testlib.so')

# create some data
data = np.arange(10, dtype = np.float64)

# copy it
temp = np.copy(data)

# don't know what this does
temp_ctype = temp.ctypes

# this edits the array in-place.  temp_ctype is more of a memory address.
testlib.arr_fucker(temp_ctype, 10)

data = temp

# delete temporary array
del temp