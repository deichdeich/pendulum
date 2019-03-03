import ctypes

pt = ctypes.CDLL('./pointer_test.so')

pointer_test_func = pt.pointer_test_function
test_func = pt.test_func

x = ctypes.c_int(5)
y = ctypes.c_int(6)
z = ctypes.c_int()

z_point = ctypes.byref(z)

pointer_test_func(test_func, x, y, z_point)
print(z.value)