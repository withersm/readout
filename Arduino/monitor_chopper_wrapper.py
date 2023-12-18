import ctypes as ctypes
so_file = "/home/matt/readout/Arduino/monitor_chopper_for_py.so"
my_func = ctypes.cdll.LoadLibrary(so_file)


outputfile = "test_output.txt"
devicepath = "/dev/ttyACM4"

my_func.run.argtypes = [ctypes.c_char_p, ctypes.c_char_p]


my_func.run(outputfile.encode('utf-8'), devicepath.encode('utf-8'))


