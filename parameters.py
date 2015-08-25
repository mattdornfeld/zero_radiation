import numpy as np
from math import pi

nm = 2 #number of modes
c = 343 # speed of sound
rho = 1.27 # density of air

L = 0.004 #length of upper vocal tract
r_pharynx = 0.003 # radius of pharynx
r_trachea = 0.003 # radius of trachea

fs = 3. #time sampling frequency
T = 3000 #total time
t = np.arange(0, T, 1/fs)

p_range = np.arange(0.0001, 0.015, 0.0001)
r_front_range = np.arange(0.0001, 0.002, 0.0001)
r_back_range = np.arange(0.0001, 0.002, 0.0001)

omega = np.array([ (2 * n + 1) * pi / 2 for n in range(nm)])

params = {'nm':nm, 'L':L, 'r_pharynx':r_pharynx, 
'r_trachea':r_trachea, 'fs':fs, 'T':T}