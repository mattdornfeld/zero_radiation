import numpy as np
from math import pi
from parameters import *

def phi(x, omega):
	return np.cos(x*omega)

dx = 0.001 #discrete spatial step
x = np.arange(0, 1, dx)

phi_matrix = np.transpose(np.array([phi(x, w) for w in omega]))
phi_inverse = np.linalg.pinv(phi_matrix)
a = np.sum(phi_inverse, 1)
b = np.dot(phi_inverse,0.5*(x-1)**2)