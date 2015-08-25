from itertools import izip_longest
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from scipy.integrate import odeint
from math import pi
from joblib import Parallel, delayed
from spatial_problem import *
from parameters import *
from shelf_interface import ShelfInterface

def flow(q, t, param):
	p = param[0]
	ce = param[1]
	cc = param[2]
	le = param[3]
	nm = param[4]
	omega = param[5]
	M = param[6]
	A = param[7]
	B = param[8]

	m = ce*le + 0.5
	c = ce**2 * (1 - cc**2)

	qdot = [0 for n in range(2*nm+1)]

	qdot[0] = p / m + sum( q[nm+1:2*nm+1] ) / m - 0.5 * c * q[0]**2 /m

	qdot[1:nm+1] = q[nm+1:2*nm+1]
		
	qdot[nm+1:2*nm+1] = ( - np.dot(M, omega**2 * q[1:nm+1]) 
		- (A + B*p*c/m**2) * q[0] 
		- B*c/m**2 * q[0] * sum(q[nm+1:2*nm+1])
		+ 0.5 * B * c**2/m**2 * q[0]**3 )

	return qdot

def vary_pressure(p_range, r_front, r_back):
	print(r_front)
	ce = r_pharynx**2 / r_front**2
	cc = r_front**2 / r_trachea**2
	le = 0.25 * r_front / r_back
	inverseM = np.eye(nm) - [b[m]/(ce*le+0.5) for m in range(nm)]
	M = np.linalg.inv(inverseM)
	A = np.dot(M, a)
	B = np.dot(M, b)
	q = [0 for i in p_range]
	inits = np.array([0 for n in range(2*nm+1)])
	for i, p in enumerate(p_range):
		#print(str(i) + "/" + str(len(p_range)))
		params = (p, ce, cc, le, nm, omega, M, A, B)
		q[i] = odeint(flow, inits, t, args = (params,) )
		inits = q[i][-1, :]

	return q

def grouper(n, iterable, fillvalue=None):
    "grouper(3, 'ABCDEFG', 'x') --> ABC DEF Gxx"
    args = [iter(iterable)] * n
    return izip_longest(fillvalue=fillvalue, *args)

def find_r_indices(r_front_range, r_back_range, n_cores):
	r= [(r_front,r_back) for r_front in r_front_range for r_back in r_back_range]
	r_indices = [[i,j] for i in range(len(r_front_range)) for j in range(len(r_back_range))]
	r_indices = list(grouper(n_cores, r_indices))
	r_indices[-1] = [i for i in r_indices[-1] if i is not None]

	return r_indices

db_path = '/media/matthew/1f84915e-1250-4890-9b74-4bfd746e2e5a/pressure_up_sweep.db'
si = ShelfInterface(db_path)
si.insert('params', params)
si.insert('p_range', p_range)
si.insert('r_front_range', r_front_range)
si.insert('r_back_range', r_front_range)
n_cores = 8
r_indices = find_r_indices(r_front_range, r_back_range, n_cores)

for idx in r_indices:
	q = Parallel(n_jobs=n_cores)(delayed(vary_pressure)
		(p_range, r_front_range[i[0]], r_back_range[i[1]]) for i in idx)
	map(si.insert, idx, q)

si.close()


	
