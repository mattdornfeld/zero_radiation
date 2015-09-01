import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from parameters import *
from spatial_problem import *
from shelf_interface import ShelfInterface

def calc_params(p, r_front, r_back):
	ce = r_pharynx**2 / r_front**2
	cc = r_front**2 / r_trachea**2
	le = 0.25 * r_front / r_back
	inverseM = np.eye(nm) - [b[m]/(ce*le+0.5) for m in range(nm)]
	M = np.linalg.inv(inverseM)
	A = np.dot(M, a)
	B = np.dot(M, b)
	params = (p, ce, cc, le, nm, omega, M, A, B)

	return params

class ParameterSlider:
	def __init__(self, p_range, r_front_range, r_back_range, fig_fourier, ax_fourier):
		self.p_range = p_range
		self.r_front_range = r_front_range
		self.r_back_range = r_back_range
		self.fig_fourier = fig_fourier
		self.ax_fourier = ax_fourier

		self.p_index = 0
		self.r_front_index = 0
		self.r_back_index = 0

		self.si = ShelfInterface(db_path)
		key = [self.r_front_index, self.r_back_index]
		self.q = self.si.get(key)
		q0 = self.q[self.p_index][:, 0]

		self.plot_fourier(q0)

	def update_values(self):
		self.p = self.p_range[self.p_index]
		self.r_front = r_front_range[self.r_front_index]
		self.r_back = r_back_range[self.r_back_index]
		self.params = calc_params(p, r_front, r_back)

		#self.qdot = np.array([flow(_, 0, params) for _ in self.q[self.p_index]])
	
	def plot_fourier(self, q0):
		T1 = 7000 
		T2 = 9000
		q0 = q0[T1:T2]
		f = c / L * np.fft.rfftfreq( len(q0), d=1/fs )
		Q0 = np.fft.rfft( q0 - np.mean(q0) )
		self.ax_fourier[0].cla()
		self.ax_fourier[0].plot(f, abs(Q0), color='darkorange')
		self.ax_fourier[0].set_xlim(0, max(f))
		self.ax_fourier[0].set_ylim(0, 0.002)
		self.ax_fourier[0].set_xlabel('Frequency (Hz)')
		self.ax_fourier[0].set_ylabel('Abs(Fourier(u_0(t))', rotation=0, labelpad=40)
		self.fig_fourier.canvas.draw()

	def plot_phase_diagram(self, q0, q0dot):
		plt.plot(q0)

	def on_pressure_slide(self, p_index):
		self.p_index = int(p_index)
		self.update_values()

		pressure_slider.valtext.set_text(str(rho*c**2*self.p)+"(pa)")
		q0 = c * self.q[self.p_index][:, 0]

		self.plot_fourier(q0)		

	def on_r_front_slide(self, r_front_index):
		self.r_front_index = int(r_front_index)
		key = [self.r_front_index, self.r_back_index]
		self.q = self.si.get(key)
		self.update_values()

		r_front_slider.valtext.set_text(str(1000*self.r_front) + "(mm)")
		q0 = c * self.q[self.p_index][:, 0]
		self.plot_fourier(q0)

	def on_r_back_slide(self, r_back_index):
		self.r_back_index = int(r_back_index)
		key = [self.r_front_index, self.r_back_index]
		self.q = self.si.get(key)
		self.update_values()

		r_back_slider.valtext.set_text(str(1000*self.r_back) + "(mm)")
		q0 = c * self.q[self.p_index][:, 0]
		self.plot_fourier(q0)		

plt.style.use('ggplot')
fig_fourier, ax_fourier = plt.subplots(nrows=4)
#lbh
ax_fourier[0].set_position([0.1, 0.25, 0.8, 0.7])
ax_fourier[1].set_position([0.1, 0.05, 0.65, 0.03])
ax_fourier[2].set_position([0.1, 0.08, 0.65, 0.03])
ax_fourier[3].set_position([0.1, 0.11, 0.65, 0.03])

pressure_slider = Slider(ax_fourier[1], 'Pressure', 0, len(p_range)-1, valinit=0, valfmt='%1.0f')
r_front_slider = Slider(ax_fourier[2], 'r_front', 0, len(r_front_range)-1, valinit=0, valfmt='%1.0f')
r_back_slider = Slider(ax_fourier[3], 'r_back', 0, len(r_back_range)-1, valinit=0, valfmt='%1.0f')

parameter_slider = ParameterSlider(p_range, r_front_range, r_back_range, fig_fourier, ax_fourier)
pressure_slider.on_changed(parameter_slider.on_pressure_slide)
r_front_slider.on_changed(parameter_slider.on_r_front_slide)
r_back_slider.on_changed(parameter_slider.on_r_back_slide)
plt.show()

p = p_range[0]
r_front = r_front_range[0]
r_back = r_back_range[0]
