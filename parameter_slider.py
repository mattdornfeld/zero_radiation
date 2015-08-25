import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from parameters import *

class ParameterSlider:
	def __init__(self, q, p_range, r_front_range, r_back_range, fig, ax):
		self.q = q
		self.p_range = p_range
		self.r_front_range = r_front_range
		self.r_back_range = r_back_range
		self.fig = fig
		self.ax = ax

		self.p_index = 0
		self.r_front_index = 0
		self.r_back_index = 0

	def plot_fourier(self, q0):
		T1 = 7000 
		T2 = 9000
		q0 = q0[T1:T2]
		f = c / L * np.fft.rfftfreq( len(q0), d=1/fs )
		Q0 = np.fft.rfft( q0 - np.mean(q0) )
		self.ax[0].cla()
		self.ax[0].plot(f, abs(Q0))
		self.ax[0].set_xlim(0, max(f))
		self.ax[0].set_ylim(0, 0.05)
		self.fig.canvas.draw()

	def on_pressure_slide(self, p_index):
		self.p_index = int(p_index)
		p = self.p_range[self.p_index]
		pressure_slider.valtext.set_text(str(rho*c**2*p)+"(pa)")
		q0 = c * self.q[self.r_front_index][self.r_back_index][self.p_index][:, 0]
		self.plot_fourier(q0)		

	def on_r_front_slide(self, r_front_index):
		self.r_front_index = int(r_front_index)
		r_front = r_front_range[self.r_front_index]
		r_front_slider.valtext.set_text(str(1000*r_front) + "(mm)")
		q0 = c * self.q[self.r_front_index][self.r_back_index][self.p_index][:, 0]
		self.plot_fourier(q0)

	def on_r_back_slide(self, r_back_index):
		self.r_back_index = int(r_back_index)
		r_back = r_back_range[self.r_back_index]
		r_back_slider.valtext.set_text(str(1000*r_back) + "(mm)")
		q0 = c * self.q[self.r_front_index][self.r_back_index][self.p_index][:, 0]
		self.plot_fourier(q0)		

fig, ax = plt.subplots(nrows=4)
#lbrt
ax[0].set_position([0.1, 0.25, 0.8, 0.7])
ax[1].set_position([0.1, 0.1, 0.65, 0.03])
ax[2].set_position([0.1, 0.14, 0.65, 0.03])
ax[3].set_position([0.1, 0.18, 0.65, 0.03])

pressure_slider = Slider(ax[1], 'Pressure', 0, len(p_range)-1, valinit=0, valfmt='%1.0f')
r_front_slider = Slider(ax[2], 'r_front', 0, len(r_front_range)-1, valinit=0, valfmt='%1.0f')
r_back_slider = Slider(ax[3], 'r_back', 0, len(r_back_range)-1, valinit=0, valfmt='%1.0f')

parameter_slider = ParameterSlider(q, p_range, r_front_range, r_back_range, fig, ax)
pressure_slider.on_changed(parameter_slider.on_pressure_slide)
r_front_slider.on_changed(parameter_slider.on_r_front_slide)
r_back_slider.on_changed(parameter_slider.on_r_back_slide)
plt.show()