from __future__ import division
import matplotlib
from math import *
import numpy as np
from numpy import linalg as LA 
import matplotlib.pyplot as plt
import argparse
from model import model
from scipy.optimize import fsolve

V_R = 10.0

def controller(state, state_d):
	x, y, theta = state
	x_d, y_d = state_d
	#######################
	# Based on state and state_d
	# you need to calculate steering
	# angle such it goes towards state_d
	# You can use a controller very similar 
	# described in the MP write-up
	#########################


	return None#(K*alpha)

	#return None

def simulation(state, reach_state, obstacle_map = None, show_plot = False, time_steps = 1000):
	safe_check = True
	data = []
	x, y, theta = state
	x_d, y_d= reach_state
	
	dist = np,sqrt((x - x_d)**2 + (y - y_d)**2)
	i = 0
	eps = 0.01
	while (dist>eps): #add relevant values in abs()
		i += 1

		data.append(state)

		if obstacle_map is not None:		
			if ((int(x) >= 1000) | (int(x) <= 0)  | ((int(y) >= 1000) | (int(y) <= 0))):
				safe_check = False
				break
			if (obstacle_map[int(x), int(y)] == (0, 0, 0)):#(0, 0, 0)
				safe_check = False 
				break
		if show_plot:
			print ('state:', state, 'iterations:', i)
			plt.plot(x, y, 'ro')

		x, y, theta = state
		control = controller(state, reach_state)
		state = model(state, control, V_R = 0.0)

		if i > time_steps:
			safe_check = False
			break
	################################
	num = len(data)

	return num, state, safe_check, data

def test(state, reach_state):
	plt.plot([state[0], reach_state[0]], [state[1], reach_state[1]])
	simulation(state, reach_state, show_plot = True)
	plt.show()


state = np.array([0, 0, 0.0])
test(state, np.array([-10,100]))




