import numpy as np 
import random 
from PIL import Image, ImageDraw
from controller.controller import SELECT_INPUT
import copy
BLACK = (0, 0, 0)
GOAL = (255, 255, 255)
PATH = (255, 255, 255)

class node():
	def __init__(self, state, parent):
		self.state = state
		self.parent = parent
		self.children = []
		
# square of l2 distance between s1 = (x1, y1) and s2 = (x2, y2)
def dist(s1, s2):
	x1 = s1[0]
	y1 = s1[1]

	x2 = s2[0]
	y2 = s2[1]

	return np.sqrt((x1-x2)*(x1-x2) + (y1-y2)*(y1-y2))

# U: a list of historical states that controller.py visited
# return value: the last state in the list
def NEW_STATE(U):
	'''
	Based on the output by the SELECT_INPUT()
	you have to compute the next state using Xnear and Deltat
	'''
	#######
	## add code here
	#######
	new_state = U[-1]	
	return new_state



class RRT: 
	def __init__(self, X_init, X_goal, winsize, static_obstacles = 'maps/fourway_intersection.jpeg'):
	#def __init__(self, X_init, X_goal, winsize, static_obstacles = 'maps/single_road.jpeg'):

		self.nodes = []
		self.Xinit = node(X_init, None) 
		self.nodes.append(self.Xinit)   
		self.Xnear = node(None, None)   
		self.winsize = winsize
		self.X_goal = X_goal
		self.static_obstacles = Image.open(static_obstacles)
		self.dynamic_obstacles =  Image.open(static_obstacles)
		self.Xrand = None
		self.Xnear = None
		self.path = None

	def findNearest(self, Xrand):
		Nodes = self.nodes[0]
		for p in self.nodes:
			if dist(p.state, Xrand) < dist(Nodes.state, Xrand):
				Nodes = p

		return Nodes

	def getPath(self, Xnew):

		node = Xnew
		path = []
		path.append(node.state)
		path.append(node.children)

		while(node != self.Xinit):
			node = node.parent
			path.append(node.state)
			path.append(node.children)
		path.reverse()
		return path

	# plan safe paths to the car by RRT random tree algorithm
	def plan(self):

		epsilon = .25
		goal_state = False
		############## Draws Goal area in the image tmp.png############
		draw = ImageDraw.Draw(self.static_obstacles)
		draw.line((self.X_goal[0], self.X_goal[1], self.X_goal[0], self.X_goal[3]), fill=GOAL)
		draw.line((self.X_goal[0], self.X_goal[1], self.X_goal[2], self.X_goal[1]), fill=GOAL)
		draw.line((self.X_goal[2], self.X_goal[3], self.X_goal[2], self.X_goal[1]), fill=GOAL)
		draw.line((self.X_goal[2], self.X_goal[3], self.X_goal[0], self.X_goal[3]), fill=GOAL)

		self.static_obstacles.save('tmp.png')
		dynamic_obstacles = Image.open('maps/fourway_intersection.jpeg').load()
		#dynamic_obstacles = Image.open('maps/single_road.jpeg').load()
		#################################################################
		self.Xnear = self.Xinit
		# for 100 iterations
		for k in range(100):
			safe_check = False
			print ('RRT iterations:', k)
			# X_rand <- Random_State()
			while ((safe_check == False)):
				checkXrand = True
				while (checkXrand):
					######### randomly sampling points in RRT #############################
					if(random.random() < epsilon):
						self.Xrand = [abs(random.randint(self.X_goal[0], self.X_goal[2])-1), abs(random.randint(self.X_goal[1], self.X_goal[3])-1)]
					else:
						self.Xrand = [abs(10*random.randint(1, 100)-1), abs(10*random.randint(1, 100)-1)] 
						#self.Xrand = [abs(100*random.randint(1, 10)-1), abs(100*random.randint(1, 10)-1)] 
						#self.Xrand = [abs(1*random.randint(1, 1000)-1), abs(1*random.randint(1, 1000)-1)] 
					
					
					######### Checks whether sampled point is an obstacle or not ############
					if ((dynamic_obstacles[self.Xrand[0], self.Xrand[1]]) == 0):
						checkXrand = True
					else: 
						checkXrand = False

				# X_near <- Nearest_Neighbor(X_rand)
				self.Xnear = self.findNearest(self.Xrand) 
				#Finds nearest node to this sampled state 
				# you need to write the code for this function
				##############################################################################
				X_i = self.Xnear.state
				# U <- Select_Input(X_rand, X_near)
				######### simulation function uses controller function which you need to edit first##########
				U, safe_check = SELECT_INPUT(X_i, self.Xrand, self.dynamic_obstacles.load())

				if safe_check:
					# X_new <- New_State(X_near, U, delta_t)
					Xnew = NEW_STATE(U)
					# add vertex X_new to map
					draw.line((Xnew[0], Xnew[1], X_i[0], X_i[1]), fill=PATH)
					self.static_obstacles.save('tmp.png')
					Xnew = node([Xnew[0], Xnew[1], float(Xnew[2]%(2*np.pi))], self.Xnear)
					self.nodes.append(Xnew)


				#############################################################################################
				## If your simulation() returns safe_check = True
				## your code needs to take one step towards the sampled point
				#  Write that code here ##############
		




import time
t0 = time.time()
RRT_planner = RRT([50.0, 550.0, 0.0], [700,400, 800, 600], (1000,1000))
#RRT_planner = RRT([50.0, 550.0, 0.0], [400,700, 600, 800], (1000,1000))
RRT_planner.plan()
t1 = time.time()
total = t1 - t0
print (total)
