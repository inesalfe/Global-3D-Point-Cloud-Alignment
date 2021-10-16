from typing import Tuple
from numpy import array
from numpy.linalg import norm
import numpy as np
import search

# you can use the class registration_iasd
# from your solution.py (previous assignment)
from solution import registration_iasd

# Choose what you think it is the best data structure
# for representing actions.
Action = Tuple[int, float]

# Choose what you think it is the best data structure
# for representing states.
State = Tuple[float, float, float, float, float, float]
# State = Tuple[float, float, float, int]

class align_3d_search_problem(search.Problem):

	def __init__(self, scan1: array((...,3)), scan2: array((...,3))) -> None:
		"""Function that instantiate your class.
		You CAN change the content of this __init__ if you want.
		:param scan1: input point cloud from scan 1 :type scan1: np.array
		:param scan2: input point cloud from scan 2 :type scan2: np.array
		"""

		# Creates an initial state.
		# You may want to change this to something representing # your initial state.
		self.initial = (-np.pi, np.pi, -np.pi, np.pi, -np.pi/2, np.pi/2)
		# self.initial = (0,0,0,1)
		# self.range = (np.pi, np.pi, np.pi/2)
		self.scan_1 = scan1
		self.scan_2 = scan2
		# MUDAR A TOLERÂNCIA AQUI
		self.tol = 1e-2
		return

	def actions(self, state: State) -> Tuple[Action, ...]:
		"""Returns the actions that can be executed in the given state.
		The result would be a list, since there are only four possible actions in any given state of the environment
			:param state: Abstract representation of your state
			:type state: State
			:return: Tuple with all possible actions
			:rtype: Tuple
		"""
		# p = 2**state[3]
		# return (tuple((i, self.range[i]/p) for i in range(3)) +
		# 		 tuple((i, -self.range[i]/p) for i in range(3)) )
		return (tuple((2*i, (state[2*i] + state[2*i+1])/2) for i in range(3)) +
				 tuple((2*i+1, (state[2*i] + state[2*i+1])/2) for i in range(3)) )

	def result(self, state: State, action: Action) -> State:
		"""Returns the state that results from executing the given action in the given state. The action must be one of
		self.actions(state).
			:param state: Abstract representation of your state
			:type state: [type]
			:param action: An action
			:type action: [type]
			:return: A new state
			:rtype: State
		"""

		result = list(state)
		result[action[0]] = action[1] 
		# result[3] += 1

		return tuple(result)

	def goal_test(self, state: State) -> bool:
		a = (state[0] + state[1])/2
		b = (state[2] + state[3])/2
		g = (state[4] + state[5])/2		
		# a = state[0]
		# b = state[1]
		# g = state[2]
		c_a = np.cos(a)
		s_a = np.sin(a)
		c_b = np.cos(b)
		s_b = np.sin(b)
		c_g = np.cos(g)
		s_g = np.sin(g)

		R = array([[c_a*c_b, c_a*s_b*s_g-s_a*c_g, c_a*s_b*c_g+s_a*s_g],
					 [s_a*c_b, s_a*s_b*s_g+c_a*c_g, s_a*s_b*c_g-c_a*s_g],
					 [-s_b, c_b*s_g, c_b*c_g]])

		pt_cloud = (R @ self.scan_1.T).T

		error = np.mean([np.min(norm(a - self.scan_2, axis=1)) for a in pt_cloud])
		print(tuple([(state[2*i] + state[2*i+1])/2 * 180/np.pi for i in range(3)]), error)
		# print(tuple([state[i] * 180/np.pi for i in range(3)]), error)
		return (error < self.tol)

	def path_cost(self, c, state1: State, action: Action, state2: State) -> float:

		"""Returns the cost of a solution path that arrives at state2 from state1 via action, assuming cost c to get up to state1. If the problem is such that the path doesn't matter, this function will only look at state2. If the path does matter, it will consider c and maybe state1
		and action. The default method costs 1 for every step in the path.
		
		:param c: cost to get to the state1
		:type c: [type]
		:param state1: parent node
		:type state1: State
		:param action: action that changes the state from state1 to state2
		:type action: Action
		:param state2: state2
		:type state2: State
		:return: [description]
		:rtype: float
		"""
		pass
	
def compute_alignment(scan1: array((...,3)), scan2: array((...,3)),) -> Tuple[bool, array, array, int]:
		
		"""Function that returns the solution.
		You can use any UN-INFORMED SEARCH strategy we study in the theoretical classes.
		:param scan1: first scan of size (..., 3) :type scan1: array
		:param scan2: second scan of size (..., 3) :type scan2: array
		:return: outputs a tuple with: 1) true or false depending on
			whether the method is able to get a solution; 2) rotation parameters (numpy array with dimension (3,3)); 3) translation parameters
			(numpy array with dimension (3,)); and 4) the depth of the obtained solution in the proposes search tree.
		:rtype: Tuple[bool, array, array, int]
		"""
		sol_node = search.breadth_first_graph_search(align_3d_search_problem(scan1,scan2))
		if(sol_node != None):
			sol_state = sol_node.state
			a = (sol_state[0] + sol_state[1])/2
			b = (sol_state[2] + sol_state[3])/2
			g = (sol_state[4] + sol_state[5])/2
			# a = sol_state[0]
			# b = sol_state[1]
			# g = sol_state[2]
			c_a = np.cos(a)
			s_a = np.sin(a)
			c_b = np.cos(b)
			s_b = np.sin(b)
			c_g = np.cos(g)
			s_g = np.sin(g)

			R = array([[c_a*c_b, c_a*s_b*s_g-s_a*c_g, c_a*s_b*c_g+s_a*s_g],
				     [s_a*c_b, s_a*s_b*s_g+c_a*c_g, s_a*s_b*c_g-c_a*s_g],
				     [-s_b, c_b*s_g, c_b*c_g]])
			print("Search solution:\n", R)
			reg = registration_iasd((R @ scan1.T).T, scan2)
			# computes the registration
			r, t = reg.get_compute()
			return (True, r @ R, t, sol_node.depth)
		return (False, np.zeros([3,3]), np.zeros(3), 0)
