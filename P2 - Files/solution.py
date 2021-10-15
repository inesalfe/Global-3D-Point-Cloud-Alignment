from typing import Tuple
from numpy import array
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
State = Tuple[float, float, float, int]

class align_3d_search_problem(search.Problem):

	def __init__(self, scan1: array((...,3)), scan2: array((...,3))) -> None:
		"""Function that instantiate your class.
		You CAN change the content of this __init__ if you want.
		:param scan1: input point cloud from scan 1 :type scan1: np.array
		:param scan2: input point cloud from scan 2 :type scan2: np.array
		"""

		# Creates an initial state.
		# You may want to change this to something representing # your initial state.
		self.initial = Tuple(0, 0, 0, 1)
		self.range = Tuple(np.pi, np.pi, np.pi/2)
		return

	def actions(self, state: State) -> Tuple[Action, ...]:
		"""Returns the actions that can be executed in the given state.
		The result would be a list, since there are only four possible actions in any given state of the environment
			:param state: Abstract representation of your state
			:type state: State
			:return: Tuple with all possible actions
			:rtype: Tuple
		"""

		return ([(i, self.range[i]/2**State[3]), (i, -self.range[i]/2**State[3]) for i in range(3)])

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
		result[action[0]] += action[1]
		result[3] += 1

		return tuple(result)

	def goal_test(self, state: State) -> bool:

		c_a = np.cos(state[0])
		s_a = np.sin(state[0])
		c_b = np.cos(state[1])
		s_b = np.sin(state[1])
		c_g = np.cos(state[2])
		s_g = np.sin(state[2])

		R = np.array([c_a*c_b, c_a*s_b*s_g-s_a*c_g, c_a*s_b*c_g+s_a*s_g],
					 [s_a*c_b, s_a*s_b*s_g-c_a*c_g, s_a*s_b*c_g-c_a*s_g],
					 [-s_b, c_b*s_g, c_b*c_g])

		pt_cloud = R @ self.scan_1

		sum([np.min(norm(a - self.scan_2, axis=1))] for a in pt_cloud)
		
		pass

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

		pass