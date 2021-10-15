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

		pass

	def path_cost(self, c, state1: State, action: Action, state2: State) -> float:

		pass

	def compute_alignment(scan1: array((...,3)), scan2: array((...,3)),) -> Tuple[bool, array, array, int]:

		pass