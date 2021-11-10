from typing import Tuple
from numpy import array, pi
from numpy.linalg import norm
import numpy as np
import search

# you can use the class registration_iasd
# from your solution.py (previous assignment)
from solution import registration_iasd

# Choose what you think it is the best data structure
# for representing actions.
Action = Tuple[int, int]
# An action is represented by 2 integers: 
# the first one defines in which of the 3 angles to rotate 
# and the second the orientation of the rotation (+/- 1)

class State_class:
	def __init__(self, angles: Tuple[float, float, float], ranges: Tuple[float, float, float], err: float, r: np.array((3,3)), h: float, depth: int):
		self.angles = angles # set of 3 angles, \alpha, \beta, \gamma
		self.ranges = ranges # current range of each angle (the current search space is angle[i]+/-range[i])
		self.err = err # current registration error - average distance between some points in cloud 1 and the closest point in cloud 2
		self.R = r # current rotation matrix
		self.h = h # heuristic value
		self.depth = depth # current depth in the search graph

	def __eq__(self, other):
		# State quality only based on the angles, not on the ranges
		return self.angles == other.angles
	
	def __hash__(self) -> int:
		# State hashing only based on the angles
		return hash(self.angles)
	
	def __lt__(self, other):
		pass


# Choose what you think it is the best data structure
# for representing states.
State = State_class

class align_3d_search_problem(search.Problem):

	def __init__(self, scan1: array((...,3)), scan2: array((...,3))) -> None:
		"""Function that instantiate your class.
		You CAN change the content of this __init__ if you want.
		:param scan1: input point cloud from scan 1 :type scan1: np.array
		:param scan2: input point cloud from scan 2 :type scan2: np.array
		"""
		self.scan_1 = scan1
		self.scan_2 = scan2
		# Fraction of the points in cloud1 that will be used when computing the correspondences and error (at result)
		# fB is the number of points that is used in the get_compute function, in the refinement process (at goal_test)
		self.fS = max(15, int((scan1.shape)[0]*0.05))
		self.fB = 25
		# Creates an initial state - no rotation and the ranges of the whole space
		self.initial = State((0, 0, 0),(pi, pi/2, pi), np.mean([np.min(norm(a - scan2, axis=1)) for a in scan1]), np.eye(3), 1, 0)
		
		# Tolerance values: 
		# tolS is the threshold below which we define a state to be "promissing";
		# tolB is the threshold below which a "promissing" state is deemed to be the goal state, after 
		# applying the refinement with the get_compute function (more on this in goal_test).

		# If tolS is too high there will be more 'promissing' states that won't necessarily be a solution,
		# so the get_compute function will be called more times, making the search slower; If this value is too
		# low the criteria for a state to be deemed 'promissing' may be too strict and a solution may take more to be found.
		# The tolerence values were computed as a fraction of the average magnitude of the 3D points in point cloud 1 (avg_dist).
		# This structure is more general than a simple if value. The ratios were tuned so that they would work for all tests
		
		avg_dist = np.average([norm(a) for a in scan1])
		self.tolS = avg_dist / 6
		self.tolB = self.tolS / 3.6
		return



	def actions(self, state: State) -> Tuple[Action, ...]:
		"""Returns the actions that can be executed in the given state.
		The result would be a list, since there are only four possible actions in any given state of the environment
			:param state: Abstract representation of your state
			:type state: State
			:return: Tuple with all possible actions
			:rtype: Tuple
		"""
		# Branching factor of 6: for each one of the 3 angles move in both directions
		# The condition ' ... > 10*pi/180 ' allows for a discretization of the search space of about 5 degrees; this isn't strictly necessary
		# given the heuristic used, but it allows for a more general code (for example, if we were using dfs this would be need so that the
		# search space wouldn't be infinite)
		t = tuple((i, -1) for i in range(3) if state.ranges[i] > 10*pi/180) + tuple((i, 1) for i in range(3) if state.ranges[i] > 10*pi/180)
		return t

	def result(self, state: State, action: Action) -> State:
		"""Returns the state that results from executing the given action in the given state. The action must be one of
		self.actions(state).
			:param state: Abstract representation of your state
			:type state: State
			:param action: An action
			:type action: Action
			:return: A new state
			:rtype: State
		"""
		new_angles = list(state.angles)
		new_ranges = list(state.ranges)
		# each action only affects one angle at a time
		new_ranges[action[0]] /= 2 # range for the current angle is reduced to a half
		# New state angle is equal to the previous one incremented/decremented by the current range value for that angles
		new_angles[action[0]] += new_ranges[action[0]] * action[1]
		# compute the rotation matrix for the current state
		c = np.cos(new_angles)
		s = np.sin(new_angles)
		r = array([[c[0]*c[1], c[0]*s[1]*s[2] - s[0]*c[2], c[0]*s[1]*c[2] + s[0]*s[2]],
				   [s[0]*c[1], s[0]*s[1]*s[2] + c[0]*c[2], s[0]*s[1]*c[2] - c[0]*s[2]],
				   [-s[1], c[1]*s[2], c[1]*c[2]]
				  ])
		# The error is the mean distance from each point in the first point cloud to the closest point
		# in the second point cloud. In order to make this process more efficient, we only compute the 
		# mean for a fraction of the points in the first point cloud (5%)
		err = np.mean([np.min(norm(a - self.scan_2, axis=1)) for a in self.scan_1[:self.fS] @ r.T])
		depth = state.depth + 1
		# the heuristic value is computed here: it is the product between the current error and the search depth;
		# The idea is to prioritize nodes at smaller depths, for similar errors. Intuitively, since the BFS search
		# worked fine and usually the solutions were at small depths, we decided to add this factor in our heuristic 
		h = err * depth 
		return State_class(tuple(new_angles), tuple(new_ranges), err, r, h, depth)

	def goal_test(self, state: State) -> bool:
		"""Checks whether a state is the goal state. Returns true if so and false otherwise
			:param state: Current state
			:type state: State
			:return: true if the state is a goal state, false otherwise
			:rtype: bool
		"""
		# The error for the current state was already computed when the state was created (in result function)
		if (state.err > self.tolS):
			return False
		reg = registration_iasd(self.scan_1[:self.fB] @ state.R.T, self.scan_2)
		r, t = reg.get_compute()
		err = np.mean([np.min(norm(a - self.scan_2, axis=1)) for a in self.scan_1[:self.fB] @ state.R.T @ r.T + t])
		return (err <= self.tolB)


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
		:return: 0 - path cost not used
		:rtype: float
		"""
		return 0

	def h(self, node) -> float:
		"""Returns the heuristic value associated with a given state
		:param node: graph node that has the current state
		:type node: Node
		:return: heuristic value of the current state; computed in the result method
		:rtype: float
		"""
		return node.state.h
	
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
		# Center each point cloud
		avg1 = np.average(scan1, axis=0)
		avg2 = np.average(scan2, axis=0)
		scan1 = scan1 - avg1
		scan2 = scan2 - avg2

		problem = align_3d_search_problem(scan1,scan2)
		# Greedy Best First Graph Search, since we are not considering path costs, so we only care about the heuristic value
		sol_node = search.greedy_best_first_graph_search(problem, problem.h)
		if(sol_node != None):
			R = sol_node.state.R
			# Apply the get_compute method to refine the solution from the search process
			reg = registration_iasd(scan1 @ R.T, scan2)
			r, t = reg.get_compute()
			# return the final rotation (product of both rotations) and the final translation 
			# (where we have to take into consideration) that both point clouds where centered at the beginning:
			# (R @ (scan1-avg1).T).T + t = (scan2-avg2) <=> (R @ scan1.T).T + (t + avg2 - R @ avg1) = scan2
			# where R is the final rotation and t' = (t + avg2 - R @ avg1) the final translation )
			return (True, r @ R, t - r @ R @ avg1 + avg2, sol_node.depth)
		return (False, np.zeros([3,3]), np.zeros(3), 0)
