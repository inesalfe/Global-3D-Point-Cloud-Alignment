from registration import registration
from get_pointcloud import point_cloud_data

import numpy as np
from numpy.linalg import norm, svd, det
from typing import Tuple


class registration_iasd(registration):

	def __init__(self, scan_1: np.array((..., 3)), scan_2: np.array((..., 3))) -> None:

		# inherit all the methods and properties from registration
		super().__init__(scan_1, scan_2)

		return


	def compute_pose(self,correspondences: dict) -> Tuple[np.array, np.array]:
		"""compute the transformation that aligns two
		scans given a set of correspondences

		:param correspondences: set of correspondences
		:type correspondences: dict
		:return: rotation and translation that align the correspondences
		:rtype: Tuple[np.array, np.array]
		"""

		S1 = [correspondences[i]['point_in_pc_1'] for i in range(len(correspondences))]
		S2 = [correspondences[i]['point_in_pc_2'] for i in range(len(correspondences))]

		# Compute the centers
		p_center, q_center = np.mean(S1, axis=0), np.mean(S2, axis=0)

		P = np.array(S1 - p_center)
		Q = np.array(S2 - q_center)

		A = Q.T @ P 

		U, _, VT = svd(A) # Singular Value Decomposition
		Rout = U @ np.diag([1,1,det(U @ VT)]) @ VT
		tout = q_center - Rout @ p_center

		return (Rout, tout)

	def find_closest_points(self) -> dict:
		"""Compute the closest points in the two scans.
		There are many strategies. We are taking all the points in the first scan
		and search for the closest in the second. This means that we can have > than 1 points in scan
		1 corresponding to the same point in scan 2. All points in scan 1 will have correspondence.
		Points in scan 2 do not have necessarily a correspondence.

		:param search_alg: choose the searching option
		:type search_alg: str, optional
		:return: a dictionary with the correspondences. Keys are numbers identifying the id of the correspondence.
				Values are a dictionaries with 'point_in_pc_1', 'point_in_pc_2' and 'dist2',
				identifying the pair of points in the correspondence and their distance.
		:rtype: dict
		"""
		# compute the correspondence for every point in scan_1
		matches = [self.scan_2[np.argmin(norm(a - self.scan_2, axis=1))] for a in self.scan_1] 

		return {i : {'point_in_pc_1' : self.scan_1[i], 'dist2' : norm(self.scan_1[i]-matches[i]), 'point_in_pc_2' : matches[i]} for i in range(len(matches))}		

class point_cloud_data_iasd(point_cloud_data):

	def __init__(
			self,
			fileName: str,
			) -> None:
			
		super().__init__(fileName)

		return


	def load_point_cloud(
			self,
			file: str
			) -> bool:
		"""Load a point cloud from a ply file

		:param file: source file
		:type file: str
		:return: returns true or false, depending on whether the data in
		the .ply file was corrupted or not
		:rtype: bool
		"""

		coord = {'x': 0, 'y': 1, 'z': 2}
		try:
			with open(file) as fd:
				lines = fd.readlines()
				st, n_pts, xyz = 0, 0, []
				for line in lines:
					st += 1
					l = line.strip().split()
					if len(l) == 1:
						if l[0] == 'end_header':
							break
					elif (l[0], l[1]) == ('element', 'vertex'): # Check number of vertices
						n_pts = int(l[-1])
					elif (l[0],l[1]) == ('property', 'float'): # Check available coordinates
						if l[2] in ('x', 'y', 'z'):
							xyz.append(l[2])
				if(len(xyz) != 3 or len(lines[st:]) < n_pts): # Not enough coordinates or wrong number of vertices
					raise ValueError("Wrong information")
				# create the dictionary with all the points, with the coordinates in the order
				#provided in the header
				self.data = {i: np.array([l.split()[coord[c]] for c in xyz], dtype=float) for i, l in enumerate(lines[st:st+n_pts]) }
		except Exception as e: # If an exception is raised, then the file is not well-formatted
			print("Error reading file: \n", e)
			return False

		return True
