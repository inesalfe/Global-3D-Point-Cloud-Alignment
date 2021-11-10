# generate data

import numpy as np
from copy import deepcopy

from typing import Union, Tuple, List


class point_cloud_data:

    def __init__(
            self,
            fileName: str,
            ) -> None:
        """creates a pointcloud object from a .ply file

        :param fileName: sets the path to the ply file
        :type fileName: str
        :param apply_rand_transform: if we want to set a baselink, defaults to False
        :type apply_rand_transform: bool, optional
        """

        # the pointcloud data will be a dictionary
        # with keys identifying the 
        self.data = {}
        self.data_numpy = []

        # load the point cloud from the ply file
        self.succ = self.load_point_cloud(fileName)

        return


    def convert_data_array(self) -> None:
        """gets the data in a numpy format shape = (..., 3)
        """
        self.data_numpy = np.stack(tuple(self.data.values()), axis=1).T

        return

    def get_point_cloud(
            self,
            numpy_array: bool=False
            ) -> Union[dict, np.array]:
        """outputs the data in either dictionary type of numpy

        :param numpy_array: set the output to be numpy, defaults to False
        :type numpy_array: bool, optional
        :return: dictionary with the data or numpy if setted
        :rtype: Union[dict, np.array]
        """
        if numpy_array:
            if self.data_numpy == []:
                self.convert_data_array()
            return self.data_numpy
        else:
            return self.data


    def gets_point_cloud_center(self) -> Tuple[float, float, float]:
        """Outputs the center coordinates of the point cloud

        :return: x, y, and z coordinates of the point clouds' center point
        :rtype: Tuple[float, float, float]
        """
        if self.data_numpy == []:
            self.convert_data_array()

        return tuple(np.average(self.data_numpy, axis=0))


    def load_point_cloud(
            self,
            file: str
            ) -> bool:
        """Loads a point cloud from a ply file

        :param file: path to the ply file
        :type file: str
        """

        pass

    def plot_list_points(self) -> None:

        maxPlotPoints = 5
        numPointPlot = min([maxPlotPoints,len(self.dataScanOne.keys())])
        print('Data in Scan One:')
        for key in list(self.dataScanOne.keys())[0:numPointPlot]:
            print(self.dataScanOne[key])
        if len(self.dataScanOne.keys()) > maxPlotPoints:
            print('...') 
        print('Data in Scan Two:')
        for key in list(self.dataScanTwo.keys())[0:numPointPlot]: 
            print(self.dataScanTwo[key])
        if len(self.dataScanOne.keys()) > maxPlotPoints:
            print('...') 

        return
