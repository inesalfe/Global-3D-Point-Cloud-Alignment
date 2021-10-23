from typing import Tuple
from solution import registration_iasd, point_cloud_data_iasd
from search_solution import compute_alignment, align_3d_search_problem

from visualization_vtk import point_clouds_visualization

from sys import argv
from math import sqrt, atan2
from numpy import array, loadtxt, eye, pi, sin, cos, mean, argmin
from time import time

from random import uniform, seed
from numpy.linalg import norm

def run(name_test: str) -> None:

    # loads the point clouds
    dataset_path = name_test
    point_cloud_1 = point_cloud_data_iasd(fileName=dataset_path+'_1.ply')
    point_cloud_2 = point_cloud_data_iasd(fileName=dataset_path+'_2.ply')

    # transform the pointclouds from dictionary style to numpy
    # (for visualization and registration purposes)
    np_point_cloud_1 = point_cloud_1.get_point_cloud(numpy_array=True)
    np_point_cloud_2 = point_cloud_2.get_point_cloud(numpy_array=True)

    # np_point_cloud_1 = np_point_cloud_1 - mean(np_point_cloud_1)
    # np_point_cloud_2 = np_point_cloud_2 - mean(np_point_cloud_2)

    ######################## FOR TESTING ########################
    ##### ROTATES THE FIRST POINT CLOUD BY A RANDOM AMMOUNT #####
    # seed(12)
    # No random - 0.15s, d4 - [-90.0, -67.5, -90.0]
    # 1 - 3s, d5 ; 2.9s, d7 - [-157.5, -22.5, 135.0]
    # 2 - 136s, d8 ; 0.75s, d6 - [-135.0, -67.5, -135.0]
    # 3 - 5s, d5 ; 1.0s, d6 - [-67.5, 0.0, -112.5]
    # 4 - 63s, d8 ; 0.35s, d5 - [-67.5, 0.0, 45.0]
    # 5 - 62s, d7 ; 0.18s, d4 - [0.0, 0.0, -168.75]
    # 6 - 2s, d5 ; 1.8s, d6 - [112.5, 0.0, -157.5]
    # 7 - 98s, d8 ; 3.1s, d7 - [-157.5, 28.125, 0.0]
    # 8 - 17s, d7 ; 1.1s, d6 - [-22.5, 0.0, 112.5]
    # 9 - 4s, d5 ; 0.12s, d4 - [-112.5, -45.0, 0.0]
    # 10 - 31s, d7 ; 1.3s, d6 - [157.5, -78.75, 0.0]
    # 11 - 34s, d7 ; 0.3s, d5 - [-90.0, -45.0, -157.5]
    # 12 - 113s, d8 ; 4.0s, d7 - [-45.0, -22.5, -157.5]

    # a = uniform(-pi, pi)
    # b = uniform(-pi, pi)
    # g = uniform(-pi/2, pi/2)
    # # print('Initial Applied Rotation: [%.1f, %.1f, %.1f]' % (a * 180/pi, b * 180/pi, g * 180/pi))
    # c_a = cos(a)
    # s_a = sin(a)
    # c_b = cos(b)
    # s_b = sin(b)
    # c_g = cos(g)
    # s_g = sin(g)
    # rot_mat = array([
    #         [c_a*c_b, c_a*s_b*s_g-s_a*c_g, c_a*s_b*c_g+s_a*s_g],
    #         [s_a*c_b, s_a*s_b*s_g+c_a*c_g, s_a*s_b*c_g-c_a*s_g],
    #         [-s_b, c_b*s_g, c_b*c_g] ])
    # np_point_cloud_1 = (rot_mat @ np_point_cloud_1.T).T
    #############################################################

    # gets the center of the points, 2nd point cloud
    # for visualization purposes
    avg_x, avg_y, avg_z = point_cloud_2.gets_point_cloud_center()

    # creates a renderer and set the default settings6
    figure = point_clouds_visualization()

    # add the camera position and orientation
    # to the environment
    figure.set_camera_settings(
        camera_position= (avg_x, avg_y+1, avg_z+.35),
        camera_focal_point= (-avg_x, -avg_y, -avg_z)
        )

    # defines the name of the window and size if need (not included)
    figure.set_windows_name_size(name='IASD21/22: Point Cloud Registration')

    # adds the pointclouds to the environment
    # keeps the id of the first, which should rotate and translate to the pc2
    # updates on the position and orientation of the vtk object is done in the
    # registration
    _, _ = figure.make_point_cloud(np_point_cloud_1, point_weight=3, point_cloud_color=(0.1, 0.9, 0.1))
    id_pc1, _ = figure.make_point_cloud(np_point_cloud_1, point_weight=3, point_cloud_color=(0.9, 0.1, 0.1))
    _, _ = figure.make_point_cloud(np_point_cloud_2, point_weight=3, point_cloud_color=(0.1, 0.1, 0.9))

    # creates the registration object
    tic = time()
    valid, r, t, d = compute_alignment(np_point_cloud_1, np_point_cloud_2)
    elapsed = time()-tic

    if valid:
        print('INFO: solver found a solution.')
        print('INFO: ouput information:')
        print('  -> depth:', d)
        print('  -> translation: ', t)
        print('  -> rotation: ', r[0,:])
        print('               ', r[1,:])
        print('               ', r[2,:])
        print('  -> time: %.4f' % elapsed)
    else:
        print('INFO: no solution was found')
        print('  -> time: %.4f' % elapsed)

    # print('  -> time: %.4f' % elapsed)
    print("Registration Error: %.4g" % mean([min(norm(a - np_point_cloud_2, axis=1)) for a in (r @ np_point_cloud_1.T).T + t]))

    # transform the point cloud given the transformation
    # obtained from the proposes search strategy.
    T = eye(4)
    T[0:3,0:3], T[0:3, 3] = r, t
    figure.transform_pointcloud(id_pc1,T)

    # renders the final result.
    print(' ')
    print('INFO: Green and blue points represent the original point-clouds.')
    print('INFO: Red pints represent the application of the computed (r,t) to the green point-cloud.')
    print('INFO: As in the previous assignment, the red points must be aligned with the blue ones.')
    print('INFO: Close the window!')
    figure.render(block=True)

    return
    
    
def main(arguments):

    # Problems are a dictionary indicating
    # with values being a string with the problem.
    # The load_solution function will load th1e data and
    # gt from the respective files.
    PROBLEMS = {
        'PUB0': ('test_nr0'),
        'PUB1': ('test_nr1'),
        'PUB2': ('test_nr2'),
        'PUB3': ('test_nr3'),
        'PUB4': ('test_nr4'),
        'PUB5': ('test_nr5'),
        'PUB6': ('test_nr6'),
        'PUB7': ('test_nr7'),
        'PUB8': ('test_nr8')
    }

    if len(arguments) < 1:
        test = 'PUB0'
    else:
        test = arguments[0]

    run(PROBLEMS[test])

if __name__=="__main__":
    main(argv[1:])
