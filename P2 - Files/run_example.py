from solution import registration_iasd, point_cloud_data_iasd
from search_solution import compute_alignment
# from visualization_vtk import point_clouds_visualization

from sys import argv

def run(name_test: str, number_iterations: int) -> None:

    # loads the point clouds
    dataset_path = name_test
    point_cloud_1 = point_cloud_data_iasd(fileName=dataset_path+'_1.ply')
    point_cloud_2 = point_cloud_data_iasd(fileName=dataset_path+'_2.ply')

    # transform the pointclouds from dictionary style to numpy
    # (for visualization and registration purposes)
    np_point_cloud_1 = point_cloud_1.get_point_cloud(numpy_array=True)
    np_point_cloud_2 = point_cloud_2.get_point_cloud(numpy_array=True)

    # creates the registration object
    # reg = registration_iasd(np_point_cloud_1, np_point_cloud_2)
    # computes the registration
    # r, t = reg.compute(
    #     show_visualization=False,
    #     verbose=True,
    #     vtk_visualization=figure,
    #     vtk_pc1=id_pc1,
    #     max_iter=number_iterations
    #     )

    _, r, t, d = compute_alignment(np_point_cloud_1, np_point_cloud_1)

    print(r)
    print(t)
    print(d)

    return
    
def main(arguments):

    # Problems are a dictionary indicating
    # with values being a string with the problem.
    # The load_solution function will load the data and
    # gt from the respective files.
    PROBLEMS ={
        'PUB1': ('test_nr1', 5),
        'PUB2': ('test_nr2', 5),
        'PUB3': ('test_nr3', 15),
        'PUB4': ('test_nr4', 10),
        'PUB5': ('test_nr5', 10),
        'PUB6': ('test_nr6', 25),
        'PUB7': ('test_nr7', 20),
        'PUB8': ('test_nr8', 20)
    }

    if len(arguments) < 1:
        test = 'PUB7'
    else:
        test = arguments[0]

    run(PROBLEMS[test][0], PROBLEMS[test][1])

if __name__=="__main__":
    main(argv[1:])
