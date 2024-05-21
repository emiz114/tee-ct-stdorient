# adapted from pca.py
# created 05_20_2024
# updated 05_21_2024

import sys
import vtk
import numpy as np
from vtkmodules.util.numpy_support import vtk_to_numpy

def read_polydata(file_path):
    """ 
    Loads VTK PolyData from the file specified by file_path
    - vtkPolyData: data structure used for geometric data (i.e. points, vertices, lines etc.)
    Applies decimation filter if flag_decimate is True
    """
    # creates a reader for vtk polydata
    reader = vtk.vtkPolyDataReader()
    reader.SetFileName(file_path)
    reader.Update()

    return reader.GetOutput()

def create_vector(start_list, directions_list, scale=1.0):
    """"""
    # create a vtkPoints object and store points
    points = vtk.vtkPoints()
    lines = vtk.vtkCellArray()

    # iterate through arrays
    for i in range(len(start_list)):
        start = points.InsertNextPoint(start_list[i])
        end = points.InsertNextPoint(start_list[i] + directions_list[i] * scale)
        
        # create a polyline to connect points
        line = vtk.vtkLine()
        line.GetPointIds().SetId(0, start)
        line.GetPointIds().SetId(1, end)
        lines.InsertNextCell(line)

    # create polydata to store everything
    polydata = vtk.vtkPolyData()
    polydata.SetPoints(points)
    polydata.SetLines(lines)

    return polydata

def apply_vector(pc, c):
    """
    """
    pc_array = [pc]
    c_array = [c]

    vector = create_vector(c_array, pc_array, 25)

    return vector

# if __name__ == "__main__":

#     if len(sys.argv) != 5: 
#         print("")
#         print("*******************************************************************************************************************")
#         print("   USAGE: python3 | ./PATH/SCRIPT.py | /INPUT_PATH | /SOURCE.vtk | /TARGET.vtk | /OUTPUT.vtk   ")
#         print("*******************************************************************************************************************")
#         print("")
#     else: 
#         # extract filenames from command line arguments
#         input_path, source_file, target_file, output_file = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]

#     # read input meshes
#     source = read_polydata(input_path + source_file)
#     target = read_polydata(input_path + target_file)

#     pc_source, c_source = compute_principal_component(source)
#     pc_target, c_target = compute_principal_component(target)

#     pc_array = [pc_source, pc_target]
#     c_array = [c_source, c_target]

#     vectors = create_vectors(c_array, pc_array)

#     write_polydate(input_path + output_file, vectors)

