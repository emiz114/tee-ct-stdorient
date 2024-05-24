# PCA + EIGENVECTORS
# created 05_23_2023

import os
import sys
import vtk
import numpy as np
from vtkmodules.util.numpy_support import vtk_to_numpy
import vtk_vector

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

def compute_pca(mesh, eig_num):
    """
    Computes the Principal Components of a 3D mesh
    Pipeline: centroid -> covariance matrix -> eigenvalues/vectors -> principal components
    """
    points = mesh.GetPoints()
    vertices = vtk_to_numpy(points.GetData())

    # compute centroid (origin)
    centroid = np.mean(vertices, axis = 0)

    # compute covariance matrix
    centered_vertices = vertices - centroid
    covariance_matrix = np.cov(centered_vertices, rowvar=False)

    # compute eigenvectors/eigenvectors (axes)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)

    # sort eigenvectors based on eigenvalues (ascending order)
    sort_indices = np.argsort(eigenvalues)
    print("SORT" + str(sort_indices[eig_num]))
    principal_component = eigenvectors[:, sort_indices[eig_num]]

    return principal_component, centroid

def write_polydata(output_path, polydata): 
    # Save aligned mesh
    writer = vtk.vtkPolyDataWriter()
    writer.SetFileName(output_path)
    writer.SetInputData(polydata)
    writer.Write()

if __name__ == "__main__":

    if (len(sys.argv) != 6): 
        print("")
        print("***************************************************************************************")
        print("   USAGE: python3 | ./PATH/SCRIPT.py | FRAME | ID | /INPUT_PATH | /INPUT.vtk | EIG_NUM ")
        print("***************************************************************************************")
        print("")
    else: 
        # extract filenames from command line arguments
        frame, id, input_path, input_file, eig_num = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5]
        folder_name = "mesh" + frame + "_" + id + "_vects"

    # Read input meshes
    mesh = read_polydata(input_path + input_file)

    folder = os.path.join(input_path, folder_name)
    os.makedirs(folder, exist_ok=True)
    
    # iterate
    for i in range(int(eig_num)): 
        print(i)
        pc, c = compute_pca(mesh, i)
        write_polydata(input_path + "/" + folder_name + "/" + folder_name + "_eig" + str(i + 1) + ".vtk", vtk_vector.apply_vector(pc, c))