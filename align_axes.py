# ALIGNS PCA TO AXES
# created 05_21_2024

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

def compute_principal_component(mesh):
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
    principal_component = eigenvectors[:, sort_indices[0]]

    return principal_component, centroid

def align_pca(mesh): 
    """
    Aligns the PCA eigenvector to the positive z-axis
    """
    pc, c = compute_principal_component(mesh)
    z = np.array([0, 0, 1])

    # Ensure consistent direction of principal components
    if np.dot(pc, z) < 0:
        pc = -pc

    # rotate source to align with pc of target
    # computes cross product of the two eigenvectors to get axis of rotation
    rotation_axis = np.cross(z, pc)
    # computes dot product then arccosine => A dot B = |A||B| cos(theta)
    rotation_angle = np.arccos(np.dot(pc, z) / (np.linalg.norm(pc) * np.linalg.norm(z)))

    # create transformation object (vtkTransform()) and applies rotation
    transform = vtk.vtkTransform()
    # ensures rotation is applied before transformation
    transform.PostMultiply() 

    transform.Translate(-c)
    transform.RotateWXYZ(np.degrees(-rotation_angle), *rotation_axis)
    transform.Translate(c)

    # apply transformation to source (moving)
    transform_filter_rigid = vtk.vtkTransformPolyDataFilter()
    transform_filter_rigid.SetInputData(mesh)
    transform_filter_rigid.SetTransform(transform)
    transform_filter_rigid.Update()

    return transform, transform_filter_rigid.GetOutput()

def write_polydata(output_path, aligned_polydata): 
    # Save aligned mesh
    writer = vtk.vtkPolyDataWriter()
    writer.SetFileName(output_path)
    writer.SetInputData(aligned_polydata)
    writer.Write()

if __name__ == "__main__":

    if (len(sys.argv) != 6): 
        print("")
        print("*******************************************************************************************************************")
        print("   USAGE: python3 | ./PATH/SCRIPT.py | /INPUT_PATH | /INPUT.vtk | /OUTPUT.vtk | /PCA_VECTOR.vtk | /Z_VECTOR.vtk ")
        print("*******************************************************************************************************************")
        print("")
    else: 
        # extract filenames from command line arguments
        input_path, input_file, output_file, pca_vector_file, z_vector_file = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5]

    # Read input meshes
    mesh = read_polydata(input_path + input_file)

    # align pca
    tform, alignedmesh = align_pca(mesh)
    
    write_polydata(input_path + output_file, alignedmesh)

    # draw pca and z-axis
    pc, c = compute_principal_component(mesh)
    pc_vector = vtk_vector.apply_vector(pc, c)
    write_polydata(input_path + pca_vector_file, pc_vector)
    z_vector = vtk_vector.apply_vector(np.array([0, 0, 1]), c)
    write_polydata(input_path + z_vector_file, z_vector)
