# PRINCIPAL COMPONENT ANALYSIS + ALIGNMENT
# adapted from @apouch github
# updated 05_21_2024

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
    # GetPoints converts vtkPolyData to vtkPoints
    # vtkPoints: object that contains a list of points in 3D space
    points = mesh.GetPoints()
    # retrieves data from vtkPoints to raw data points
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

def align_meshes(source, target):

    # compute centroids and principal components
    pc_source, c_source = compute_principal_component(source)
    pc_target, c_target = compute_principal_component(target)

    # Ensure consistent direction of principal components
    if np.dot(pc_source, pc_target) < 0:
        pc_source = -pc_source

    # print("PCA Fixed Mesh:", pc_target)
    # print("Centroid Fixed Mesh:", c_target)
    
    # print("PCA Moving Mesh:", pc_source)
    # print("Centroid Moving Mesh:", c_source)

    # rotate source to align with pc of target
    # computes cross product of the two eigenvectors to get axis of rotation
    rotation_axis = np.cross(pc_target, pc_source)
    # computes dot product then arccosine => A dot B = |A||B| cos(theta)
    rotation_angle = np.arccos(np.dot(pc_source, pc_target) / (np.linalg.norm(pc_source) * np.linalg.norm(pc_target)))

    # create transformation object (vtkTransform()) and applies rotation
    transform = vtk.vtkTransform()
    # ensures rotation is applied before transformation
    transform.PostMultiply() 

    transform.Translate(-c_source)
    transform.RotateWXYZ(np.degrees(-rotation_angle), *rotation_axis)
    transform.Translate(c_target)

    # apply transformation to source (moving)
    transform_filter_rigid = vtk.vtkTransformPolyDataFilter()
    transform_filter_rigid.SetInputData(source)
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

    if (len(sys.argv) != 5) and (len(sys.argv) != 6): 
        print("")
        print("*******************************************************************************************************************")
        print("   USAGE: python3 | ./PATH/SCRIPT.py | /INPUT_PATH | /SOURCE.vtk | /TARGET.vtk | /OUTPUT.vtk | /VECTOR_OUTPUT.vtk ")
        print("*******************************************************************************************************************")
        print("")
    else: 
        # extract filenames from command line arguments
        input_path, source_file, target_file, output_file = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
        if (len(sys.argv) == 6): 
            vect_bool = True
            vector_file = sys.argv[5]
        else: 
            vect_bool = False

    # Read input meshes
    source = read_polydata(input_path + source_file)
    target = read_polydata(input_path + target_file)

    # align meshes
    tform, alignedmesh = align_meshes(source, target)

    if vect_bool: 
        pc, c = compute_principal_component(alignedmesh)
        alignedvector = vtk_vector.apply_vector(alignedmesh)
        write_polydata(input_path + vector_file, alignedvector)
    
    write_polydata(input_path + output_file, alignedmesh)
