# adapted from pca.py and icp_decimated.py
# updated 05_20_2024

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

def decimate(file):
    """
    Applies a decimation filter to a VTK surface mesh
    """
    # determines percentage of polygons to keep
    reduction_factor = 0.1 # 10%

    # creates instance of the decimation
    decimate = vtk.vtkQuadricClustering()
    decimate.SetInputData(file)

    # determine number of polygons in output mesh
    divisions = 1 + int(1 / reduction_factor)
    # sets number of divisions in each dimension (3D -> 3)
    decimate.SetNumberOfDivisions(divisions, divisions, divisions)
    decimate.Update()
        
    return decimate.GetOutput()

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

    print("PCA Fixed Mesh:", pc_target)
    print("Centroid Fixed Mesh:", c_target)
    
    print("PCA Moving Mesh:", pc_source)
    print("Centroid Moving Mesh:", c_source)

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

    return transform_filter_rigid.GetOutput(), transform

def iterative_closest_point(aligned_source, target):
    """
    Computes the transformation required to align source PolyData to target PolyData
    Uses IterativeClosestPoint (ICP) algorithm
    """
    icp = vtk.vtkIterativeClosestPointTransform()
    icp.SetSource(aligned_source)
    icp.SetTarget(target)
    icp.GetLandmarkTransform().SetModeToSimilarity()
    icp.SetMaximumNumberOfLandmarks(aligned_source.GetNumberOfPoints())
    icp.SetMaximumNumberOfIterations(5000)
    icp.StartByMatchingCentroidsOn()
    icp.Modified()

    return icp

def apply_icp(source, icp, pca):
    """
    Applies the icp transformation on the pca_aligned source
    Concatenates pca + icp transform and finds t_matrix
    """
    transform_filter = vtk.vtkTransformPolyDataFilter()
    transform_filter.SetInputData(source)
    transform_filter.SetTransform(icp)
    transform_filter.Update()

    aligned_source = transform_filter.GetOutput()

    pca.Concatenate(icp)
    t_matrix = pca.GetMatrix()

    return aligned_source, t_matrix

def write_polydata(output_path, aligned_polydata):
    """
    Writes the aligned PolyData object to the specified path
    """ 
    # Save aligned mesh
    writer = vtk.vtkPolyDataWriter()
    writer.SetFileName(output_path)
    writer.SetInputData(aligned_polydata)
    writer.Write()

if __name__ == "__main__":

    if len(sys.argv) != 5: 
        print("")
        print("*******************************************************************************************************************")
        print("   USAGE: python3 | ./PATH/SCRIPT.py | /INPUT_PATH | /SOURCE.vtk | /TARGET.vtk | /OUTPUT.vtk   ")
        print("*******************************************************************************************************************")
        print("")
    else: 
        # extract filenames from command line arguments
        input_path, source_file, target_file, output_file = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]

    # Read input meshes (nd)
    source_nd = read_polydata(input_path + source_file)
    target_nd = read_polydata(input_path + target_file)

    # align meshes
    source_nd_PCAaligned, pca = align_meshes(source_nd, target_nd)

    # apply decimation filter to aligned mesh and target
    source_PCA_aligned = decimate(source_nd_PCAaligned)
    target = decimate(target_nd)

    # finds and applies icp
    icp = iterative_closest_point(source_PCA_aligned, target)
    source_aligned, t_matrix = apply_icp(source_nd_PCAaligned, icp, pca)

    write_polydata(input_path + output_file, source_aligned)