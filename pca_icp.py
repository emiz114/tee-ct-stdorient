# adapted from @apouch github
# updated 04_18_2024

import sys
import vtk
import numpy as np
from vtkmodules.util.numpy_support import vtk_to_numpy

def read_vtkpolydata(filename):
    
    # creates a reader for vtk polydata
    reader = vtk.vtkPolyDataReader()
    reader.SetFileName(filename)
    # updates the reader to read data
    reader.Update()
    # retrieves output
    mesh = reader.GetOutput()
    # mesh is of type vtkPolyData
    # vtkPolyData: data structure used for geometric data (i.e. points, vertices, lines etc.)
    return mesh

def compute_principal_component(mesh):
    
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

    return centroid, principal_component

def align_meshes(mesh1, mesh2):

    # compute centroids and principal components
    pc_1, c_1 = compute_principal_component(mesh1)
    pc_2, c_2 = compute_principal_component(mesh2)

    print("PCA Fixed Mesh:", pc_1)
    print("Centroid Fixed Mesh:", pc_1)
    
    print("PCA Moving Mesh:", pc_2)
    print("Centroid Moving Mesh:", pc_2)

    # align centroids by moving c_2 (moving) to origin? (DONT UNDERSTAND)
    translation = -c_2

    # rotate mesh2 to align with pc of mesh1
    # computes cross product of the two eigenvectors to get axis of rotation
    rotation_axis = np.cross(pc_1, pc_2)
    # computes dot product then arccosine => A dot B = |A||B| cos(theta)
    rotation_angle = np.arccos(np.dot(pc_1, pc_2) / (np.linalg.norm(pc_1) * np.linalg.norm(pc_2)))

    # create transformation object (vtkTransform()) and applies rotation
    transform = vtk.vtkTransform()
    # ensures rotation is applied before transformation
    transform.PostMultiply() 
    # moves c_2 to origin
    transform.Translate(translation)
    # rotates the transform
    transform.RotateWXYZ(np.degrees(-rotation_angle), *rotation_axis)
    # moves c_2 from origin to c_1
    transform.Translate(c_1)

    # apply transformation to mesh2 (moving)
    # vtkTransformPolyDataFilter object transforms polygonal data objects
    transform_filter_rigid = vtk.vtkTransformPolyDataFilter()
    # sets input mesh
    transform_filter_rigid.SetInputData(mesh2)
    # applies transform
    transform_filter_rigid.SetTransform(transform)
    transform_filter_rigid.Update()

    # retrieves transformed mesh
    mesh2_rigid_align = transform_filter_rigid.GetOutput()
    
    return transform, mesh2_rigid_align

def iterative_closest_point(transform, mesh1, mesh2, t_mesh2):

    icp = vtk.vtkIterativeClosestPointTransform()
    icp.SetSource(t_mesh2)
    icp.SetTarget(mesh1)
    icp.GetLandmarkTransform().SetModeToSimilarity()
    icp.SetMaximumNumberOfLandmarks(t_mesh2.GetNumberOfPoints())
    icp.SetMaximumNumberOfIterations(100)
    icp.Modified()

    transform.Concatenate(icp)

    transform_filter_cat = vtk.vtkTransformPolyDataFilter()
    transform_filter_cat.SetInputData(mesh2)
    transform_filter_cat.SetTransform(transform)
    transform_filter_cat.Update()

    t_mesh2_new = transform_filter_cat.GetOutput()
    t_matrix = transform.GetMatrix()

    return t_mesh2_new, t_matrix

# Read input meshes
mesh1 = read_vtkpolydata("/Users/emiz/Desktop/Research/picsl/practice/img12_bav02/mesh12_bav02_rootwall.vtk")
mesh2 = read_vtkpolydata("/Users/emiz/Desktop/Research/picsl/practice/img12_bav02/mesh12_bav02_rootwall_moving.vtk")

# align meshes
tform, t_mesh2 = align_meshes(mesh1, mesh2)

# apply icp
tform_icp, t_matrix = iterative_closest_point(tform, mesh1, mesh2, t_mesh2)

# note: takes a while to load
print("Transform:", t_mesh2)

# Invert the transform
inverse_transform = vtk.vtkTransform()
inverse_transform.SetMatrix(t_matrix)
inverse_transform.Inverse()
t_matrix_inv = inverse_transform.GetMatrix()
    
# Save aligned mesh
writer = vtk.vtkPolyDataWriter()
writer.SetFileName("/Users/emiz/Desktop/Research/picsl/practice/img12_bav02/mesh12_bav02_rootwall_aligned.vtk")
writer.SetInputData(t_mesh2)
writer.Write()
