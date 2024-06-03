# LOCATES CUSP COMMISSURE POINT
# created 05_27_2024
# updated 06_03_2024

import sys
import vtk 
import numpy as np
from vtkmodules.util.numpy_support import vtk_to_numpy
import vtk_fileIO as vtkIO
from scipy.spatial import KDTree
import vtk_objects

def mesh_to_points(mesh):
    """
    *** MAY NEED TO BE MODIFIED FOR NON OVERLAPPING POINTS ***
    """
    points = vtk_to_numpy(mesh.GetPoints().GetData())

    return points

def closest_triplet(mesh1, mesh2, mesh3): 
    """
    """
    points1 = mesh_to_points(mesh1)
    points2 = mesh_to_points(mesh2)
    points3 = mesh_to_points(mesh3)

    # Construct k-d trees
    tree1 = KDTree(points1)
    tree2 = KDTree(points2)
    tree3 = KDTree(points3)

    # Initialize minimum distance to a large value
    min_distance = float('inf')
    closest_points = (None, None, None)

    # For each point in P1, find the closest point in P2
    for p1 in points1:
        # Find the closest point in cloud2 to point1
        dist2, idx2 = tree2.query(p1)
        p2 = points2[idx2]
        
        # Find the closest point in cloud3 to the midpoint of point1 and point2
        midpoint12 = (p1 + p2) / 2
        dist3, idx3 = tree3.query(midpoint12)
        p3 = points3[idx3]
        
        # Calculate the total distance for this triplet
        total_distance = np.linalg.norm(p1 - p2) + np.linalg.norm(p2 - p3) + np.linalg.norm(p3 - p1)
        
        # Update the minimum distance and best triplet if necessary
        if total_distance < min_distance:
            min_distance = total_distance
            closest_points = (p1, p2, p3)
    
    return closest_points

def locate_commissure(mesh_cusp1, mesh_cusp2, mesh_stj): 
    """
    Locates the commissure point between two root cusps. 
    """
    p1, p2, _ = closest_triplet(mesh_cusp1, mesh_cusp2, mesh_stj)
    return (p1 + p2) / 2

# def closest_point(mesh1, mesh2):
#     """
#     Finds the closest point pair between two point clouds
#     Uses a KDTree to search
#     """
#     points1 = vtk_to_numpy(mesh1.GetPoints().GetData())
#     points2 = vtk_to_numpy(mesh2.GetPoints().GetData())

#     # construct kdtrees
#     tree1 = KDTree(points1)
#     tree2 = KDTree(points2)

#     # initialize minimum distance to a large value
#     min_distance = float('inf')
#     closest_point_pair = (None, None)

#     # for each point in P1, find the closest point in P2
#     for point in points1:
#         distance, index = tree2.query(point)
#         if distance < min_distance:
#             min_distance = distance
#             closest_point_pair = (point, points2[index])
    
#     return closest_point_pair

# def locate_commissure(mesh_cusp1, mesh_cusp2, mesh_stj): 
#     """
#     Locates the commissure point closest to the STJ
#     """
#     p1, _ = closest_point(mesh_cusp1, mesh_stj)
#     p2, _ = closest_point(mesh_cusp2, mesh_stj)
#     return (p1 + p2)/2
