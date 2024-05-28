# ORIENTS AV TO AXES
# created 05_21_2024
# updated 05_28_2024

import os
import sys
import vtk
import numpy as np
import vtk_fileIO as vtkIO
import pca_align as pca
import commissure as com

def calc_tform(pc, c, axis): 
    """
    Calculates the transformation of aligning pc from centroid to axis
    - pc: specified direction vector
    - c: specified centroid point
    - axis: numpy array of unit axis vector
    """
    if np.dot(pc, axis) < 0:
        pc = -pc

    rotation_axis = np.cross(axis, pc)
    rotation_angle = np.arccos(np.dot(pc, axis) / (np.linalg.norm(pc) * np.linalg.norm(axis)))

    # create transformation object (vtkTransform()) and applies rotation
    tform = vtk.vtkTransform()
    tform.PostMultiply()
    tform.Translate(-c) # brings to origin
    tform.RotateWXYZ(np.degrees(-rotation_angle), *rotation_axis)

    return tform

def apply_tform(mesh, tform): 
    """
    Applies input transformation to specified vtk mesh.
    """
    tform_filter = vtk.vtkTransformPolyDataFilter()
    tform_filter.SetInputData(mesh)
    tform_filter.SetTransform(tform)
    tform_filter.Update()

    return tform_filter.GetOutput()

if __name__ == "__main__":

    if (len(sys.argv) != 4): 
        print("")
        print("************************************************************************************")
        print("   USAGE: python3 | ./PATH/SCRIPT.py | /INPUT_PATH | FRAME | ID ")
        print("************************************************************************************")
        print("")
    else: 
        # extract filenames from command line arguments
        input_path, frame, id = sys.argv[1], sys.argv[2], sys.argv[3]

    # path
    path = input_path + "/mesh" + frame + "_" + id

    # necessary vtk files: 6 
    # order: multi, vaj, rootwall, lcusp, rcusp, ncusp
    components = np.array([".vtk", 
                       "_vaj.vtk", 
                       "_rootwall.vtk", 
                       "_lcusp.vtk", 
                       "_rcusp.vtk", 
                       "_ncusp.vtk"])
    meshes = np.zeros(6, dtype = object)

    # ORIENT Z AXIS
    # read input meshes
    for i in range(6): 
        print(path + components[i])
        meshes[i] = vtkIO.read_polydata(path + components[i])
    
    # calculate principal components and centroids
    pc_vaj, c_vaj = pca.compute_pca(meshes[1])
    pc_rw, c_rw = pca.compute_pca(meshes[2])
    
    # calculate tform
    tformz = calc_tform(pc_rw, c_rw, np.array([0, 0, 1]))
    
    # apply meshes
    for i in range(6): 
        meshes[i] = apply_tform(meshes[i], tformz)

    # FIND L_R COMMISSURE
    p_cusp, _ = com.locate_commissure(meshes[3], meshes[4], meshes[1])
    # project point onto centroid x-y plane
    p_cusp[2] = 0

    points = vtk.vtkPoints()
    points.InsertNextPoint(p_cusp)
    polyData = vtk.vtkPolyData()
    polyData.SetPoints(points)
    vtkIO.write_polydata(input_path + "/mesh01_bavcta001_baseline_lr.vtk", polyData)

    # ORIENT X AXIS
    tformx = calc_tform(p_cusp, np.array([0, 0, 0]), np.array([1, 0, 0]))
    vtkIO.write_polydata(input_path + "/mesh" + frame + "_" + id + "_oriented.vtk", apply_tform(meshes[0], tformx))
    # for i in range(6):
    #     meshes[i] = apply_tform(meshes[i], tformx)
    #     vtkIO.write_polydata(input_path + "file" + components[i], meshes[i])