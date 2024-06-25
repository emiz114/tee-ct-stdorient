# ORIENTS AV VTK SURFACE MESH TO STD AXES
# created 05_21_2024
# updated 06_24_2024

import os
import sys
import vtk
import numpy as np
import vtk_fileIO as vtkIO
import pca_align as pca
import commissure as com
import vtk_objects
import itk_tform

def calc_origin_tform(c): 
    """
    Calculates the transformation of bringing the centroid to the origin
    - c: specified centroid point
    """
    tform = vtk.vtkTransform()
    tform.PostMultiply()
    tform.Translate(-c) # brings to origin
    return tform

def calc_tform(pc, axis): 
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
    #tform.Translate(-c) # brings to origin
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

def find_ncusp_axis(meshes, method=1): 
    """
    Determines the axis running through NCusp to be aligned with the x axis
    Implements specified method
    """
    p_vect = np.array([0, 0, 0])
    
    lr_pt = com.locate_commissure(meshes[1], meshes[2], meshes[5])
    # l-r commissure vect
    if method == 0: 
        p_vect = lr_pt
        p_vect[2] = 0
    # l-n, r-n commissure bisection vect
    else:
        ln_pt = com.locate_commissure(meshes[1], meshes[3], meshes[5])
        rn_pt = com.locate_commissure(meshes[2], meshes[3], meshes[5])
        # vtk_objects.create_point(ln_pt, "/Users/emiz/Desktop/lnpt.vtk")
        # vtk_objects.create_point(rn_pt, "/Users/emiz/Desktop/rnpt.vtk")
        # calculate mid-point and vector from lr_pt
        for i in range(2): 
            p_vect[i] = lr_pt[i] - (ln_pt[i] + rn_pt[i])/2

    return p_vect

if __name__ == "__main__":

    if (len(sys.argv) != 4): 
        print("")
        print("******************************************************************")
        print("   USAGE: python3 | ./PATH/SCRIPT.py | /INPUT_PATH | FRAME | ID   ")
        print("******************************************************************")
        print("")
    # elif (int(sys.argv[4]) != 0 and int(sys.argv[4]) != 1): 
    #     print("")
    #     print("***************************************************")
    #     print("   METHOD 0 -> L-R COMMISSURE VECT   ")
    #     print("   METHOD 1 -> L-N:R-N COMMISSURE BISECTION VECT   ")
    #     print("***************************************************")
    #     print("")
    else: 
        # extract filenames from command line arguments
        input_path, frame, id = sys.argv[1], sys.argv[2], sys.argv[3]

    # path
    path = input_path + "/mesh" + frame + "_" + id + "/mesh" + frame + "_" + id

    # necessary vtk files: 6 
    components = np.array([".vtk",      # 0 multi
                       "_lcusp.vtk",    # 1 lcusp
                       "_rcusp.vtk",    # 2 rcusp
                       "_ncusp.vtk",    # 3 ncusp
                       "_rootwall.vtk", # 4 rootwall
                       "_stj.vtk",      # 5 stj
                       ])
    meshes = np.zeros(6, dtype = object)

    # read input meshes
    for i in range(6): 
        meshes[i] = vtkIO.read_polydata(path + components[i])
    
    # calculate principal components and centroids
    pc_rw, c_rw = pca.compute_pca(meshes[4])
    
    # calculate tform to origin
    tformo = calc_origin_tform(c_rw)
    print(tformo.GetMatrix())
    for i in range(5):
        meshes[i+1] = apply_tform(meshes[i+1], tformo)
    vtkIO.write_polydata(input_path + "/mesh" + frame + "_" + id + 
                         "/*mesh" + frame + "_" + id + "_o.vtk", apply_tform(meshes[0], tformo))

    # ORIENT Z AXIS
    # calculate tform for z
    tformz = calc_tform(pc_rw, c_rw, np.array([0, 0, 1]))

    # apply meshes
    for i in range(5): 
        meshes[i+1] = apply_tform(meshes[i+1], tformz)
    vtkIO.write_polydata(input_path + "/mesh" + frame + "_" + id + 
                         "/*mesh" + frame + "_" + id + "_z.vtk", apply_tform(meshes[0], tformz))

    # find ncusp vector / axis
    ncusp_vect = find_ncusp_axis(meshes)

    # ORIENT X AXIS
    tformx = calc_tform(ncusp_vect, np.array([0, 0, 0]), np.array([1, 0, 0]))
    vtkIO.write_polydata(input_path + "/mesh" + frame + "_" + id + 
                         "/*mesh" + frame + "_" + id + ".vtk", apply_tform(meshes[0], tformx))
    
    tformz.Concatenate(tformx)
    itk_tform.write_itk_tform(input_path + "/mesh" + frame + "_" + id + 
                             "/*mesh" + frame + "_" + id + "_tform.txt", tformz.GetMatrix())
    
    # APPLY TFORM TO ALL COMPONENTS
    # for i in range(6):
    #     meshes[i] = apply_tform(meshes[i], tformx)
    #     vtkIO.write_polydata(input_path + "file" + components[i], meshes[i])
