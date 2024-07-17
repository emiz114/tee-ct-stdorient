# ORIENTS AV VTK SURFACE MESH TO STD AXES (with segmentation img)
# created 07_08_2024
# updated 07_17_2024

import os
import sys
import vtk
import numpy as np
import vtk_fileIO as vtkIO
import pca_align as pca
import commissure as com
import SimpleITK as sitk
import extract_mesh
import parsefile

def extract_surface_meshes(seg_img_path): 
    """
    Extracts all necessary surface meshes from a segmentation img.
    """
    meshes = np.zeros(6, dtype = object)
    # meshes[0] = extract_mesh.extract_multi_mesh(seg_img_path)
    for i in range(5): 
        meshes[i] = extract_mesh.extract_mesh(seg_img_path, i)
    meshes[5] = extract_mesh.extract_mesh(seg_img_path, 6)
    return meshes

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
    # if np.dot(pc, axis) < 0:
    #     pc = -pc

    rotation_axis = np.cross(axis, pc)
    rotation_angle = np.arccos(np.dot(pc, axis) / (np.linalg.norm(pc) * np.linalg.norm(axis)))

    # create transformation object (vtkTransform()) and applies rotation
    tform = vtk.vtkTransform()
    tform.PostMultiply()
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
    
    lr_pt = com.locate_commissure(meshes[1], meshes[3], meshes[5])
    # l-r commissure vect
    if method == 0: 
        p_vect = lr_pt
        p_vect[2] = 0
    # l-n, r-n commissure bisection vect
    else:
        ln_pt = com.locate_commissure(meshes[1], meshes[2], meshes[5])
        rn_pt = com.locate_commissure(meshes[3], meshes[2], meshes[5])
        # vtk_objects.create_point(ln_pt, "/Users/emiz/Desktop/lnpt.vtk")
        # vtk_objects.create_point(rn_pt, "/Users/emiz/Desktop/rnpt.vtk")
        # calculate mid-point and vector from lr_pt
        for i in range(2): 
            p_vect[i] = (ln_pt[i] + rn_pt[i])/2 - lr_pt[i]
    #print(p_vect)
    return p_vect

def orient(meshes, output_path): 
    """
    Applies orientation: 
    (1) Bring to origin
    (2) Rotate pca to z-axis
    (3) Rotate NCusp bisection to x-axis
    - meshes: np.array of surface meshes
    """
    # calculate principal components and centroids
    pc_rw, c_rw = pca.compute_pca(meshes[4])

    # calculate tform to origin
    tform = calc_origin_tform(c_rw)
    for i in range(6):
        meshes[i] = apply_tform(meshes[i], tform)

    # ORIENT Z AXIS
    # calculate tform for z
    tformz = calc_tform(pc_rw, np.array([0, 0, 1]))

    # apply meshes
    for i in range(6):
        meshes[i] = apply_tform(meshes[i], tformz)

    # find ncusp vector / axis
    ncusp_vect = find_ncusp_axis(meshes)

    # ORIENT X AXIS
    tformx = calc_tform(ncusp_vect, np.array([1, 0, 0]))
    vtkIO.write_polydata(output_path, apply_tform(meshes[0], tformx))
                
    tform.Concatenate(tformz)
    tform.Concatenate(tformx)

    return tform

if __name__ == "__main__":

    if (len(sys.argv) != 4): 
        print("")
        print("************************************************************************************************")
        print("   USAGE: python3 | ./PATH/SCRIPT.py | /SEG_INPUT_PATH | FRAME(0 FOR ALL SEGS IN FOLDER) | ID   ")
        print("************************************************************************************************")
        print("")
    else: 
        # extract filenames from command line arguments
        folder_path, frame, id = sys.argv[1], sys.argv[2], sys.argv[3]

    # necessary vtk files: 6
    # 0 multi
    # 1 lcusp
    # 2 ncusp
    # 3 rcusp
    # 4 rootwall
    # 5 stj

    if (frame != "0"): 
        seg_img_path = os.path.join(folder_path, "seg" + frame + "_CT_" + id + ".nii.gz")
        meshes = extract_surface_meshes(seg_img_path)
        # example = meshes[0]
        output_path = "/Users/emiz/Desktop/research/bavcta001_baseline_meshes/" + frame + ".vtk"
        tform = orient(meshes, output_path)
    else: 
        for item in os.listdir(folder_path):
            seg_img_path = os.path.join(folder_path, item)
            # Check if the item is a file (and not a folder)
            if os.path.isfile(seg_img_path):
                new_folder_path = folder_path + "/oriented_meshes"
                if not os.path.exists(new_folder_path):
                        os.makedirs(new_folder_path)
                if (item != ".DS_Store"):
                    print(item)
                    frm, id = parsefile.parse_frame_id(item)
                    meshes = extract_surface_meshes(seg_img_path)
                    output_path = new_folder_path + "/mesh" + frm + ".vtk"
                    tform = orient(meshes, output_path)

            
