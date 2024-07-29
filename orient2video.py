# ORIENT SEGMENTATION DATASET TO VIDEO
# created 07_24_2024
# updated 07_29_2024

### NEED TO CHANGE INTERPRETER PATH ###
# /Applications/ParaView.app/Contents/bin/pvpython

# import sys
# if '--virtual-env' in sys.argv:
#   virtualEnvPath = sys.argv[sys.argv.index('--virtual-env') + 1]
#   # Linux
#   virtualEnv = virtualEnvPath + '/bin/activate_this.py'
#   if sys.version_info.major < 3:
#     execfile(virtualEnv, dict(__file__=virtualEnv))
#   else:
#     exec(open(virtualEnv).read(), {'__file__': virtualEnv})

# import SimpleITK as sitk
#from paraview.simple import *
import os
import vtk
import numpy as np
# from vtkmodules.util.numpy_support import vtk_to_numpy
# import extract_mesh
# import commissure as com
from moviepy.editor import VideoFileClip

### FILE IO ###

def parse_frame_id(filename): 
    """
    Parses a .vtk, .nii.gz, etc. file for av patient frame and id
    """
    frame = ""
    id = ""
    mode = 0
    for char in filename: 
        if mode == 0: 
            if char.isdigit(): 
                frame +=char
            if char == "_": 
                mode = 1
        if mode == 1: 
            if char == ".": 
                break
            id += char
    return frame, id

### VTK FILE IO ###

def read_polydata(input_path):
    """ 
    Loads VTK PolyData from the file specified by input_path
    - vtkPolyData: data structure used for geometric data (i.e. points, vertices, lines etc.)
    """
    reader = vtk.vtkPolyDataReader()
    reader.SetFileName(input_path)
    reader.Update()

    return reader.GetOutput()

def write_polydata(output_path, polydata): 
    """
    Writes VTK polydata to a file specified by output_path
    """
    writer = vtk.vtkPolyDataWriter()
    writer.SetFileName(output_path)
    writer.SetInputData(polydata)
    writer.Write()

### COMPUTE PCA ###

def compute_pca(mesh):
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

def correct_pca(pc, mesh_stj): 
    """
    Corrects the pca vector to the direction of the stj
    - pc: specified direction vector
    - mesh_stj: surface mesh of stj (meshes[5])
    """
    _, c_stj = compute_pca(mesh_stj)
    pc1 = np.linalg.norm(c_stj - pc)
    pc2 = np.linalg.norm(c_stj + pc)
    if pc1 > pc2: 
        return -pc
    else: 
        return pc

### EXTRACT SURFACE MESH ###

# def extract_surface_meshes(seg_img_path): 
#     """
#     Extracts all necessary surface meshes from a segmentation img.
#     """
#     meshes = np.zeros(6, dtype = object)
#     # meshes[0] = extract_mesh.extract_multi_mesh(seg_img_path)
#     for i in range(5): 
#         meshes[i] = extract_mesh.extract_mesh(seg_img_path, i)
#     meshes[5] = extract_mesh.extract_mesh(seg_img_path, 6)
#     return meshes

### CALCULATE AND APPLY TFORMS ###

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
    pc_rw, c_rw = compute_pca(meshes[4])
    pc = correct_pca(pc_rw, meshes[5])

    # calculate tform to origin
    tform = calc_origin_tform(c_rw)
    for i in range(6):
        meshes[i] = apply_tform(meshes[i], tform)
    print(tform)
    # ORIENT Z AXIS
    # calculate tform for z
    tformz = calc_tform(pc, np.array([0, 0, 1]))

    # apply meshes
    for i in range(6):
        meshes[i] = apply_tform(meshes[i], tformz)

    # find ncusp vector / axis
    ncusp_vect = find_ncusp_axis(meshes)

    # ORIENT X AXIS
    tformx = calc_tform(ncusp_vect, np.array([1, 0, 0]))
    write_polydata(output_path, apply_tform(meshes[0], tformx))
                
    tform.Concatenate(tformz)
    tform.Concatenate(tformx)

    return tform

folder_path = "/Users/emiz/Desktop/bavcta001/baseline/"
id = "bavcta001_baseline"
meshes_path = folder_path + "/oriented_meshes"
tforms_path = folder_path + "/tforms"

### CONVERT TO .MOV ###

def avi2mov(file_path):
    """
    Converts the .avi file to a .mov file
    """
    vid = VideoFileClip(file_path + ".avi")
    vid.write_videofile(file_path + ".mov", codec='libx264', audio_codec='aac')
    os.remove(file_path + ".avi")

### ORIENT ###

if not os.path.exists(meshes_path):
    os.makedirs(meshes_path)
if not os.path.exists(tforms_path):
    os.makedirs(tforms_path)

# for seg in os.listdir(folder_path):
#     seg_img_path = os.path.join(folder_path, seg)
#     if os.path.isfile(seg_img_path):
#         if (seg != ".DS_Store"):
#             frame, id = parse_frame_id(seg)
#             meshes = extract_surface_meshes(seg_img_path)
#             output_path = meshes_path + "/mesh" + frame + ".vtk"
#             tform = orient(meshes, output_path)
#             print(seg)

### OPEN IN PARAVIEW AND ANIMATE ###

# renderView1 = CreateRenderView()
# renderView1.ViewSize = [512, 512]

files = []
for mesh in os.listdir(meshes_path):
    mesh_img_path = os.path.join(meshes_path, mesh)
    if os.path.isfile(mesh_img_path):
        if (mesh != ".DS_Store"):
            files.append(mesh_img_path)
            print(mesh)

# #files.sort()         
# reader = OpenDataFile(files)
# scene = GetAnimationScene()
# scene.PlayMode = 'Sequence'
# SaveAnimation(folder_path + id + ".avi", View=renderView1, ImageResolution=[512,512])

avi2mov(folder_path + id)