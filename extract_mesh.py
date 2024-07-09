# EXTRACTS SURFACE MESH LABELS FROM AN IMAGE/SEGMENTATION
# created 06_25_2024
# updated 07_08_2024

import SimpleITK as sitk
import vtk
import numpy as np
from vtkmodules.util.numpy_support import vtk_to_numpy


def sitk_to_vtk(img): 
    """
    Converts a SimpleITK image to a VTK image.
    """
    # dictionary mapping sitk pixels to vtk pixels
    sitk_to_vtk_type = {
        sitk.sitkUInt8: vtk.VTK_UNSIGNED_CHAR,
        sitk.sitkInt8: vtk.VTK_CHAR,
        sitk.sitkUInt16: vtk.VTK_UNSIGNED_SHORT,
        sitk.sitkInt16: vtk.VTK_SHORT,
        sitk.sitkUInt32: vtk.VTK_UNSIGNED_INT,
        sitk.sitkInt32: vtk.VTK_INT,
        sitk.sitkUInt64: vtk.VTK_UNSIGNED_LONG,
        sitk.sitkInt64: vtk.VTK_LONG,
        sitk.sitkFloat32: vtk.VTK_FLOAT,
        sitk.sitkFloat64: vtk.VTK_DOUBLE,
    }
    # creates a new vtk image
    vtk_img = vtk.vtkImageData()
    vtk_img.SetDimensions(img.GetSize())
    vtk_img.SetSpacing(img.GetSpacing())
    vtk_img.SetOrigin(img.GetOrigin())
    vtk_img.AllocateScalars(sitk_to_vtk_type[img.GetPixelID()], 1)
    # get np array of sitk img
    np_img = sitk.GetArrayViewFromImage(img)
    np_vtk_img = vtk.util.numpy_support.vtk_to_numpy(vtk_img.GetPointData().GetScalars()).reshape(np_img.shape)

    # copy the data
    np_vtk_img[:] = np_img
    
    return vtk_img

def extract_mesh(seg_img_path, label_index): 
    """
    Extracts a surface mesh from a segmentation image. 
    """
    seg = sitk.ReadImage(seg_img_path)
    
    # binarizes the image by setting voxels with specified label to 1, all others 0
    binarized_img = sitk.BinaryThreshold(seg, lowerThreshold=label_index, upperThreshold=label_index, insideValue=1, outsideValue=0)
    # convert sitk img to vtk img
    vtk_img = sitk_to_vtk(binarized_img)
    
    # apply marching cubes algorithm to extract surface mesh
    marching_cubes = vtk.vtkMarchingCubes()
    marching_cubes.SetInputData(vtk_img)
    # set iso threshold to 0.5 to locate surface (binary image is either 0 or 1)
    marching_cubes.SetValue(0, 0.5) 
    marching_cubes.Update()

    return marching_cubes.GetOutput()

def combine_meshes(meshes):
    """
    Combines multiple surface meshes into a single PolyData Object
    """
    append_filter = vtk.vtkAppendPolyData()
    for mesh in meshes:
        append_filter.AddInputData(mesh)
    append_filter.Update()
    return append_filter.GetOutput()

def extract_multi_mesh(seg_img_path): 
    """
    Extracts the multi-component surface mesh from a segmentation image. 
    """
    seg = sitk.ReadImage(seg_img_path)
    meshes = np.zeros(16, dtype=object)
    for i in range(16): 
        meshes[i] = extract_mesh(seg, i)
    return combine_meshes(meshes)

# example
# seg_img_path = "/Users/emiz/Desktop/research/picsl/stdorient/seg13_bavcta001_baseline.nii.gz"
# output_path = "/Users/emiz/Desktop/research/picsl/stdorient/attempt.vtk"
# label_index = 1

# extract_mesh(seg_img_path, output_path, label_index)
