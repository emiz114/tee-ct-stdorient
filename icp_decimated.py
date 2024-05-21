# ITERATIVE CLOSEST POINT (ICP) WITH DECIMATION FILTER
# adapted from @apouch github
# updated 05_16_2024

import os
import sys
import vtk

def load_polydata(file_path, flag_decimate): 
    """ 
    Loads VTK PolyData from the file specified by file_path
    Applies decimation filter if flag_decimate is True
    """
    reader = vtk.vtkPolyDataReader()
    reader.SetFileName(file_path)
    reader.Update()

    # applies decimation filter
    # used to reduce number of polygons in a mesh while preserving shape

    if flag_decimate:
        # determines percentage of polygons to keep
        reduction_factor = 0.1 # 10%

        # creates instance of the decimation
        decimate = vtk.vtkQuadricClustering()
        decimate.SetInputData(reader.GetOutput())

        # determine number of polygons in output mesh
        divisions = 1 + int(1 / reduction_factor)
        # sets number of divisions in each dimension (3D -> 3)
        decimate.SetNumberOfDivisions(divisions, divisions, divisions)
        decimate.Update()
        
        polydata = decimate.GetOutput()
        
    else:
        
        polydata = reader.GetOutput()
    
    return polydata

def compute_tform_icp(source, target): 
    """
    Computes the transformation required to align source PolyData to target PolyData
    Uses IterativeClosestPoint (ICP) algorithm
    """
    # creates an instance of icp transformation object
    icp = vtk.vtkIterativeClosestPointTransform()

    # set source and target
    icp.SetSource(source)
    icp.SetTarget(target)

    # similarity includes translation, rotation, and uniform scaling
    icp.GetLandmarkTransform().SetModeToSimilarity()

    icp.SetMaximumNumberOfLandmarks(source.GetNumberOfPoints())
    icp.SetMaximumNumberOfIterations(5000)

    icp.StartByMatchingCentroidsOn()
    icp.Modified()

    return icp

def apply_tform(source, icp):
    """
    Applies the icp transformation on the source PolyData object
    """
    transform_filter = vtk.vtkTransformPolyDataFilter()
    transform_filter.SetInputData(source)
    transform_filter.SetTransform(icp)
    transform_filter.Update()

    return transform_filter.GetOutput()

def write_aligned_polydata(output_path, aligned_polydata):
    """
    Writes the aligned PolyData object to the specified path
    """
    writer = vtk.vtkPolyDataWriter()
    writer.SetFileName(output_path)
    writer.SetInputData(aligned_polydata)
    writer.Write()

if __name__ == "__main__":

    if len(sys.argv) != 5: 
        print("")
        print("*******************************************************************************************************************")
        print("   USAGE: python3 | ./PATH/SCRIPT.py | /INPUT_PATH | /SOURCE.vtk | /TARGET.vtk | /OUTPUT_PATH/OUTPUT.vtk   ")
        print("*******************************************************************************************************************")
        print("")
    else: 
        # extract filenames from command line arguments
        input_path, source_file, target_file, output_file = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
    
    # load PolyData
    source = load_polydata(input_path + source_file, True)
    target = load_polydata(input_path + target_file, True)

    # obtain alignment with icp
    icp = compute_tform_icp(source, target)

    # find non-decimated source
    source_nd = load_polydata(input_path + source_file, False)

    # apply tform
    aligned = apply_tform(source_nd, icp)
    write_aligned_polydata(output_file, aligned)
