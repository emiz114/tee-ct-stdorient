# VTK FILE I/0
# created 05_27_2024

import vtk
import numpy as np

def read_polydata(input_path):
    """ 
    Loads VTK PolyData from the file specified by input_path
    - vtkPolyData: data structure used for geometric data (i.e. points, vertices, lines etc.)
    """
    # creates a reader for vtk polydata
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

def parse_filename(vtkfile): 
    """
    Removes the .vtk at the end of a file name
    """
    filename = ""
    for char in vtkfile:
        if char != '.':
            filename += char
        else:
            break
    return filename