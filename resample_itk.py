# RESAMPLE IMAGE IN ITK
# created 06_20_2024
# updated

import SimpleITK as sitk
import numpy as np

# define the dimension
dim = 3

# define input, output file names
input = "/Users/emiz/Desktop/img13_bavcta001_baseline.nii.gz"
output = "/Users/emiz/Desktop/img13_bavcta001_baseline_oriented.nii.gz"

# mesh13_bavcta001_baseline tform matrix 
# vtk_matrix = np.array([
#     [5.31623339e-01, -5.35937537e-01, -6.55856373e-01, -2.87067760e+01],
#     [9.39464939e-02,  8.06880227e-01, -5.83196670e-01, -2.66670143e+02],
#     [8.41754526e-01,  2.48425554e-01,  4.79305813e-01,  7.71682369e+01],
#     [0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  1.00000000e+00]])

vtk_matrix = np.array([
    [1.0, 0.0, 0.0, 24.6428], 
    [0.0, 1.0, 0.0, -180.615], 
    [0.0, 0.0, 1.0, 211.336], 
    [0.0, 0.0, 0.0, 1.0]
])

rotation = vtk_matrix[:3, :3].flatten()
translation = vtk_matrix[:3, 3].tolist()

# read input image
reader = sitk.ImageFileReader()
reader.SetFileName(input)
input_image = reader.Execute()

# set up affine tform
tform = sitk.AffineTransform(dim)
# tform.SetMatrix(rotation)
tform.SetTranslation(translation)

# setup resample filter
resample_filter = sitk.ResampleImageFilter()
resample_filter.SetTransform(tform)
resample_filter.SetReferenceImage(input_image)

# set up interpolator
interpolator = sitk.sitkNearestNeighbor
resample_filter.SetInterpolator(interpolator)

# define the output image size, spacing, and origin
size = (512, 512, 410)
vox = (0.367188, 0.367188, 0.3)
origin = (-61.82, -255.3, -150.3)
# origin = (0, 0, 0)

resample_filter.SetOutputSpacing(vox)
resample_filter.SetOutputOrigin(origin)
resample_filter.SetSize(size)

# execute
output_image = resample_filter.Execute(input_image)

# write output image
writer = sitk.ImageFileWriter()
writer.SetFileName(output)
writer.Execute(output_image)