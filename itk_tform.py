# ITKSNAP TRANSFORMATIONS
# created 05_24_2024

def write_itk_tform(filename, matrix): 
    """
    Writes a 4x4 np array to a itk transform txt file
    """
    with open(filename, 'w') as f:
        f.write("#Insight Transform File V1.0\n")
        f.write("# Transform 0\n")
        f.write("Transform: AffineTransform_double_3_3\n")
        f.write("Parameters: ")
        params = matrix[:3, :3].flatten().tolist() + matrix[:3, 3].tolist()
        f.write(' '.join(map(str, params)) + "\n")
        f.write("FixedParameters: 0 0 0\n")
