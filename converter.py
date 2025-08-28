import numpy as np
from scipy.interpolate import interp1d

# CIE 1931 2-degree observer color matching functions, sampled every 10 nm
CIE_CMF_DATA = np.array([
    [380, 0.0014, 0.0000, 0.0065],
    [390, 0.0042, 0.0001, 0.0201],
    [400, 0.0143, 0.0004, 0.0679],
    [410, 0.0435, 0.0012, 0.2074],
    [420, 0.1344, 0.0040, 0.6456],
    [430, 0.2839, 0.0116, 1.3856],
    [440, 0.3483, 0.0230, 1.7471],
    [450, 0.3362, 0.0380, 1.7721],
    [460, 0.2908, 0.0600, 1.6692],
    [470, 0.1954, 0.0910, 1.2876],
    [480, 0.0956, 0.1390, 0.8130],
    [490, 0.0320, 0.2080, 0.4652],
    [500, 0.0049, 0.3230, 0.2720],
    [510, 0.0093, 0.5030, 0.1582],
    [520, 0.0633, 0.7100, 0.0782],
    [530, 0.1655, 0.8620, 0.0422],
    [540, 0.2904, 0.9540, 0.0203],
    [550, 0.4334, 0.9950, 0.0087],
    [560, 0.5945, 0.9950, 0.0039],
    [570, 0.7621, 0.9520, 0.0021],
    [580, 0.9163, 0.8700, 0.0017],
    [590, 1.0263, 0.7570, 0.0011],
    [600, 1.0622, 0.6310, 0.0008],
    [610, 1.0026, 0.5030, 0.0003],
    [620, 0.8544, 0.3810, 0.0002],
    [630, 0.6424, 0.2650, 0.0000],
    [640, 0.4479, 0.1750, 0.0000],
    [650, 0.2835, 0.1070, 0.0000],
    [660, 0.1649, 0.0610, 0.0000],
    [670, 0.0874, 0.0320, 0.0000],
    [680, 0.0468, 0.0170, 0.0000],
    [690, 0.0227, 0.0082, 0.0000],
    [700, 0.0114, 0.0041, 0.0000]
])

def get_cie_cmf():
    """
    Returns an interpolation function for the CIE color matching functions.
    """
    return interp1d(CIE_CMF_DATA[:, 0], CIE_CMF_DATA[:, 1:], axis=0, bounds_error=False, fill_value=0)

def spectrum_to_xyz(wavelengths, spectrum_data, cie_cmf_func):
    """
    Converts a spectrum to an XYZ tristimulus value.
    """
    # Ensure spectrum_data is a 1D array
    if spectrum_data.ndim > 1:
        spectrum_data = spectrum_data.flatten()
        
    cmf = cie_cmf_func(wavelengths)
    
    xyz = np.trapz(spectrum_data[:, np.newaxis] * cmf, wavelengths, axis=0)
    
    # A rough normalization
    xyz /= 100
    
    return xyz

def xyz_to_rgb(xyz):
    """
    Converts from XYZ to sRGB.
    """
    # XYZ to linear sRGB conversion matrix
    M = np.array([
        [ 3.2404542, -1.5371385, -0.4985314],
        [-0.9692660,  1.8760108,  0.0415560],
        [ 0.0556434, -0.2040259,  1.0572252]
    ])
    
    rgb_linear = np.dot(M, xyz)
    
    # Gamma correction
    rgb = np.where(rgb_linear <= 0.0031308,
                   12.92 * rgb_linear,
                   1.055 * (rgb_linear**(1/2.4)) - 0.055)
    
    # Clip to [0, 1] and scale to [0, 255]
    return np.clip(rgb, 0, 1) * 255
