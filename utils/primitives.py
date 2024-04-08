import numpy as np
from scipy.signal import medfilt2d, savgol_filter
from scipy.stats import kurtosis, skew
from scipy import ndimage
import scipy as sp
import cv2

sp.special.seterr(loss='ignore')
#----------------------------------------------------
#  Define special datatypes for restricting GP-tree
#----------------------------------------------------
"""Hypercube"""
class HyperspectralImg(np.ndarray):
    #numpy ndarray object with precomputed mask and reflectance spectra attributes
    def __new__(cls, input_array, mask, spectra):
        obj = np.asarray(input_array).view(cls)
        obj.mask = mask
        obj.spectra = ReflectanceSpectra(spectra)
        return obj
    def __array_finalize__(self, obj):
        if obj is None: return
        self.mask = getattr(obj, 'mask', None)
        self.spectra = getattr(obj, 'spectra', None)

"""Greyscale image"""
class GreyscaleImg(np.ndarray):
    #numpy ndarray object with precomputed mask attribute
    def __new__(cls, input_array, mask):
        obj = np.asarray(input_array).view(cls)
        obj.mask = mask
        return obj
    def __array_finalize__(self, obj):
        if obj is None: return
        self.mask = getattr(obj, 'mask', None)

"""Reflectance spectra"""
class ReflectanceSpectra(np.ndarray):
    # numpy ndarray object
    def __new__(cls, spectra):
        obj = np.asarray(spectra).view(cls)
        return obj

"""Spectral feature"""
class SpectralFeature(float):
    # floating point number object
    def __new__(self, value):
        return float.__new__(self, value)

"""Feature vector"""
class FeatureVector(np.ndarray):
    # numpy ndarray object
    def __new__(cls, feature_vec):
        obj = np.asarray(feature_vec).view(cls)
        return obj

#----------------------------
#  Define terminal datatypes
#----------------------------

"""Sigma"""
class Sigma(float):
    def __init__(float):
        pass

"""Orientation of Gabor filter"""
class GaborTheta(float):
    def __init__(float):
        pass

"""Frequency of Gabor filter"""
class GaborFreq(float):
    def __init__(float):
        pass

"""GLCM matrix angle"""
class GLCMTheta(float):
    def __init__(float):
        pass

"""GLCM matrix distance"""
class GLCMDistance(int):
    def __init__(int):
        pass

"""Central wavelength band"""
class WavelengthBand(int):
    def __init__(int):
        pass

"""Window width"""
class WindowWidth(int):
    def __init__(int):
        pass

"""Kernel size"""
class KernelSize(int):
    def __init__(int):
        pass

#----------------------------
#    Define function set
#----------------------------
def img_interval_selection(img, w, k):
    """Takes the average of the k wavelengths centered on band w. When part of the window lies outside of [0, num_wavelengths-1], 
    then only the wavelengths in the window within this range are averaged.
    Returns: single channel GrayscaleImg
    """
    im = np.average(img[:,:,max(w-k//2,0):min(w+k//2+1,img.shape[2])],axis=2)
    mask = img.mask
    return GreyscaleImg(im, mask)

def img_interval_selection_gaussian(img, w, k, sigma):
    """Takes the 1D gaussian weighting of the k wavelengths centered on band w. When part of the window lies outside of [0, num_wavelengths-1], 
    then the components of the gaussian kernel corresponding to bands within the image rescaled to sum to 1 and the weighting is applied.
    Returns: single channel GrayscaleImg
    """
    r = range(-int(k/2),int(k/2)+1)
    kernel = np.asarray([1 / (sigma * np.sqrt(2*np.pi)) * np.exp(-float(x)**2/(2*sigma**2)) for x in r])
    kernel = kernel[max(0, -1*(w-k//2)):min(2*(k//2)+1, (img.shape[2]-w)+k//2)]

    kernel = kernel/np.sum(kernel)
    im = np.average(img[:,:,max(w-k//2,0):min(w+k//2+1,img.shape[2])], weights=kernel, axis=2).astype(np.float32)
    mask = img.mask
    return GreyscaleImg(im, mask)

def img_interval_selection_median(img, w, k):
    """Takes the median of the k wavelengths centered on band w. When part of the window lies outside of [0, num_wavelengths-1], 
    then the median is taken of the wavelengths in the window within this range.
    Returns: single channel GrayscaleImg
    """
    im = np.median(img[:,:,max(w-k//2,0):min(w+k//2+1,img.shape[2])],axis=2)
    mask = img.mask
    return GreyscaleImg(im, mask)

def mean_interval_selection(spectra, w, k):
    """Takes the average (mean) of the k mean spectral values centered on wavelength w. 
    When part of the window lies outside of the range [0, num_wavelengths-1] then the 
    average is taken of the wavelengths in this window within the range.
    Returns: floating point SpectralFeature.
    """
    return np.average(spectra[max(w-k//2,0):min(w+k//2+1, spectra.shape[0])])

def median_interval_selection(spectra, w, k):
    """Takes the average (median) of the k mean spectral values centered on wavelength w.
    When part of the window lies outside of the range [0, num_wavelengths-1] then the 
    average is taken of the wavelengths in this window within the range.
    Returns: floating point SpectralFeature.
    """
    return np.median(spectra[max(w-k//2,0):min(w+k//2+1, spectra.shape[0])])

def savitzky_golay(spectra):
    """Applies cubic savitzky golay smoothing to the reflectance spectra using a window 
    length of 11.
    Returns: a ReflectanceSpectra.
    """
    return savgol_filter(np.asarray(spectra), window_length=11, polyorder=3)

def standard_normal_variate(spectra):
    """Applies standard normal variate normalisation to the reflectance spectra.
    Returns: a ReflectanceSpectra.
    """
    return (spectra - np.average(spectra))/np.std(spectra)

def gaussian_filter(img, k, sigma):
    """Applies a Gaussian blur to the GrayscaleImg with a kernel size (k,k) with a given sigma.
    Returns: single channel GrayscaleImg
    """
    im = cv2.GaussianBlur(img.base, (k,k), sigma)
    mask = img.mask
    return GreyscaleImg(im, mask)

def derivative_of_gaussian_x(img, sigma):
    res = ndimage.gaussian_filter(img, sigma=sigma, order=[0,1])
    return GreyscaleImg(res, img.mask)

def derivative_of_gaussian_y(img, sigma):
    res = ndimage.gaussian_filter(img, sigma=sigma, order=[1,0])
    return GreyscaleImg(res, img.mask)

def median_filter(img, k):
    """Applies a median blur to the GrayscaleImg with a kernel size (k,k).
    Implemented using the SciPy implementation because of OpenCV typing constraints.
    Returns: single channel GrayscaleImg
    """
    im = medfilt2d(img.base, k).astype(np.float32)
    mask = img.mask
    return GreyscaleImg(im, mask)

def gabor_7x7(img, freq, theta):
    """Applies a 7x7 gabor filter with a lambda = freq and orientation = theta. 
    Default values psi (phase offset) = 1, gamma (aspect ratio) = 0.3, sigma (variance) = 2.
    Returns: single channel GrayscaleImg
    """
    gb = cv2.getGaborKernel((7,7), sigma=2, theta=theta, lambd=freq, gamma=0.3, psi=1)
    im = cv2.filter2D(img.base.astype(np.single), cv2.CV_32F, gb)
    #im = (im-np.min(im))/(np.max(im)-np.min(im))
    mask = img.mask
    return GreyscaleImg(im, mask)

def gabor_9x9(img, freq, theta):
    """Applies a 9x9 gabor filter with a lambda = freq and orientation = theta. 
    Default values psi (phase offset) = 1, gamma (aspect ratio) = 0.3, sigma (variance) = 2.
    Returns: single channel GrayscaleImg
    """
    gb = cv2.getGaborKernel((9,9), sigma=2, theta=theta, lambd=freq, gamma=0.3, psi=1)
    im = cv2.filter2D(img.base.astype(np.single), cv2.CV_32F, gb)
    #im = (im-np.min(im))/(np.max(im)-np.min(im))
    mask = img.mask
    return GreyscaleImg(im, mask)

def max_filter(img, k):
    """Applies a maximum filter (dilation) to the GrayscaleImg with a kernel size (k,k).
    Returns: single channel GrayscaleImg
    """
    im = cv2.dilate(img.base, np.ones((k,k)))
    mask = img.mask
    return GreyscaleImg(im, mask)

def min_filter(img, k):
    """Applies a minimum filter (erosion) to the GrayscaleImg with a kernel size (k,k).
    Returns: single channel GrayscaleImg
    """
    im = cv2.erode(img.base, np.ones((k,k)))
    mask = img.mask
    return GreyscaleImg(im, mask)

def minmax_scaling(img):
    """Applies min-max (0-1) scaling to the image.
    Returns: single channel GrayscaleImg
    """
    if np.max(img.base)-np.min(img.base) == 0:
        return GreyscaleImg((img.base-np.min(img.base)), img.mask)
    return GreyscaleImg((img.base-np.min(img.base))/(np.max(img.base)-np.min(img.base)), img.mask)

def contrast_limited_adaptive_he(img):
    """Applies contrast limited adaptive histogram equalisation to the image.
    Image is first converted to 16-bit unsigned int and then converted back. Some information may be lost.
    Returns: single channel GrayscaleImg
    """
    im = (65535*img.base).astype(np.uint16)
    clahe = cv2.createCLAHE()
    return GreyscaleImg(clahe.apply(im)/65535, img.mask)

def spectral_aggregation(img):
    """Averages the pixel values within the ROI of a grayscale image.
    Returns: a SpectralFeature.
    """
    return SpectralFeature(np.ma.array(img.base, mask=img.mask).mean())

def mean_spectra_extraction(img):
    """Extracts the mean spectral response over the ROI for the n bands. 
    Returns: a ReflectanceSpectra.
    """
    return img.spectra

def glcm_entropy(img, d, theta):
    if img.max() == img.min():
        return SpectralFeature(1)
    y_off = 0 if theta == 0 else 1
    x_off = 0 if theta == 2 else (-1 if theta == 3 else 1)
    
    # Create GLCM
    glcm = greycomatrix_custom(img.base, img.mask, d*x_off, d*y_off)
    
    # Calculate entropy
    entropy = -np.sum(glcm*np.log2(glcm + (glcm==0)))
    return SpectralFeature(entropy)

def glcm_energy(img, d, theta):
    if img.max() == img.min():
        return SpectralFeature(1)
    y_off = 0 if theta == 0 else 1
    x_off = 0 if theta == 2 else (-1 if theta == 3 else 1)
    
    # Create GLCM
    glcm = greycomatrix_custom(img.base, img.mask, d*x_off, d*y_off)
    
    # Calculate energy
    energy = np.sqrt(np.sum(np.power(glcm,2)))
    return SpectralFeature(energy)

def glcm_correlation(img, d, theta):
    if img.max() == img.min():
        return SpectralFeature(1)
    y_off = 0 if theta == 0 else 1
    x_off = 0 if theta == 2 else (-1 if theta == 3 else 1)
    
    # Create GLCM
    glcm = greycomatrix_custom(img.base, img.mask, d*x_off, d*y_off)
    
    # Calculate correlation
    I = np.array(range(glcm.shape[0])).reshape(1,-1)
    J = np.array(range(glcm.shape[0])).reshape(-1,1)
    diff_i = I - np.sum(I * glcm)
    diff_j = J - np.sum(J * glcm)
    std_i = np.sqrt(np.sum(glcm * (diff_i) ** 2))
    std_j = np.sqrt(np.sum(glcm * (diff_j) ** 2))
    if std_i == 0 or std_j == 0:
        return SpectralFeature(1)
    cov = np.sum(glcm * (diff_i * diff_j))
    return SpectralFeature(cov/(std_i*std_j))

def glcm_contrast(img, d, theta):
    if img.max() == img.min():
        return SpectralFeature(0)
    y_off = 0 if theta == 0 else 1
    x_off = 0 if theta == 2 else (-1 if theta == 3 else 1)
    
    # Create GLCM
    glcm = greycomatrix_custom(img.base, img.mask, d*x_off, d*y_off)
    
    # Calculate contrast
    I, J = np.ogrid[0:glcm.shape[0], 0:glcm.shape[0]]
    weights = (I - J) ** 2
    contrast = np.sum(np.multiply(glcm, weights))
    return SpectralFeature(contrast)

def glcm_homogeneity(img, d, theta):
    if img.max() == img.min():
        return SpectralFeature(1)
    y_off = 0 if theta == 0 else 1
    x_off = 0 if theta == 2 else (-1 if theta == 3 else 1)
    
    # Create GLCM
    glcm = greycomatrix_custom(img.base, img.mask, d*x_off, d*y_off)
    
    # Calculate homogeneity
    I, J = np.ogrid[0:glcm.shape[0], 0:glcm.shape[0]]
    weights = ((I - J) ** 2)+1
    homogeneity = np.sum(np.divide(glcm,weights))
    return SpectralFeature(homogeneity)

def mean_feature(img):
    """Returns: mean of pixels inside ROI. Sum of pixel values/Number of pixels.
    """
    return SpectralFeature(np.ma.array(img.base, mask=img.mask).mean())

def std_feature(img):
    """Returns: sqrt of squared deviations from the mean. sqrt(sum((x[i]-mean(x))**2))
    """
    m = np.mean(img.base[img.mask])
    vals = np.power(img.base[img.mask] - m, 2)
    summed = np.sum(vals)
    return SpectralFeature(np.asarray([np.sqrt(summed)]))

def skewness_feature(img):
    """Returns: Skewness ()
    """
    return SpectralFeature(skew(np.ma.array(img.base, mask=img.mask).flatten()))

def kurtosis_feature(img):
    """Returns: Kurtosis ()
    """
    return SpectralFeature(kurtosis(np.ma.array(img.base.astype(np.float32), mask=img.mask).flatten()))

def addition(spec1, spec2):
    """Addition of two spectral features.
    Returns: floating point SpectralFeature.
    """
    return spec1+spec2

def subtraction(spec1, spec2):
    """Subtraction of two spectral features.
    Returns: floating point SpectralFeature.
    """
    return spec1-spec2

def multiplication(spec1, spec2):
    """Multiplication of two spectral features.
    Returns: floating point SpectralFeature.
    """
    return spec1*spec2

def division(spec1, spec2):
    """Safe division of two spectral features.
    Returns: floating point SpectralFeature.
    """
    if spec2 != 0:
        return spec1/spec2
    else:
        return 0

def convert_to_feature(a):
    """Convert SpectralFeature to 1x1 FeatureVector. Restricts spectral combination 
    functions to only combine spectral features.
    """
    return np.asarray([a])

def root2(a, b):
    """Concatenates two feature vectors into a single feature vector.
    """
    return np.concatenate([np.asarray(a),np.asarray(b)])

def root3(a, b, c):
    """Concatenates three feature vectors into a single feature vector.
    """
    return np.concatenate([np.asarray(a),np.asarray(b), np.asarray(c)])

def root4(a, b, c, d):
    """Concatenates three feature vectors into a single feature vector.
    """
    return np.concatenate([a,b,c,d])

def greycomatrix_custom(image, mask, x_off, y_off):
    # Rescale image to be maximum of 64 values
    image = (64*((image-np.min(image))/(np.max(image)-np.min(image)))).astype(np.uint8)
    # Add one to all pixels and set background to 0
    image = image + 1
    image[~mask.astype(bool)] = 0

    h, w = image.shape
    levels = np.max(image) + 1
    matrix = np.zeros((levels, levels), dtype=np.uint32, order='C')

    # Temporary matrix: Align corresponding pixels
    temp = np.zeros((h+y_off, w+x_off), int)
    temp[:h, :w] = image
    temp = temp[y_off:, x_off:]

    # Create stack of correspondences
    temp = np.vstack([image.flatten(), temp.flatten()]).T

    # Filter correspondences with at least one background entry, and count correspondences
    temp = temp[~np.any(temp == 0, axis=1), :]
    temp -= 1
    temp = np.unique(temp, axis=0, return_counts=True)

    matrix[temp[0][:, 0], temp[0][:, 1]] = temp[1]

    #Normalise GLCM values
    matrix = matrix.astype(np.float64)
    glcm_sums = np.apply_over_axes(np.sum, matrix, axes=(0, 1))
    glcm_sums[glcm_sums == 0] = 1
    matrix /= glcm_sums

    return matrix