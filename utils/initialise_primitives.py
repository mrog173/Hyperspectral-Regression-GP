from .primitives import *
import random
import math
from deap import gp

#------------------------------
#    Initialise primitives
#------------------------------
def image_based(num_wavelengths):
    """Initialise the primitive set.
    num_wavelengths : the number of wavelengths/channels in the data
    """
    primitive_set = gp.PrimitiveSetTyped("MAIN", [HyperspectralImg], FeatureVector, "IN")

    # Interval selection
    # - Image
    primitive_set.addPrimitive(img_interval_selection, [HyperspectralImg, WavelengthBand, WindowWidth], GreyscaleImg)
    primitive_set.addPrimitive(img_interval_selection_gaussian, [HyperspectralImg, WavelengthBand, WindowWidth, Sigma], GreyscaleImg)
    primitive_set.addPrimitive(img_interval_selection_median, [HyperspectralImg, WavelengthBand, WindowWidth], GreyscaleImg)
    # - Spectra
    primitive_set.addPrimitive(mean_interval_selection, [ReflectanceSpectra, WavelengthBand, WindowWidth], SpectralFeature)
    primitive_set.addPrimitive(median_interval_selection, [ReflectanceSpectra, WavelengthBand, WindowWidth], SpectralFeature)

    # Image filtering and normalisation
    primitive_set.addPrimitive(gaussian_filter, [GreyscaleImg, KernelSize, Sigma], GreyscaleImg)
    primitive_set.addPrimitive(derivative_of_gaussian_x, [GreyscaleImg, Sigma], GreyscaleImg)
    primitive_set.addPrimitive(derivative_of_gaussian_y, [GreyscaleImg, Sigma], GreyscaleImg)
    primitive_set.addPrimitive(median_filter, [GreyscaleImg, KernelSize], GreyscaleImg)
    primitive_set.addPrimitive(gabor_7x7, [GreyscaleImg, GaborFreq, GaborTheta], GreyscaleImg)
    primitive_set.addPrimitive(gabor_9x9, [GreyscaleImg, GaborFreq, GaborTheta], GreyscaleImg)
    primitive_set.addPrimitive(max_filter, [GreyscaleImg, KernelSize], GreyscaleImg)
    primitive_set.addPrimitive(min_filter, [GreyscaleImg, KernelSize], GreyscaleImg)
    primitive_set.addPrimitive(minmax_scaling, [GreyscaleImg], GreyscaleImg)
    
    # Spectral combination
    primitive_set.addPrimitive(addition, [SpectralFeature, SpectralFeature], SpectralFeature)
    primitive_set.addPrimitive(subtraction, [SpectralFeature, SpectralFeature], SpectralFeature)
    primitive_set.addPrimitive(multiplication, [SpectralFeature, SpectralFeature], SpectralFeature)
    primitive_set.addPrimitive(division, [SpectralFeature, SpectralFeature], SpectralFeature)

    # Spectral preprocessing
    primitive_set.addPrimitive(savitzky_golay, [ReflectanceSpectra], ReflectanceSpectra)
    primitive_set.addPrimitive(standard_normal_variate, [ReflectanceSpectra], ReflectanceSpectra)

    # Aggregation
    primitive_set.addPrimitive(spectral_aggregation, [GreyscaleImg], SpectralFeature)
    primitive_set.addPrimitive(mean_spectra_extraction, [HyperspectralImg], ReflectanceSpectra)

    # Image features
    primitive_set.addPrimitive(glcm_contrast, [GreyscaleImg, GLCMDistance, GLCMTheta], SpectralFeature)
    primitive_set.addPrimitive(glcm_correlation, [GreyscaleImg, GLCMDistance, GLCMTheta], SpectralFeature)
    primitive_set.addPrimitive(glcm_energy, [GreyscaleImg, GLCMDistance, GLCMTheta], SpectralFeature)
    primitive_set.addPrimitive(glcm_homogeneity, [GreyscaleImg, GLCMDistance, GLCMTheta], SpectralFeature)
    primitive_set.addPrimitive(glcm_entropy, [GreyscaleImg, GLCMDistance, GLCMTheta], SpectralFeature)
    primitive_set.addPrimitive(mean_feature, [GreyscaleImg], SpectralFeature)
    primitive_set.addPrimitive(std_feature, [GreyscaleImg], SpectralFeature)
    primitive_set.addPrimitive(skewness_feature, [GreyscaleImg], SpectralFeature)
    primitive_set.addPrimitive(kurtosis_feature, [GreyscaleImg], SpectralFeature)

    # Convert from spectral feature to feature vector: duplicated to balance selection probability
    primitive_set.addPrimitive(convert_to_feature, [SpectralFeature], FeatureVector)

    # Feature concatenation
    primitive_set.addPrimitive(root2, [FeatureVector, FeatureVector], FeatureVector)
    primitive_set.addPrimitive(root3, [FeatureVector, FeatureVector, FeatureVector], FeatureVector)
    primitive_set.addPrimitive(root4, [FeatureVector, FeatureVector, FeatureVector, FeatureVector], FeatureVector)
    
    # Terminals
    primitive_set.addEphemeralConstant("WavelengthBand", lambda: random.randint(0, num_wavelengths-1), WavelengthBand)
    primitive_set.addEphemeralConstant("WindowWidth", lambda: random.choice([1,3,5,7,9,11,13]), WindowWidth)
    primitive_set.addEphemeralConstant("Sigma", lambda: 4.9*(1-random.random())+0.1, Sigma)
    primitive_set.addEphemeralConstant("KernelSize", lambda:random.choice([3,5,7,9,11]), KernelSize)
    primitive_set.addEphemeralConstant("GaborTheta", lambda: random.randint(0, 7 )* math.pi/8, GaborTheta)
    primitive_set.addEphemeralConstant("GaborFreq", lambda: math.pi/(2*(2**random.randint(0, 3)**0.5)), GaborFreq)
    primitive_set.addEphemeralConstant("GLCMTheta", lambda: random.randint(0, 3) * math.pi/4, GLCMTheta)
    primitive_set.addEphemeralConstant("GLCMDistance", lambda: random.randint(1, 5), GLCMDistance)

    return primitive_set

def spectra_based(num_wavelengths):
    """Initialise the primitive set.
    num_wavelengths : the number of wavelengths/channels in the data
    """
    primitive_set = gp.PrimitiveSetTyped("MAIN", [ReflectanceSpectra], FeatureVector, "IN")

    # Interval selection
    primitive_set.addPrimitive(mean_interval_selection, [ReflectanceSpectra, WavelengthBand, WindowWidth], SpectralFeature)
    primitive_set.addPrimitive(median_interval_selection, [ReflectanceSpectra, WavelengthBand, WindowWidth], SpectralFeature)

    # Spectral combination
    primitive_set.addPrimitive(addition, [SpectralFeature, SpectralFeature], SpectralFeature)
    primitive_set.addPrimitive(subtraction, [SpectralFeature, SpectralFeature], SpectralFeature)
    primitive_set.addPrimitive(multiplication, [SpectralFeature, SpectralFeature], SpectralFeature)
    primitive_set.addPrimitive(division, [SpectralFeature, SpectralFeature], SpectralFeature)

    # Convert from spectral feature to feature vector: duplicated to balance selection probability
    primitive_set.addPrimitive(convert_to_feature, [SpectralFeature], FeatureVector)

    # Feature concatenation
    primitive_set.addPrimitive(root2, [FeatureVector, FeatureVector], FeatureVector)
    primitive_set.addPrimitive(root3, [FeatureVector, FeatureVector, FeatureVector], FeatureVector)
    primitive_set.addPrimitive(root4, [FeatureVector, FeatureVector, FeatureVector, FeatureVector], FeatureVector)
    
    # Terminals
    primitive_set.addEphemeralConstant("WavelengthBand", lambda: random.randint(0, num_wavelengths-1), WavelengthBand)
    primitive_set.addEphemeralConstant("WindowWidth", lambda: random.choice([1,3,5,7,9,11,13]), WindowWidth)

    return primitive_set