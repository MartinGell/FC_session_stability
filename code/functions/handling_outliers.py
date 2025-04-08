
import numpy as np
from scipy.special import erfcinv

def isthisanoutlier(a, dim=None, threshold_factor=3):
    """
    ISOUTLIER Find outliers in data
    Translated from MATLAB function isthisanoutlier.m
    
    TF = ISOUTLIER(A) returns a boolean array whose elements are True when 
    an outlier is detected in the corresponding element. An outlier is an 
    element that is greater than 3 scaled median absolute deviation (MAD) away 
    from the median. The scaled MAD is defined as K*MEDIAN(ABS(A-MEDIAN(A))) 
    where K is the scaling factor and is approximately 1.4826. If A is a 
    matrix, ISOUTLIER operates on each column separately. If A is
    an N-D array, ISOUTLIER operates along the first array dimension
    whose size does not equal 1.

    # Example usage:
    a = np.random.randn(100)
    tf, lthresh, uthresh, center = is_this_an_outlier(a)
    print("Outliers:", outliers)
    print("Lower Threshold:", lthresh)
    print("Upper Threshold:", uthresh)
    print("Center (Median):", center)
    """

    a = np.asarray(a)
    
    if dim is None:
        dim = next((i for i, s in enumerate(a.shape) if s != 1), 0)
    
    madfactor = -1 / (np.sqrt(2) * erfcinv(3/2))  # ~1.4826
    center = np.nanmedian(a, axis=dim)
    mad = madfactor * np.nanmedian(np.abs(a - np.expand_dims(center, axis=dim)), axis=dim)

    lthresh = center - threshold_factor * mad
    uthresh = center + threshold_factor * mad
    
    tf = (a < np.expand_dims(lthresh, axis=dim)) | (a > np.expand_dims(uthresh, axis=dim))
    
    return tf, lthresh, uthresh, center

