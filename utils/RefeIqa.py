#%%
import numpy as np
import math
from scipy.signal import convolve2d
from numpy.core.fromnumeric import transpose

#%%
def to255(X):
    return (((X-X.min())/(X.max()-X.min()))*255).astype(np.int16)

def RMSE(x, y):
    """
    RMSE on single spectral
    """
    return np.sqrt(np.mean(np.power(x - y, 2)))

def SAM(x, y):
    """
    SAM all spectral
    x : CWH
    y : CWH
    """
    def sam1(src, dst):
        """
        numpy array : src, dst
        """
        val = np.dot(src, dst)/(np.linalg.norm(src)*np.linalg.norm(dst))
        sam = np.arccos(val)
        return sam

    def sam2(src, dst):
        """
        numpy array : src, dst
        """
        val = np.dot(src, dst)/(np.mean(np.power(src, 2))*np.mean(np.power(dst, 2)))
        sam = np.arccos(val)
        return sam

    Sam = 0
    for i in range(x.shape[1]):
        for j in range(x.shape[2]):
            Sam = Sam + sam1(x[:,i,j], y[:,i,j])

    return Sam/(x.shape[1] * x.shape[2])

def SRE(x, y):
    """
    SRE on single spectral
    """
    sre = 10*np.log10(
        np.sum(np.power(x, 2))/          \
        np.sum(np.power(y-x,2))
    )
    return sre    

def UIQA(x, y):
    """
    UIQA on single spectral
    https://github.com/tgandor/urban_oculus/blob/4fb32138641a276e77b61acafaf0de77caa0cf22/metrics/image_quality_index.py
    """
    def universal_image_quality_index_conv(x, y, kernelsize=8):
        """Compute the Universal Image Quality Index (UIQI) of x and y.
        Not normalized with epsilon, and using scipy.signal.convolve2d."""

        N = kernelsize ** 2
        kernel = np.ones((kernelsize, kernelsize))

        x = x.astype(np.float)
        y = y.astype(np.float)

        # sums and auxiliary expressions based on sums
        S_x = convolve2d(x, kernel, mode='valid')
        S_y = convolve2d(y, kernel, mode='valid')
        PS_xy = S_x * S_y
        SSS_xy = S_x*S_x + S_y*S_y

        # sums of squares and product
        S_xx = convolve2d(x*x, kernel, mode='valid')
        S_yy = convolve2d(y*y, kernel, mode='valid')
        S_xy = convolve2d(x*y, kernel, mode='valid')

        Q_s = 4 * PS_xy * (N * S_xy - PS_xy) / (N*(S_xx + S_yy) - SSS_xy) / SSS_xy

        return np.mean(Q_s)

    return universal_image_quality_index_conv(x, y)

