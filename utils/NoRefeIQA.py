import cv2 as cv
import numpy as np
import math


def Brenner(img):
    '''
    INPUT -> 2D grayscale image
    OUTPUT -> The clearer the bigger.
    '''
    shape = np.shape(img)
    out = 0
    for x in range(0, shape[0]):
        for y in range(0, shape[1] - 2):
            out += (int(img[x, y + 2]) - int(img[x, y])) ** 2
    return out


def Laplacian(img):
    '''
    INPUT -> 2D grayscale image
    OUTPUT -> The clearer the bigger.
    '''
    return cv.Laplacian(img, cv.CV_64F).var()


def SMD(img):
    '''
    INPUT -> 2D grayscale image
    OUTPUT -> The clearer the bigger.
    '''
    shape = np.shape(img)
    out = 0
    for x in range(1, shape[0]):
        for y in range(0, shape[1] - 1):
            out += math.fabs(int(img[x, y]) - int(img[x - 1, y]))
            out += math.fabs(int(img[x, y] - int(img[x, y + 1])))
    return out


def SMD2(img):
    '''
    INPUT -> 2D grayscale image
    OUTPUT -> The clearer the bigger.
    '''
    shape = np.shape(img)
    out = 0
    for x in range(0, shape[0] - 1):
        for y in range(0, shape[1] - 1):
            out += math.fabs(int(img[x, y]) - int(img[x + 1, y])) * math.fabs(int(img[x, y] - int(img[x, y + 1])))
    return out


def Variance(img):
    '''
    INPUT -> 2D grayscale image
    OUTPUT -> The clearer the bigger.
    '''
    out = 0
    u = np.mean(img)
    shape = np.shape(img)
    for x in range(0, shape[0]):
        for y in range(0, shape[1]):
            out += (img[x, y] - u) ** 2
    return out


def Energy(img):
    '''
    INPUT -> 2D grayscale image
    OUTPUT -> The clearer the bigger.
    '''
    shape = np.shape(img)
    out = 0
    for x in range(0, shape[0] - 1):
        for y in range(0, shape[1] - 1):
            out += ((int(img[x + 1, y]) - int(img[x, y])) ** 2) * ((int(img[x, y + 1] - int(img[x, y]))) ** 2)
    return out


def Vollath(img):
    '''
    INPUT -> 2D grayscale image
    OUTPUT -> The clearer the bigger.
    '''
    shape = np.shape(img)
    u = np.mean(img)
    out = -shape[0] * shape[1] * (u ** 2)
    for x in range(0, shape[0]):
        for y in range(0, shape[1] - 1):
            out += int(img[x, y]) * int(img[x, y + 1])
    return out


def Entropy(img):
    '''
    INPUT -> 2D grayscale image
    OUTPUT -> The clearer the bigger.
    '''
    img = img.astype('int64')
    out = 0
    count = np.shape(img)[0] * np.shape(img)[1]
    p = np.bincount(np.array(img).flatten())
    for i in range(0, len(p)):
        if p[i] != 0:
            out -= p[i] * math.log(p[i] / count) / count
    return out


def numpy2img(NpImg):
    im = np.clip(NpImg, 0, 1)
    im = im * 255
    return im


def EvaAll(img):
    img = numpy2img(img)
    brenner = Brenner(img)
    smd2 = SMD2(img)
    variance = Variance(img)
    energy = Energy(img)
    vollath = Vollath(img)
    entropy = Entropy(img)
    return [brenner, smd2, variance, energy, vollath, entropy]
