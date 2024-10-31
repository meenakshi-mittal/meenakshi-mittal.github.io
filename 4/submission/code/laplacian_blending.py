import scipy.signal
import numpy as np
import skimage as sk
import skimage.io as skio
import cv2
from matplotlib import pyplot as plt

from main import *

def normalize(img):
    img = img - np.min(img)
    return img / np.max(img)

def convolve(image, filter):
    ch1 = scipy.signal.convolve2d(image[:, :, 0], filter, mode='same', boundary='symm')
    ch2 = scipy.signal.convolve2d(image[:, :, 1], filter, mode='same', boundary='symm')
    ch3 = scipy.signal.convolve2d(image[:, :, 2], filter, mode='same', boundary='symm')
    return np.dstack([ch1, ch2, ch3])

def gaussian_filter(ksize):
    one_d = cv2.getGaussianKernel(ksize, ksize / 6)
    two_d = np.outer(one_d, one_d)
    return two_d

def gaussian(image, ksize):
    filter = gaussian_filter(ksize)
    im_dog = convolve(image, filter)
    return im_dog

def laplacian_filter(ksize, alpha):
    gauss = gaussian_filter(ksize)
    impulse = np.zeros((ksize,ksize))
    center = ksize // 2
    impulse[center, center] = 1
    filter = ((1 + alpha) * impulse) - (alpha * gauss)
    return filter


def gaussian_stack(image, num_levels, ksize):
    g_stack = [image]
    blurred = image
    for i in range(1, num_levels):
        blurred = gaussian(blurred, ksize)
        g_stack.append(blurred)
    return np.array(g_stack)


def laplacian_stack(g_stack):
    l_stack = []
    for i in range(len(g_stack) - 1):
        laplacian = g_stack[i] - g_stack[i + 1]
        l_stack.append(laplacian)
    l_stack.append(g_stack[-1])
    return np.array(l_stack)


def mask_stack(im_stack, mask_stack):
    masked_stack = []
    for i, j in zip(im_stack, mask_stack):
        masked_stack.append(i * j)
    return np.array(masked_stack)


def oraple_stack(im1, im2, ksize, num_levels, out_path=None, mask=None):
    im1_g_stack = gaussian_stack(im1, num_levels, ksize)
    im1_l_stack = laplacian_stack(im1_g_stack)

    im2_g_stack = gaussian_stack(im2, num_levels, ksize)
    im2_l_stack = laplacian_stack(im2_g_stack)

    mask_g_stack = gaussian_stack(mask, num_levels, ksize)
    masked_im1 = mask_stack(im1_l_stack, mask_g_stack)
    masked_im2 = mask_stack(im2_l_stack, 1 - mask_g_stack)
    masked_combined = [im1 + im2 for im1, im2 in zip(masked_im1, masked_im2)]

    oraple = np.sum(masked_combined, axis=0)
    return oraple

