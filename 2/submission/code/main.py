import scipy.signal
import numpy as np
import skimage as sk
import skimage.io as skio
import cv2
import matplotlib.pyplot as plt


def normalize(img):
    img = img - np.min(img)
    return img / np.max(img)

def convolve(image, filter):
    ch1 = scipy.signal.convolve2d(image[:, :, 0], filter, mode='same', boundary='symm')
    ch2 = scipy.signal.convolve2d(image[:, :, 1], filter, mode='same', boundary='symm')
    ch3 = scipy.signal.convolve2d(image[:, :, 2], filter, mode='same', boundary='symm')
    return np.dstack([ch1, ch2, ch3])

def get_best_ksize(image):
    im_size = np.average(image.shape[0:2])
    approx = int(im_size/100)
    if approx % 2 == 0:
        approx += 1
    return approx

def finite_diff_x(image):
    return convolve(image, [[1, -1]]) + 0.5

def finite_diff_y(image):
    return convolve(image, [[1], [-1]]) + 0.5

def gradient(image):
    diff_x = convolve(image, [[1, -1]])
    diff_y = convolve(image, [[1], [-1]])
    return np.sqrt(diff_x**2 + diff_y**2)

def binarize(image, threshold):
    return (image >= threshold).astype(np.uint8)

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

def DoG_2_step(image, ksize):
    blurred = gaussian(image, ksize)
    dog_diff_x = finite_diff_x(blurred)
    dog_diff_y = finite_diff_y(blurred)
    return dog_diff_x, dog_diff_y


def DoG_1_step(image, ksize):
    filter = gaussian_filter(ksize)
    gauss_diff_x = scipy.signal.convolve2d(filter, [[1, -1]], mode='same', boundary='symm')
    display_save_image(gauss_diff_x, '../media/cameraman/gaussian_diff_x.png')
    gauss_diff_y = scipy.signal.convolve2d(filter, [[1], [-1]], mode='same', boundary='symm')
    display_save_image(gauss_diff_y, '../media/cameraman/gaussian_diff_y.png')

    dog_diff_x = convolve(image, gauss_diff_x) + 0.5
    dog_diff_y = convolve(image, gauss_diff_y) + 0.5
    return dog_diff_x, dog_diff_y

def sharpen_2_step(image, ksize, alpha):
    low_freq = gaussian(image.copy(), ksize)
    high_freq = image - low_freq
    return np.clip(image + (alpha * high_freq), 0, 1)

def sharpen_1_step(image, ksize, alpha):
    filter = laplacian_filter(ksize, alpha)
    return np.clip(convolve(image, filter), 0, 1)

def blur_sharpen(image, ksize, alpha):
    blurred = gaussian(image, ksize)
    sharpened = sharpen_2_step(blurred, ksize, alpha)
    return sharpened

def display_save_image(image, fname=None):
    image = normalize(image)
    image = (255 * image).astype(np.uint8)
    if fname:
        skio.imsave(fname, image)
    skio.imshow(image)
    skio.show()


if __name__ == "__main__":

    imname = 'cameraman.png'
    input_path = f'cameraman/{imname}'
    im = sk.img_as_float(skio.imread(input_path))
    im_out = np.dstack([im[:,:,0], im[:,:,1], im[:,:,2]])

    dx = [[1, -1]]
    im_diff_x = finite_diff_x(im)
    display_save_image(im_diff_x, f'out/dx_{imname}')

    dy = [[1], [-1]]
    im_diff_y = finite_diff_y(im)
    display_save_image(im_diff_y, f'out/dy_{imname}')

    im_grad = gradient(im)
    im_grad_bin = binarize(im_grad, 0.3)

    display_save_image(im_grad_bin, f'cameraman/gradient_bin_{imname}')


    im_gauss = gaussian(im, 13)
    display_save_image(im_gauss, f'out/gaussian_{imname}')

    im_dog_x_2, im_dog_y_2 = DoG_2_step(im, 13)
    display_save_image(im_dog_x_2, f'out/DoG_x_2_conv_{imname}')
    display_save_image(im_dog_y_2, f'out/DoG_y_2_conv_{imname}')

    im_dog_x_1, im_dog_y_1 = DoG_1_step(im, 13)
    display_save_image(im_dog_x_1, f'out/DoG_x_1_conv_{imname}')
    display_save_image(im_dog_y_1, f'out/DoG_y_1_conv_{imname}')
