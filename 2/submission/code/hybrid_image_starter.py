import matplotlib.pyplot as plt
import numpy as np
from align_image_code import align_images
from main import gaussian_filter, convolve, normalize, display_save_image
import skimage.io as skio
import skimage as sk


directory = 'harrump'

im1_aligned = plt.imread(f'{directory}/aligned_trump.jpg')/255.

im2_aligned = plt.imread(f'{directory}/aligned_kamala.jpg')/255.

im1_aligned, im2_aligned = align_images(im1_aligned, im2_aligned)
display_save_image(im1_aligned, f'{directory}/aligned_trump.jpg')
display_save_image(im2_aligned, f'{directory}/aligned_kamala.jpg')

def hybrid_image(im1, im2, k1, k2, out1, out2, out3):

    im1_filter = gaussian_filter(k1)
    im2_filter = gaussian_filter(k2)

    im1_high_freq = im1 - convolve(im1, im1_filter)
    display_save_image(im1_high_freq, out1)

    im2_low_freq = convolve(im2, im2_filter)
    display_save_image(im2_low_freq, out2)

    combined = im1_high_freq + im2_low_freq
    display_save_image(combined, out3)


out1 = f'{directory}/high_trump.jpg'
out2 = f'{directory}/low_kamala.jpg'
out3 = f'{directory}/harrump.jpg'
hybrid = hybrid_image(im1_aligned, im2_aligned, 65, 55, out1, out2, out3)


# best cat man: k1: 85, k2: 85
# best osktree: k1: 45, k2: 35
# best harrump: k1: 65, k2: 55