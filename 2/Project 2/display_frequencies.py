import math
import numpy as np
import matplotlib.pyplot as plt
import skimage.transform as sktr
import skimage as sk
import skimage.io as skio
from filters import *

def disp_freq(imname):
    input_path = f'{directory}/{imname}'
    # output_path = f'spline/aligned_{imname}'
    im = (skio.imread(input_path))
    gray_image = sk.color.rgb2gray(im)

    freq = np.log(np.abs(np.fft.fftshift(np.fft.fft2(gray_image))))
    display_save_image(freq, f'harrump frequencies/{imname}')

# directory = 'osktree'
# for i in ['aligned_oski.jpg', 'low_oski.jpg', 'aligned_tree.jpg', 'high_tree.jpg', 'osktree.jpg']:
#     disp_freq(i)
#
# directory = 'catman'
# for i in ['aligned_cat.jpg', 'high_cat.jpg', 'aligned_man.jpg', 'low_man.jpg', 'catman.jpg']:
#     disp_freq(i)

directory = 'harrump'
for i in ['aligned_trump.jpg', 'high_trump.jpg', 'aligned_kamala.jpg', 'low_kamala.jpg', 'harrump.jpg']:
    disp_freq(i)