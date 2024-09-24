from filters import *


def display_stack(stack):
    num_levels = len(stack)
    num_images_to_display = 6

    # Calculate evenly spaced indices
    indices = np.linspace(0, num_levels - 1, num=num_images_to_display, dtype=int)

    plt.figure(figsize=(20, 5))  # Adjust the figsize for wider display

    for i, idx in enumerate(indices):
        plt.subplot(1, num_images_to_display, i + 1)  # 1 row, num_images_to_display columns
        plt.imshow(normalize(stack[idx]))  # Normalize to display properly
        plt.title(f'{idx + 1}')  # Label each image with the layer number
        plt.axis('off')  # Hide axis for clean display

    plt.tight_layout(pad=1.0)  # Add padding between plots to avoid overlap
    plt.show()

import matplotlib.pyplot as plt
import numpy as np

def display_three_stacks(stack1, stack2, stack3):
    num_levels = len(stack1)
    num_images_to_display = 6

    # Calculate evenly spaced indices
    indices = np.linspace(0, num_levels - 1, num=num_images_to_display, dtype=int)

    # Combine all stacks into a list for easy access in the nested loop
    stacks = [stack1, stack2, stack3]

    # Adjust the figure size (height controls the vertical space between rows)
    fig, axes = plt.subplots(num_images_to_display, 4, figsize=(15, num_images_to_display * 2))  # Add an extra column for labels

    for i, idx in enumerate(indices):
        # Add the label on the left of each row in the first column
        axes[i, 0].text(0.5, 0.5, f'{idx + 1}', fontsize=12, ha='center', va='center')
        axes[i, 0].axis('off')  # Hide the axis for the label column

        for j, stack in enumerate(stacks):
            # Normalize and display the image from each stack in the appropriate column
            axes[i, j + 1].imshow(normalize(stack[idx]))  # Normalize the image
            axes[i, j + 1].axis('off')  # Hide the axes for the images

    plt.tight_layout(pad=0.3)  # Reduce space between plots
    plt.show()



def gaussian_stack(image, num_levels, ksize):
    g_stack = [image]
    blurred = image
    for i in range(1, num_levels):
        blurred = gaussian(blurred, ksize)  # Increase the blur level
        g_stack.append(blurred)
    return np.array(g_stack)


def laplacian_stack(g_stack):
    l_stack = []
    for i in range(len(g_stack) - 1):
        laplacian = g_stack[i] - g_stack[i + 1]
        l_stack.append(laplacian)
    l_stack.append(g_stack[-1])
    return np.array(l_stack)


def half_and_half_mask(image, up=False):
    mask = image.copy()
    if not up:
        mask[:, :image.shape[1] // 2] = 1
        mask[:, image.shape[1] // 2:] = 0
    else:
        mask[:image.shape[1] // 2, :] = 1
        mask[image.shape[1] // 2:, :] = 0
    return np.array(mask)


def mask_stack(im_stack, mask_stack):
    masked_stack = []
    for i, j in zip(im_stack, mask_stack):
        masked_stack.append(i * j)
    return np.array(masked_stack)


def oraple_stack(im1, im2, ksize, num_levels, out_path, mask=None):
    im1_g_stack = gaussian_stack(im1, num_levels, ksize)
    # display_stack(im1_g_stack)

    im1_l_stack = laplacian_stack(im1_g_stack)

    im2_g_stack = gaussian_stack(im2, num_levels, ksize)
    # display_stack(im2_g_stack)

    im2_l_stack = laplacian_stack(im2_g_stack)

    if mask is None:
        mask = half_and_half_mask(im1)
    mask_g_stack = gaussian_stack(mask, num_levels, ksize)
    # display_stack(mask_g_stack)

    display_three_stacks(im1_g_stack, im2_g_stack, mask_g_stack)

    masked_im1 = mask_stack(im1_l_stack, mask_g_stack)
    display_stack(masked_im1)
    masked_im2 = mask_stack(im2_l_stack, 1 - mask_g_stack)
    display_stack(masked_im2)
    masked_combined = [im1 + im2 for im1, im2 in zip(masked_im1, masked_im2)]
    display_stack(masked_combined)
    display_three_stacks(masked_im1, masked_im2, masked_combined)

    oraple = np.sum(masked_combined, axis=0)
    display_save_image(oraple, out_path)


if __name__ == '__main__':
    imname = 'apple.jpeg'
    input_path = f'oraple/{imname}'
    apple = sk.img_as_float(skio.imread(input_path))

    mask = half_and_half_mask(apple)
    display_save_image(mask, '../media/oraple/just_mask.jpg')

    imname = 'orange.jpeg'
    input_path = f'oraple/{imname}'
    orange = sk.img_as_float(skio.imread(input_path))

    oraple_stack(apple, orange, 11, 300, 'out/oraple/oraple.jpg')
    #
    # # oraple_stack.png: ksize: 25, levels: 8
    # # oraple_stack2.png: ksize: 35, levels: 8
    # # oraple_stack3.png + oraple.png: ksize: 25 + 5 each level, levels: 10
    #
    # # oraple_stack6.png + oraple6.png: ksize: 11, levels: 200
    # # final: ksize: 11, levels: 300


    # imname = 'aligned_moon.jpg'
    # input_path = f'moorth/{imname}'
    # moon = sk.img_as_float(skio.imread(input_path))
    # moon = sk.transform.rescale(moon, 0.8, anti_aliasing=True, channel_axis=-1)
    #
    # imname = 'aligned_earth.jpg'
    # input_path = f'moorth/{imname}'
    # earth = sk.img_as_float(skio.imread(input_path))
    # earth = sk.transform.rescale(earth, 0.8, anti_aliasing=True, channel_axis=-1)
    #
    # oraple_stack(earth, moon, 11, 300, 'out/moorth/moorth.jpg')



    imname = 'aligned_wiktor.jpg'
    input_path = f'wikshi/{imname}'
    wiktor = sk.img_as_float(skio.imread(input_path))

    imname = 'aligned_me.jpg'
    input_path = f'wikshi/{imname}'
    me = sk.img_as_float(skio.imread(input_path))

    imname = 'wikshi_mask.png'
    input_path = f'wikshi/{imname}'
    mask = sk.img_as_float(skio.imread(input_path))

    oraple_stack(wiktor, me, 11, 50, 'out/wikshi/wikshi2.jpg', mask)

