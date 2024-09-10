import numpy as np
import skimage as sk
import skimage.io as skio


# Calculate Euclidean distance between two images
def euclidean_distance(im1, im2):
    return np.sqrt(np.sum((im1 - im2) ** 2))

# Calculate Normalized Cross-Correlation (NCC) between two images
def ncc(im1, im2):
    zero_centered_im1 = im1 - np.mean(im1)
    zero_centered_im2 = im2 - np.mean(im2)
    norm_im1 = zero_centered_im1 / np.linalg.norm(zero_centered_im1)
    norm_im2 = zero_centered_im2 / np.linalg.norm(zero_centered_im2)
    return np.sum(norm_im1 * norm_im2)

# Align two images using NCC, searching for the best shift within a range
def align_ncc(im1, im2, shift=20, crop=50):
    max_corr = -np.inf
    best_shift = (0, 0)
    cropped_im2 = im2[crop:-crop, crop:-crop]
    for x in range(-shift, shift + 1):
        for y in range(-shift, shift + 1):
            shifted_im1 = np.roll(im1, shift=(y, x), axis=(0, 1))  # Shift image
            cropped_im1 = shifted_im1[crop:-crop, crop:-crop]  # Crop shifted image
            corr = ncc(cropped_im1, cropped_im2)  # Calculate NCC on cropped images
            if corr > max_corr:
                max_corr = corr
                best_shift = (y, x)
    return best_shift

# Align two images using Euclidean distance, searching for the best shift within a range
def align_euc(im1, im2, shift=20, crop=50):
    min_distance = np.inf
    best_shift = (0, 0)
    cropped_im2 = im2[crop:-crop, crop:-crop]
    for x in range(-shift, shift + 1):
        for y in range(-shift, shift + 1):
            shifted_im1 = np.roll(im1, shift=(y, x), axis=(0, 1))  # Shift image
            cropped_im1 = shifted_im1[crop:-crop, crop:-crop]  # Crop shifted image
            distance = euclidean_distance(cropped_im1, cropped_im2)  # Calculate Euclidean distance
            if distance < min_distance:
                min_distance = distance
                best_shift = (y, x)
    return best_shift

# Pyramid scaling for aligning large images
def pyramid_scaling(im1, im2, min_size):
    if min(im1.shape[0], im1.shape[1]) <= min_size:  # Base case: stop scaling when small enough
        crop = int(max(0.1 * im1.shape[0], 0.1 * im1.shape[1]))
        return align_euc(im1, im2, 50, crop)

    # Rescale images by half
    im1_halved = sk.transform.rescale(im1, 0.5, anti_aliasing=False)
    im2_halved = sk.transform.rescale(im2, 0.5, anti_aliasing=False)

    # Recursive pyramid scaling
    downscaled_shift = pyramid_scaling(im1_halved, im2_halved, min_size)
    upscaled_shift = (2 * downscaled_shift[0], 2 * downscaled_shift[1])  # Double the shift for upscaled image
    im1_shifted = np.roll(im1, shift=upscaled_shift, axis=(0, 1))  # Apply shift to upscaled image
    crop = int(max(0.1 * im1_shifted.shape[0], 0.1 * im1_shifted.shape[1]))
    final_shift = align_euc(im1_shifted, im2, 2, crop)
    total_shift = (final_shift[0] + upscaled_shift[0], final_shift[1] + upscaled_shift[1])

    return total_shift

# Retrieve normalized edge images of each channel
def normalized_edges(arr):
    arr = np.array(sk.filters.prewitt(arr))  # Apply edge detection
    arr = arr.astype(np.float32)
    return (arr - np.min(arr)) / (np.max(arr) - np.min(arr))  # Normalize values between 0 and 1

# Pyramid alignment for RGB channels based on edge detection
def pyramid_align(r, g, b, min_size):
    edge_r = normalized_edges(r)
    edge_g = normalized_edges(g)
    edge_b = normalized_edges(b)

    best_r_shift = pyramid_scaling(edge_r, edge_b, min_size)
    best_r = np.roll(r, shift=best_r_shift, axis=(0, 1))  # Shift the red channel

    best_g_shift = pyramid_scaling(edge_g, edge_b, min_size)
    best_g = np.roll(g, shift=best_g_shift, axis=(0, 1))  # Shift the green channel

    return (best_r, best_r_shift), (best_g, best_g_shift)

# Save aligned image to file
def save_image(image, fname):
    image = (255 * image).astype(np.uint8)  # Convert to 8-bit format
    skio.imsave(fname, image)

# Main function to read, align, and save the image
def main(input_path, output_path):
    im = sk.img_as_float(skio.imread(input_path))  # Read the image
    height = np.floor(im.shape[0] / 3.0).astype(int)  # Divide image into three color channels

    # Split into blue, green, and red channels
    b = im[:height]
    g = im[height: 2 * height]
    r = im[2 * height: 3 * height]

    min_size = 500

    # Align red and green channels to the blue channel
    (ar, r_shift), (ag, g_shift) = pyramid_align(r, g, b, min_size)

    im_out = np.dstack([ar, ag, b])  # Stack the aligned channels
    save_image(im_out, output_path)  # Save the output image

    return r_shift, g_shift


if __name__ == '__main__':

    r_shift, g_shift = main('path/to/input/image','path/to/output/image')  # Run the program
    print(f'R: {r_shift}')  # Print the red shift
    print(f'G: {g_shift}')  # Print the green shift
