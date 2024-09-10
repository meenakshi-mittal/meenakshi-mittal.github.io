import numpy as np
import skimage as sk
import skimage.io as skio

# Normalize the contrast of the image to scale pixel values between 0 and 1
def contrast(array):
    min_val = np.min(array)
    max_val = np.max(array)
    norm_array = (array - min_val) / (max_val - min_val)  # Normalize based on min/max
    return norm_array

# Perform white balancing by adjusting RGB channels to average values
def white_balancing(image, average=True):
    # Calculate the average for each color channel
    avg_r = np.mean(image[:, :, 0])
    avg_g = np.mean(image[:, :, 1])
    avg_b = np.mean(image[:, :, 2])

    if average:
        avg_gray = (avg_r + avg_g + avg_b) / 3.0  # Avg world white balancing
    else:
        avg_gray = 0.5  # Gray world white balancing

    # Calculate scale factors for each channel
    scale_r = avg_gray / avg_r
    scale_g = avg_gray / avg_g
    scale_b = avg_gray / avg_b

    # Apply scaling to each channel
    image[:, :, 0] *= scale_r
    image[:, :, 1] *= scale_g
    image[:, :, 2] *= scale_b

    return np.clip(image, 0, 1)  # Ensure pixel values remain within [0, 1]

# Save image to a file
def save_image(image, fname):
    image = (255 * image).astype(np.uint8)  # Convert to 8-bit format
    skio.imsave(fname, image)

# Main function to read input image, apply edits, and save them
def main(input_path, output_paths):
    im = sk.img_as_float(skio.imread(input_path))  # Read and convert image to float

    edits = {}
    # Apply different contrast and white balancing techniques
    edits['0_1'] = contrast(im.copy())  # Basic 0-1 contrast
    edits['hist_eq'] = sk.exposure.equalize_hist(im.copy())  # Histogram equalization
    edits['ad_hist_eq'] = sk.exposure.equalize_adapthist(im.copy(), clip_limit=0.01)  # Adaptive histogram equalization
    edits['gray_world'] = white_balancing(im.copy(), False)  # Gray world white balancing
    edits['avg_world'] = white_balancing(im.copy(), True)  # Average world white balancing

    # Save each edited image to the corresponding output path
    for i in output_paths:
        save_image(edits[i], output_paths[i])

if __name__ == '__main__':
    input_path = 'path/to/input/image'
    # Output paths for different edited versions
    output_paths = {'0_1': 'path/to/0_1/output',
                    'hist_eq': 'path/to/hist_eq/output',
                    'ad_hist_eq': 'path/to/ad_hist_eq/output',
                    'gray_world': 'path/to/gray_world/output',
                    'avg_world': 'path/to/avg_world/output'}

    main(input_path, output_paths)  # Execute the main function
