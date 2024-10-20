import csv
from laplacian_blending import *


def read_label_csv(in_path):
    with open(in_path, 'r') as file:
        reader = csv.reader(file)
        loaded_points = [[float(row[0]), float(row[1])] for row in reader]
    return np.array(loaded_points)


def computeH(p1, p2):
    eqs = []
    p_prime = []
    for (x1, y1), (x2, y2) in zip(p1, p2):
        eqs.append([x1, y1, 1, 0, 0, 0, -x1 * x2, -y1 * x2])
        eqs.append([0, 0, 0, x1, y1, 1, -x1 * y2, -y1 * y2])
        p_prime.append(x2)
        p_prime.append(y2)

    H_vars, _, _, _ = np.linalg.lstsq(np.array(eqs), np.array(p_prime), rcond=None)

    H = [
        [H_vars[0], H_vars[1], H_vars[2]],
        [H_vars[3], H_vars[4], H_vars[5]],
        [H_vars[6], H_vars[7], 1]
    ]
    return np.array(H)


def warpImage(im, H, outpath=None):
    H_inv = np.linalg.inv(H)

    height, width = im.shape[:2]
    col_c = [0, 0, width - 1, width - 1]
    row_c = [0, height - 1, height - 1, 0]
    p_corners = np.array([col_c, row_c, [1, 1, 1, 1]])

    col_c_prime, row_c_prime, w = np.dot(H, p_corners)
    col_c_prime /= w
    row_c_prime /= w

    min_row = int(np.floor(min(0, min(row_c_prime))))
    min_col = int(np.floor(min(0, min(col_c_prime))))
    max_row = int(np.ceil(max(row_c_prime)))
    max_col = int(np.ceil(max(col_c_prime)))

    box_height = max_row - min_row + 1
    box_width = max_col - min_col + 1

    box = np.zeros((box_height, box_width, im.shape[2]))

    row_box, col_box = sk.draw.polygon([0, box_height - 1, box_height - 1, 0], [0, 0, box_width - 1, box_width - 1])

    X = np.stack([col_box + min_col, row_box + min_row, np.ones(col_box.shape)])

    col, row, w = np.dot(H_inv, X)
    col /= w
    row /= w

    col = np.round(col).astype(int)
    row = np.round(row).astype(int)

    mask_p = (col >= 0) & (col < width) & (row >= 0) & (row < height)
    valid_row_box = row_box[mask_p]
    valid_col_box = col_box[mask_p]
    box[valid_row_box, valid_col_box] = im[row[mask_p], col[mask_p]]

    return box, [min_row, min_col, max_row, max_col]


def mosaic(im1, im2, im1_pts, im2_pts, outpath=None):
    H = computeH(im1_pts, im2_pts)

    im1 = alpha_feathering(im1)
    im2 = alpha_feathering(im2)

    im1_warped, [min_row, min_col, max_row, max_col] = warpImage(im1, H, outpath)

    height_im1_warped, width_im1_warped = im1_warped.shape[:2]
    height_im2, width_im2 = im2.shape[:2]

    mosaic_height = max(max_row, height_im2) - min(0, min_row) + 1
    mosaic_width = max(max_col, width_im2) - min(0, min_col) + 1

    mosaic_box1 = np.zeros((mosaic_height, mosaic_width, im1.shape[2]))
    mosaic_box2 = np.zeros((mosaic_height, mosaic_width, im1.shape[2]))
    row_im1 = min_row - min(0, min_row)
    col_im1 = min_col - min(0, min_col)

    min_row_im2 = -min(0, min_row)
    min_col_im2 = -min(0, min_col)

    mosaic_box1[row_im1:row_im1 + height_im1_warped, col_im1:col_im1 + width_im1_warped] = im1_warped
    mosaic_box2[min_row_im2:min_row_im2 + height_im2, min_col_im2:min_col_im2 + width_im2] = im2

    feather_1 = mosaic_box1[:,:,3]
    feather_2 = mosaic_box2[:,:,3]

    mask = feather_1 > feather_2
    mask = np.dstack([mask, mask, mask])
    display_save_image(mask, outpath)

    mosaic_box1 = mosaic_box1[:,:,:3]
    mosaic_box2 = mosaic_box2[:,:,:3]

    combined = oraple_stack(mosaic_box1, mosaic_box2, 35, 2, out_path=None, mask=mask)
    combined = np.clip(combined, 0,1)

    display_save_image(combined, outpath)
    return combined


def alpha_feathering(im):
    height, width = im.shape[0], im.shape[1]

    max_dist = max(height / 2, width / 2)

    rows = np.arange(height)
    cols = np.arange(width)

    rows = np.min([rows, height - rows - 1], axis=0)
    cols = np.min([cols, width - cols - 1], axis=0)
    alpha_channel = np.minimum.outer(rows, cols) / max_dist

    alpha_channel = np.clip(alpha_channel, 0, 1)
    im = np.dstack([im, alpha_channel])
    return im


def display_save_image(image, fname=None):
    image = (255 * image).astype(np.uint8)
    if fname:
        skio.imsave(fname, image)
    skio.imshow(image)
    skio.show()


if __name__ == '__main__':

    path = '/Users/meenakshimittal/Desktop/cs180/meenakshi-mittal.github.io/4/Project 4'
    im = sk.img_as_float(skio.imread(f'{path}/rectification/laptop.JPG'))
    im_pts_path = f'{path}/rectification/laptop_labels.csv'
    warp_points = [[0, 240], [0, 40], [280, 240], [280, 240]]
    im_pts = read_label_csv(im_pts_path)

    H = computeH(im_pts, warp_points)
    warpImage(im, H, f'{path}/rectification/laptop_rectified.jpg')

    im = sk.img_as_float(skio.imread(f'{path}/rectification/tennis2.JPG'))

    im_pts_path = f'{path}/rectification/tennis2_labels.csv'
    warp_points = [[0, 390], [0, 0], [180, 0], [180, 390]]
    im_pts = read_label_csv(im_pts_path)

    H = computeH(im_pts, warp_points)
    warped, _ = warpImage(im, H)
    display_save_image(warped, f'{path}/rectification/tennis2_rectified.jpg')


    i1 = 'glass4'
    i2 = 'glass3'
    iout = 'glass34_mask'
    im1 = sk.img_as_float(skio.imread(f'{path}/glass/images/{i1}.jpg'))
    im1_pts_path = f'{path}/glass/points/{i1}_labels.csv'
    im2 = sk.img_as_float(skio.imread(f'{path}/glass/images/{i2}.jpg'))
    im2_pts_path = f'{path}/glass/points/{i2}_labels.csv'
    #
    im1_pts = read_label_csv(im1_pts_path)
    im2_pts = read_label_csv(im2_pts_path)

    mosaic(im1, im2, im1_pts, im2_pts, f'{path}/glass/images/{iout}.jpg')

