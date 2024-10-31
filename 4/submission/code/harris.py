import numpy as np
from matplotlib import pyplot as plt
from skimage.feature import corner_harris, peak_local_max
import skimage as sk
import skimage.io as skio
# from main import *
import scipy

def get_harris_corners(im, edge_discard=20, threshold=0.00):
    """
    This function takes a b&w image and an optional amount to discard
    on the edge (default is 5 pixels), and finds all harris corners
    in the image. Harris corners near the edge are discarded and the
    coordinates of the remaining corners are returned. A 2d array (h)
    containing the h value of every pixel is also returned.

    h is the same shape as the original image, im.
    coords is 2 x n (ys, xs).
    """

    assert edge_discard >= 20

    # find harris corners
    h = corner_harris(im, method='eps', sigma=1)
    h[h < threshold * h.max()] = 0

    coords = peak_local_max(h, min_distance=1)

    # discard points on edge
    edge = edge_discard  # pixels
    mask = (coords[:, 0] > edge) & \
           (coords[:, 0] < im.shape[0] - edge) & \
           (coords[:, 1] > edge) & \
           (coords[:, 1] < im.shape[1] - edge)
    coords = coords[mask].T
    return h, coords


def dist2(x, c):
    """
    dist2  Calculates squared distance between two sets of points.

    Description
    D = DIST2(X, C) takes two matrices of vectors and calculates the
    squared Euclidean distance between them.  Both matrices must be of
    the same column dimension.  If X has M rows and N columns, and C has
    L rows and N columns, then the result has M rows and L columns.  The
    I, Jth entry is the  squared distance from the Ith row of X to the
    Jth row of C.

    Adapted from code by Christopher M Bishop and Ian T Nabney.
    """

    ndata, dimx = x.shape
    ncenters, dimc = c.shape
    assert dimx == dimc, 'Data dimension does not match dimension of centers'

    return (np.ones((ncenters, 1)) * np.sum((x ** 2).T, axis=0)).T + \
        np.ones((ndata, 1)) * np.sum((c ** 2).T, axis=0) - \
        2 * np.inner(x, c)

def computeH(p1, p2):
    eqs = []
    p_prime = []
    for (x1, y1), (x2, y2) in zip(p1, p2):
        eqs.append([x1, y1, 1, 0, 0, 0, -x1 * x2, -y1 * x2])
        eqs.append([0, 0, 0, x1, y1, 1, -x1 * y2, -y1 * y2])
        p_prime.append(x2)
        p_prime.append(y2)

    if len(p1) == 4:
        H_vars = np.linalg.solve(np.array(eqs), np.array(p_prime))
    else:
        H_vars, _, _, _ = np.linalg.lstsq(np.array(eqs), np.array(p_prime), rcond=None)

    H = [
        [H_vars[0], H_vars[1], H_vars[2]],
        [H_vars[3], H_vars[4], H_vars[5]],
        [H_vars[6], H_vars[7], 1]
    ]
    return np.array(H)

def anms(h, coords, c_robust=0.9, num_points=500):
    distances = dist2(coords.T, coords.T)
    h_of_coords = np.array([h[row, col] for row, col in coords.T])
    print(np.mean(h_of_coords))
    print(len(h_of_coords))

    radii = []
    for idx, (row,col) in enumerate(coords.T):
        candidates = np.array(h[row,col] < c_robust*h_of_coords)
        masked = distances[idx][candidates]
        radius = min(masked) if len(masked) > 0 else np.inf
        radii.append(radius)

    num_points = min(num_points,len(coords.T))
    threshold = sorted(radii, reverse=True)[num_points-1]
    anms_coords = [coords.T[i] for i in range(len(coords.T)) if radii[i] >= threshold][:num_points]

    return np.array(anms_coords).T

def normalize(img):
    img = img - np.min(img)
    return img / np.max(img)


def get_feature_descriptors(im, coords):
    feats = []
    for (row, col) in coords.T:
        feat = im[row-20:row+20, col-20:col+20]
        scaled_feat = sk.transform.rescale(feat, 0.2, anti_aliasing=True)
        norm_scaled_feat = (scaled_feat-np.mean(scaled_feat))/np.std(scaled_feat)
        feats.append(norm_scaled_feat)
    return feats

def match_features(im1_feats, im2_feats, threshold=0.3):
    # iterate over im1_feats
    # compare each one to all im2_feats
    # keep track of top 2 nearest neighbors for each im1_feat
    # store index of feat, and error: [[index of 1st nearest, error], [index of 2nd nearest, error]]
    nn_ratios = []
    for idx1, i in enumerate(im1_feats):
        nn = [[0,np.inf],[0,np.inf]]
        for idx2, j in enumerate(im2_feats):
            error = np.sum((i - j) ** 2)
            if error < nn[0][1]: # if error is less than current nearest neighbor
                nn[1] = nn[0] # 1st nearest neighbor moves down to second
                nn[0] = [idx2,error]
            elif error < nn[1][1]: # else, if error is less than current 2nd nn
                nn[1] = [idx2,error] # replace 2nd nn with current
        nn_ratios.append([nn[0][0],nn[0][1]/nn[1][1]])

    matches = [[idx, i[0]] for idx,i in enumerate(nn_ratios) if i[1]<threshold]
    print(matches)

    return np.array(matches)

def ransac(im1_matches, im2_matches, num_samples=10000, epsilon=4):
    best_H = None
    most_inliers = 0

    for i in range(num_samples):
        indices = np.random.choice(len(im1_matches.T), 4, replace=False)
        im1_sample = im1_matches[:, indices]
        im2_sample = im2_matches[:, indices]

        H = computeH(im1_sample.T, im2_sample.T)

        im1_matches_ones = np.stack([im1_matches[0], im1_matches[1], np.ones(im1_matches[0].shape)])

        col1, row1, w = np.dot(H, im1_matches_ones)
        w += 1e-16 # avoid divide by zero
        col1 /= w
        row1 /= w

        warped_im1 = np.array([col1, row1])
        distances = np.sqrt(np.sum((warped_im1 - im2_matches) ** 2, axis=0))

        num_inliers = np.sum(distances < epsilon)

        if num_inliers > most_inliers:
            most_inliers = num_inliers
            best_H = H
    return best_H

def display_save_image(image, fname=None):
    image = (255 * image).astype(np.uint8)
    if fname:
        skio.imsave(fname, image)
    skio.imshow(image)
    skio.show()

def get_best_H(im1, im2):
    h, coords1 = get_harris_corners(sk.color.rgb2gray(im1))
    coords1 = anms(h, coords1)
    im1_feats = get_feature_descriptors(im1, coords1)

    h, coords2 = get_harris_corners(sk.color.rgb2gray(im2))
    coords2 = anms(h, coords2)
    im2_feats = get_feature_descriptors(im2, coords2)

    matches = match_features(im1_feats, im2_feats)

    im1_matches = np.array(coords1[:, matches[:, 0]])[::-1]
    im2_matches = np.array(coords2[:, matches[:, 1]])[::-1]

    best_H = ransac(im1_matches, im2_matches)

    print(best_H)
    return best_H
