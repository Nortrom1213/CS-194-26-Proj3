import matplotlib.pyplot as plt
import skimage as sk
import skimage.io as skio
import skimage.data as data
import skimage.transform as sktr
from skimage.draw import polygon
from scipy.spatial import Delaunay
import os
import glob
import numpy as np
import cv2
import pdb
import time
from sklearn.decomposition import PCA
from sklearn.neighbors import KDTree
import sys


def define_points(num_points, image):
    plt.imshow(image)
    print("Select %d points." % num_points)
    points = np.array(plt.ginput(num_points, timeout=0))
    plt.close()
    return points



img_source = sk.transform.resize(plt.imread('me.jpg'), [380,304])


if os.path.exists('source_points_58.npy'):
    source_points = np.load('source_points_58.npy')
else:
    source_points = define_points(58, img_source)
    padding = np.array([
        [0, 0],
        [img_source.shape[1] - 1, 0],
        [0, img_source.shape[0] - 1],
        [img_source.shape[1] - 1, img_source.shape[0] - 1]
    ])
    source_points = np.vstack([source_points, padding])
    np.save('source_points_58.npy', source_points)





def find_basis(points, index, basis_mat, mid_tri):
    tri = points[index]
    trans_mat = np.linalg.inv(computeAffine(tri, mid_tri))
    new_basis = np.dot(trans_mat, basis_mat).astype(int)
    return new_basis


def compute_masks(target, source, target_points, source_points, triangulation, warp_frac, dissolve_frac):
    if warp_frac < 0 or warp_frac > 1 or dissolve_frac < 0 or dissolve_frac > 1:
        return None

    masks = [np.zeros((WIDTH, LENGTH, 3)) for _ in range(3)]
    mid_pts = source_points + warp_frac * (target_points - source_points)

    for tri_index in triangulation.simplices:
        mid_tri = mid_pts[tri_index]
        mid_basis = np.array([polygon(mid_tri[:, 0], mid_tri[:, 1])])
        basis_mat = np.vstack([mid_basis, np.ones((1, mid_basis.shape[1]))])

        tgt_basis = find_basis(target_points, tri_index, basis_mat, mid_tri)
        masks[0][mid_basis[1], mid_basis[0]] = target[tgt_basis[1], tgt_basis[0]]

        src_basis = find_basis(source_points, tri_index, basis_mat, mid_tri)
        masks[2][mid_basis[1], mid_basis[0]] = source[src_basis[1], src_basis[0]]

    masks[1] = masks[2] + dissolve_frac * (masks[0] - masks[2])
    return masks