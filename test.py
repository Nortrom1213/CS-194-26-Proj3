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