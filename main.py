import numpy as np
import matplotlib
import matplotlib.animation as animation
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


IMG_SHAPE = (750, 600)
NUM_POINTS = 33

def define_points(num_points, image):
    plt.imshow(image)
    print("Select %d points." % num_points)
    points = np.array(plt.ginput(num_points, timeout=0))
    plt.close()
    return points

# read img
target = plt.imread('depp.jpg')
source = plt.imread('me.jpg')
target = sk.transform.resize(target, IMG_SHAPE)
source = sk.transform.resize(source, IMG_SHAPE)
# select points
if os.path.exists('target_points.npy'):
    target_points = np.load('target_points.npy')
    source_points = np.load('source_points.npy')
else:
    target_points = define_points(NUM_POINTS, target)
    source_points = define_points(NUM_POINTS, source)
    # save points
    np.save("target_points.npy", target_points)
    np.save("source_points.npy", source_points)


corners = np.array([
    [0, 0],
    [IMG_SHAPE[1]-1, 0],
    [0, IMG_SHAPE[0]-1],
    [IMG_SHAPE[1]-1, IMG_SHAPE[0]-1]
])
target_points = np.vstack([target_points, corners])
source_points = np.vstack([source_points, corners])


# triangulation
mid_points = (target_points + source_points)/2
triangulation = Delaunay(mid_points)

plt.figure(figsize=(20, 10))
for i, (img, points) in enumerate(zip((target, source), (target_points, source_points))):
    plt.subplot(1, 2, i+1)
    plt.imshow(img)
    plt.triplot(points[:,0], points[:,1], triangulation.simplices)
    plt.plot(points[:,0], points[:,1], 'o')

plt.savefig('triangulation.jpg')


def trans(points):
    vector1 = np.reshape((points[1] - points[0]), (2, 1))
    vector2 = np.reshape((points[2] - points[0]), (2, 1))
    origin = np.reshape(points[0], (2, 1))
    upper_matrix = np.hstack([vector1, vector2, origin])
    trans_matrix = np.vstack([upper_matrix, np.array([[0, 0, 1]])])
    return trans_matrix

def computeAffine(tri1_pts, tri2_pts):
    return np.dot(trans(tri2_pts), np.linalg.inv(trans(tri1_pts)))


def compute_masks(target, source, target_points, source_points, triangulation, warp_frac, dissolve_frac):
    if warp_frac < 0 or warp_frac > 1 or dissolve_frac < 0 or dissolve_frac > 1:
        return None

    masks = [np.zeros((IMG_SHAPE[0], IMG_SHAPE[1], 3)) for _ in range(3)]
    mid_pts = source_points + warp_frac * (target_points - source_points)

    for tri_index in triangulation.simplices:
        mid_tri = mid_pts[tri_index]
        rr, cc = polygon(mid_tri[:, 0], mid_tri[:, 1])
        mid_xys = np.array([rr, cc])
        tmp = np.vstack([mid_xys, np.ones((1, mid_xys.shape[1]))])

        src1_tri = target_points[tri_index]
        mat_T1 = np.linalg.inv(computeAffine(src1_tri, mid_tri))
        src1_xys = np.dot(mat_T1, tmp).astype(int)
        masks[0][mid_xys[1], mid_xys[0]] = target[src1_xys[1], src1_xys[0]]

        src2_tri = source_points[tri_index]
        mat_T2 = np.linalg.inv(computeAffine(src2_tri, mid_tri))
        src2_xys = np.dot(mat_T2, tmp).astype(int)
        masks[2][mid_xys[1], mid_xys[0]] = source[src2_xys[1], src2_xys[0]]

    masks[1] = masks[2] + dissolve_frac * (masks[0] - masks[2])
    return masks


plt.figure(figsize=(15, 5))
masks = compute_masks(target, source, target_points, source_points, triangulation, 0.5, 0.5)
for i in range(3):
    plt.subplot(1, 3, i + 1)
    plt.imshow(masks[i])
plt.savefig('midway_face.jpg')

NUM_FRAMES = 45

def morph(im1, im2, pts1, pts2, tri, warp_frac, dissolve_frac):
    return compute_masks(target, source, target_points, source_points, triangulation, x, x)[1]

# generate morphing
results = []
fig = plt.figure(figsize=(8, 10))
for x in np.linspace(0, 1, NUM_FRAMES):
    img = morph(target, source, target_points, source_points, triangulation, x, x)
    img = plt.imshow(img, animated=True)
    results.append([img])

# generate animation and save
vid = animation.ArtistAnimation(
    fig,
    results,
    interval=100,
    blit=True,
    repeat_delay=1000
)
vid.save('morphing.gif', fps=30)
vid.save('morphing_slowmo.gif', fps=10)






# mean of a population

DIR = 'DB'

def get_pts_from_file(filename, pad=True):
    im = plt.imread('DB/01-1m.jpg')

    data = open(filename, 'r').readlines()[16:74]
    data = [l.split('\t')[2:4] for l in data]
    relative_points = np.array(data).astype(float)

    real_points = np.multiply(np.array([im.shape[1], im.shape[0]]), relative_points)
    if pad:
        corners = np.array([
            [0, 0],
            [im.shape[1]-1, 0], [0, im.shape[0]-1],
            [im.shape[1]-1, im.shape[0]-1]
        ])
        real_points = np.vstack([real_points, corners])
    return real_points

def load_imgs(keyword):
    images = []
    points = []
    for filename in os.listdir(DIR):
        basename, ext = os.path.splitext(filename)
        if keyword in basename and ext == '.jpg':
            images.append(plt.imread(os.path.join(DIR, filename)) / 255.)
            pts = get_pts_from_file(os.path.join(DIR, basename + '.asf'))
            points.append(pts)
    return images, points


def get_mean_face(images, pts):
    mean_pts = np.mean(pts, axis=0)
    num_ims = len(images)
    triangulation = Delaunay(mean_pts)
    result = np.zeros((images[0].shape[0], images[0].shape[1], 3))
    for tri_index in triangulation.simplices:
        mid_tri = mean_pts[tri_index]
        rr, cc = polygon(mid_tri[:, 0], mid_tri[:, 1])
        mid_xys = np.array([rr, cc])
        tmp = np.vstack([mid_xys, np.ones((1, mid_xys.shape[1]))])

        for i in range(num_ims):
            source_triangle = pts[i][tri_index]
            mat_T = computeAffine(mid_tri, source_triangle)
            src_xys = np.dot(mat_T, tmp).astype(int)
            result[mid_xys[1], mid_xys[0]] += images[i][src_xys[1], src_xys[0]] / num_ims

    return result, mean_pts

keywords = ['1m', '2m', '1f', '2f']
images = []
points = []
mean_faces = []
mean_points = []
for i, keyword in enumerate(keywords):
    imgs, pts = load_imgs(keyword)
    images.append(imgs)
    points.append(pts)

    mean_face, mean_pts = get_mean_face(imgs, pts)
    mean_faces.append(mean_face)

    mean_points.append(mean_pts)

# plot average faces
titles = [
    'average face of neutral males',
    'average face of happy males',
    'average face of neutral females',
    'average face of happy females'
]
plt.figure(figsize=(20, 16))
for i in range(4):
    plt.subplot(2, 2, i+1)
    plt.title(titles[i])
    plt.imshow(mean_faces[i])
plt.savefig('DB/mean_face.jpg')

def normalize(mat, coord):
    coord1 = coord[0]
    coord2 = coord[1]
    for i in range(len(coord1)):
        if coord1[i] >= mat.shape[0]:
            coord1[i] = mat.shape[0] - 1

    for i in range(len(coord2)):
        if coord2[i] >= mat.shape[1]:
            coord2[i] = mat.shape[1] - 1

# warp some faces to mean geometry
def compute_warp(img_src, points_src, points_tgt, a=1.):
    points_mean = points_src + (points_tgt - points_src) / 2
    points_tgt = points_src + (points_tgt - points_src) * a
    triangulation = Delaunay(points_mean)
    result = np.zeros((img_src.shape[0], img_src.shape[1], 3))
    for index in triangulation.simplices:
        triangle_tgt = points_tgt[index]
        r, c = polygon(triangle_tgt[:, 0], triangle_tgt[:, 1])
        coord_tgt = np.array([r, c])
        tmp = np.vstack([coord_tgt, np.ones((1, coord_tgt.shape[1]))])

        triangle_src = points_src[index]
        mat = computeAffine(triangle_tgt, triangle_src)
        coord_src = np.dot(mat, tmp).astype(int)
        result[coord_tgt[1], coord_tgt[0]] = img_src[coord_src[1], coord_src[0]]

    return result


PID = 0
titles = [
    'neutral male',
    'happy male',
    'neutral female',
    'happy female'
]
plt.figure(figsize=(32, 10))
for i in range(4):
    plt.subplot(2, 4, i + 1)
    plt.title(titles[i])
    plt.imshow(images[i][PID])

    plt.subplot(2, 4, i + 5)
    plt.title(titles[i] + 'warped to average')
    img = compute_warp(images[i][PID], points[i][PID], mean_points[i])
    plt.imshow(img)
plt.savefig('DB/warp_to_mean.jpg')

cropped_mean_faces = []
cropped_mean_points = []
for face, pts in zip(mean_faces, mean_points):
    cropped_face = face[50:430, 168:472]
    padding = np.array([
        [0, 0],
        [cropped_face.shape[1] - 1, 0],
        [0, cropped_face.shape[0] - 1],
        [cropped_face.shape[1] - 1, cropped_face.shape[0] - 1]
    ])
    cropped_points = pts - np.array([168, 50])
    cropped_points = np.vstack([cropped_points[:-4], padding])

    cropped_mean_faces.append(cropped_face)
    cropped_mean_points.append(cropped_points)

# get coordinates
img_source = sk.transform.resize(plt.imread('me.jpg'), cropped_mean_faces[0].shape)

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


# warp my face
plt.figure(figsize=(20, 12))
for i in range(4):
    plt.subplot(2, 4, i+1)
    plt.title('me to' + titles[i])
    img_to_avg = compute_warp(img_source, source_points, cropped_mean_points[i])
    plt.imshow(img_to_avg)

    plt.subplot(2, 4, i+5)
    plt.title(titles[i] + ' to me')
    img_from_avg = compute_warp(cropped_mean_faces[i], cropped_mean_points[i], source_points)
    plt.imshow(img_from_avg)
plt.savefig('me_vs_avg.jpg')


