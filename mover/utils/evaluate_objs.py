import numpy as np
import scipy.io
from shapely.geometry.polygon import Polygon
from scipy.stats import multivariate_normal
from scipy.spatial import ConvexHull
import matplotlib.pyplot as plt

def get_iou_cuboid(cu1, cu2):
    """
        Calculate the Intersection over Union (IoU) of two 3D cuboid.

        Parameters
        ----------
        cu1 : numpy array, 8x3
        cu2 : numpy array, 8x3

        Returns
        -------
        float
            in [0, 1]
    """
    polygon_1 = Polygon([(cu1[0][0], cu1[0][1]), (cu1[1][0], cu1[1][1]), (cu1[2][0], cu1[2][1]), (cu1[3][0], cu1[3][1])])
    polygon_2 = Polygon([(cu2[0][0], cu2[0][1]), (cu2[1][0], cu2[1][1]), (cu2[2][0], cu2[2][1]), (cu2[3][0], cu2[3][1])])
    intersect_2d = polygon_1.intersection(polygon_2).area
    inter_vol = intersect_2d * max(0.0, min(cu1[0][2], cu2[0][2]) - max(cu1[4][2], cu2[4][2]))
    vol1 = polygon_1.area * (cu1[0][2] - cu1[4][2])
    vol2 = polygon_2.area * (cu2[0][2] - cu2[4][2])
    return inter_vol / (vol1 + vol2 - inter_vol)

def get_corners_of_bb3d(bb3d):
    corners = np.zeros((8, 3))
    # order the basis
    basis_tmp = bb3d['basis']
    inds = np.argsort(np.abs(basis_tmp[:, 0]))[::-1]
    basis = basis_tmp[inds, :]
    coeffs = bb3d['coeffs'].T[inds]

    inds = np.argsort(np.abs(basis[1:3, 1]))[::-1]

    if inds[0] == 1:
        basis[1:3, :] = np.flip(basis[1:3, :], 0)
        coeffs[1:3] = np.flip(coeffs[1:3], 1)
    centroid = bb3d['centroid']

    basis = flip_toward_viewer(basis, np.tile(centroid, (3, 1)))
    coeffs = np.abs(coeffs)

    corners[0, :] = -basis[0, :] * coeffs[0] + basis[1, :] * coeffs[1] + basis[2, :] * coeffs[2]
    corners[1, :] = basis[0, :] * coeffs[0] + basis[1, :] * coeffs[1] + basis[2, :] * coeffs[2]
    corners[2, :] = basis[0, :] * coeffs[0] + -basis[1, :] * coeffs[1] + basis[2, :] * coeffs[2]
    corners[3, :] = -basis[0, :] * coeffs[0] + -basis[1, :] * coeffs[1] + basis[2, :] * coeffs[2]

    corners[4, :] = -basis[0, :] * coeffs[0] + basis[1, :] * coeffs[1] + -basis[2, :] * coeffs[2]
    corners[5, :] = basis[0, :] * coeffs[0] + basis[1, :] * coeffs[1] + -basis[2, :] * coeffs[2]
    corners[6, :] = basis[0, :] * coeffs[0] + -basis[1, :] * coeffs[1] + -basis[2, :] * coeffs[2]
    corners[7, :] = -basis[0, :] * coeffs[0] + -basis[1, :] * coeffs[1] + -basis[2, :] * coeffs[2]

    corners = corners + np.tile(centroid, (8, 1))
    return corners


def flip_toward_viewer(normals, points):
    
    points = np.divide(points, np.tile(np.sqrt(np.sum(np.square(points), axis=1)), (3, 1)))
    proj = np.sum(np.multiply(points, normals), axis=1)
    flip = proj > 0
    normals[flip, :] = -normals[flip, :]

    return normals

# in 2d, return the intersection vertial length of two bbx / length of bbx1 #1
def vertical_intersect_ratio(bbx1, bbx2):
    if bbx2[0] > bbx1[2] or bbx2[2] < bbx1[0]:
        return 0
    else:
        return (np.min([bbx1[2], bbx2[2]]) - np.max([bbx1[0], bbx2[0]])) / float(bbx1[2] - bbx1[0])


# return the intersection area of two cuboid in x-y coordinates / area of cuboid #1
def intersection_2d_ratio(cu1, cu2):
    polygon_1 = Polygon([(cu1[0][0], cu1[0][1]), (cu1[1][0], cu1[1][1]), (cu1[2][0], cu1[2][1]), (cu1[3][0], cu1[3][1])])
    polygon_2 = Polygon([(cu2[0][0], cu2[0][1]), (cu2[1][0], cu2[1][1]), (cu2[2][0], cu2[2][1]), (cu2[3][0], cu2[3][1])])
    intersection_ratio = polygon_1.intersection(polygon_2).area / polygon_1.area
    return intersection_ratio


