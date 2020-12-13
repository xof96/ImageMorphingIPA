"""
pai_basic.basis module
A set of basic functions to operate on gray-scale images
@author: jsaavedr
"""

import numpy as np
import skimage.io as sk_io


# to read
def imread(filename, as_gray=False):
    image = sk_io.imread(filename, as_gray=as_gray)
    if image.dtype == np.float64:
        image = to_uint8(image)
    return image


# to uint8
def to_uint8(image):
    if image.dtype == np.float64:
        image = image * 255
    image[image < 0] = 0
    image[image > 255] = 255
    image = image.astype(np.uint8, copy=False)
    return image


def warping(dest_x, dest_q, dest_p, ori_p, dest_pq, ori_pq, dest_per_pq, ori_per_pq, dest_len_pq, ori_len_pq, n_lines,
            p, b, a):
    d_sum = np.zeros(2)
    w_sum = 0
    for k in range(n_lines):
        dest_px_k = dest_x - dest_p[k]
        dest_qx_k = dest_x - dest_q[k]
        u = np.dot(dest_px_k, dest_pq[k]) / np.square(dest_len_pq[k])
        v = np.dot(dest_px_k, dest_per_pq[k]) / dest_len_pq[k]
        ori_x_k = ori_p[k] + u * ori_pq[k] + v * ori_per_pq[k] / ori_len_pq[k]
        dk = ori_x_k - dest_x
        if u < 0:
            dist = np.sqrt(np.sum(np.square(dest_px_k)))
        elif u > 1:
            dist = np.sqrt(np.sum(np.square(dest_qx_k)))
        else:
            dist = np.abs(v)
        weight = (dest_len_pq[k] ** p / (a + dist)) ** b
        d_sum += dk * weight
        w_sum += weight
    ori_x = dest_x + d_sum / w_sum
    return ori_x


def is_inside_img(x, y, shape):
    return x < 0 or y < 0 or y + 1 >= shape[0] or x + 1 >= shape[1]


def bi_linear_interpolation(p, q, a, b, img):
    return (1 - a) * ((1 - b) * img[p, q] + b * img[p, q + 1]) + a * ((1 - b) * img[p + 1, q] + b * img[p + 1, q + 1])
