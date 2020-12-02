"""
pai_basic.basis module
A set of basic functions to operate on gray-scale images
@author: jsaavedr
"""

from os import path
import numpy as np
import skimage.io as sk_io
import scipy.ndimage.filters as nd_filters


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


def compute_local_orientations(image, cell_size):
    g_x_local = np.zeros((cell_size, cell_size), dtype=np.float32)
    g_y_local = np.zeros((cell_size, cell_size), dtype=np.float32)
    r_local = np.zeros((cell_size, cell_size), dtype=np.float32)
    gx_mask = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], np.float32)
    gy_mask = np.transpose(gx_mask)
    gx = nd_filters.convolve(image.astype(np.float32), gx_mask)
    gy = nd_filters.convolve(image.astype(np.float32), gy_mask)
    mag = np.sqrt(np.square(gy) + np.square(gx))
    idx_rows, idx_cols = np.indices(image.shape)
    idx_grid_rows = np.floor(cell_size * idx_rows / image.shape[0])
    idx_grid_cols = np.floor(cell_size * idx_cols / image.shape[1])
    for p in np.arange(cell_size):
        for q in np.arange(cell_size):
            rows, cols = np.where((idx_grid_rows == p) & (idx_grid_cols == q))
            local_gx = gx[rows, cols]
            local_gy = gy[rows, cols]
            local_mag = mag[rows, cols]
            g_x_local[p, q] = np.sum((np.square(local_gx) - np.square(local_gy)))
            g_y_local[p, q] = np.sum(2.0 * (local_gx * local_gy))
            r_local[p, q] = np.mean(local_mag)
    local_ang = np.arctan2(g_y_local, g_x_local) * 0.5
    local_ang = local_ang + np.pi * 0.5  # 0 <= ang  <= pi
    return local_ang, r_local


def compute_local_orientations_softly(image, cell_size):
    g_x_local = np.zeros((cell_size, cell_size), dtype=np.float32)
    g_y_local = np.zeros((cell_size, cell_size), dtype=np.float32)
    r_local = np.zeros((cell_size, cell_size), dtype=np.float32)

    gx_mask = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], np.float32)
    gy_mask = np.transpose(gx_mask)
    gx = nd_filters.convolve(image.astype(np.float32), gx_mask)
    gy = nd_filters.convolve(image.astype(np.float32), gy_mask)

    mag = np.sqrt(np.square(gy) + np.square(gx))

    # Here p' p q' and q are calculated as learned in class 7
    idx_rows, idx_cols = np.indices(image.shape)
    idx_grid_rows_prime = cell_size * idx_rows / image.shape[0]
    idx_grid_rows = np.floor(idx_grid_rows_prime)
    idx_grid_cols_prime = cell_size * idx_cols / image.shape[1]
    idx_grid_cols = np.floor(idx_grid_cols_prime)


    top = np.floor(idx_grid_rows_prime - 0.5)
    bottom = top + 1
    left = np.floor(idx_grid_cols_prime - 0.5)
    right = left + 1

    dist_rows = idx_grid_rows_prime - idx_grid_rows
    dist_cols = idx_grid_cols_prime - idx_grid_cols

    w_t = np.zeros(image.shape, dtype=np.float32)
    w_b = np.zeros(image.shape, dtype=np.float32)
    w_l = np.zeros(image.shape, dtype=np.float32)
    w_r = np.zeros(image.shape, dtype=np.float32)

    # Conditions to compute weights
    w_t[dist_rows < 0.5] = 0.5 - dist_rows[dist_rows < 0.5]
    w_b[dist_rows < 0.5] = 1 - w_t[dist_rows < 0.5]
    w_b[dist_rows >= 0.5] = dist_rows[dist_rows >= 0.5] - 0.5
    w_t[dist_rows >= 0.5] = 1 - w_b[dist_rows >= 0.5]

    w_l[dist_cols < 0.5] = 0.5 - dist_cols[dist_cols < 0.5]
    w_r[dist_cols < 0.5] = 1 - w_l[dist_cols < 0.5]
    w_r[dist_cols >= 0.5] = dist_cols[dist_cols >= 0.5] - 0.5
    w_l[dist_cols >= 0.5] = 1 - w_r[dist_cols >= 0.5]

    w_lt = w_l * w_t
    w_lb = w_l * w_b
    w_rt = w_r * w_t
    w_rb = w_r * w_b

    for p in np.arange(cell_size):
        for q in np.arange(cell_size):
            rows_lt, cols_lt = np.where((top == p) & (left == q))
            rows_lb, cols_lb = np.where((bottom == p) & (left == q))
            rows_rt, cols_rt = np.where((top == p) & (right == q))
            rows_rb, cols_rb = np.where((bottom == p) & (right == q))

            local_gx_lt = gx[rows_lt, cols_lt]
            local_gx_lb = gx[rows_lb, cols_lb]
            local_gx_rt = gx[rows_rt, cols_rt]
            local_gx_rb = gx[rows_rb, cols_rb]

            local_gy_lt = gy[rows_lt, cols_lt]
            local_gy_lb = gy[rows_lb, cols_lb]
            local_gy_rt = gy[rows_rt, cols_rt]
            local_gy_rb = gy[rows_rb, cols_rb]

            local_mag_lt = w_lt[rows_lt, cols_lt] * mag[rows_lt, cols_lt]
            local_mag_lb = w_lb[rows_lb, cols_lb] * mag[rows_lb, cols_lb]
            local_mag_rt = w_rt[rows_rt, cols_rt] * mag[rows_rt, cols_rt]
            local_mag_rb = w_rb[rows_rb, cols_rb] * mag[rows_rb, cols_rb]

            g_x_local_lt = np.sum(w_lt[rows_lt, cols_lt] * (np.square(local_gx_lt) - np.square(local_gy_lt)))
            g_x_local_lb = np.sum(w_lb[rows_lb, cols_lb] * (np.square(local_gx_lb) - np.square(local_gy_lb)))
            g_x_local_rt = np.sum(w_rt[rows_rt, cols_rt] * (np.square(local_gx_rt) - np.square(local_gy_rt)))
            g_x_local_rb = np.sum(w_rb[rows_rb, cols_rb] * (np.square(local_gx_rb) - np.square(local_gy_rb)))

            g_x_local[p, q] = g_x_local_lt + g_x_local_lb + g_x_local_rt + g_x_local_rb

            g_y_local_lt = np.sum(w_lt[rows_lt, cols_lt] * (2.0 * local_gx_lt * local_gy_lt))
            g_y_local_lb = np.sum(w_lb[rows_lb, cols_lb] * (2.0 * local_gx_lb * local_gy_lb))
            g_y_local_rt = np.sum(w_rt[rows_rt, cols_rt] * (2.0 * local_gx_rt * local_gy_rt))
            g_y_local_rb = np.sum(w_rb[rows_rb, cols_rb] * (2.0 * local_gx_rb * local_gy_rb))

            g_y_local[p, q] = g_y_local_lt + g_y_local_lb + g_y_local_rt + g_y_local_rb

            r_local[p, q] = np.mean(np.concatenate((local_mag_lt, local_mag_lb, local_mag_rt, local_mag_rb)))

    local_ang = np.arctan2(g_y_local, g_x_local) * 0.5
    local_ang = local_ang + np.pi * 0.5  # 0 <= ang  <= pi
    return local_ang, r_local


def compute_orientation_histogram(image, L, K, local_orientations=True, soft_orientations=True):
    h = np.zeros(L, np.float32)

    if local_orientations:
        if soft_orientations:
            ang, mag = compute_local_orientations_softly(image=image, cell_size=K)
        else:
            ang, mag = compute_local_orientations(image=image, cell_size=K)
    else:
        gx_mask = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], np.float32)
        gy_mask = np.transpose(gx_mask)
        gx = nd_filters.convolve(image.astype(np.float32), gx_mask)
        gy = nd_filters.convolve(image.astype(np.float32), gy_mask)
        ang = np.arctan2(gy, gx)
        ang[ang < 0] = ang[ang < 0] + np.pi  # sin signo
        mag = np.sqrt(np.square(gy) + np.square(gx))

    idx_prime = L * ang / np.pi
    idx = np.floor(idx_prime)

    left = np.floor(idx_prime - 0.5)
    right = left + 1

    # Version of indx[indx == K] = 0 has been extended
    # Borders
    right[right == L] = 0
    left[left == -1] = L - 1

    dist = idx_prime - idx

    w_l = np.zeros(mag.shape, dtype=np.float32)
    w_r = np.zeros(mag.shape, dtype=np.float32)

    w_l[dist < 0.5] = 0.5 - dist[dist < 0.5]
    w_r[dist < 0.5] = 1 - w_l[dist < 0.5]
    w_r[dist >= 0.5] = dist[dist >= 0.5] - 0.5
    w_l[dist >= 0.5] = 1 - w_r[dist >= 0.5]

    for i in range(L):
        rows_l, cols_l = np.where(left == i)
        rows_r, cols_r = np.where(right == i)

        mag_l = np.sum(w_l[rows_l, cols_l] * mag[rows_l, cols_l])
        mag_r = np.sum(w_r[rows_r, cols_r] * mag[rows_r, cols_r])

        h[i] = mag_l + mag_r
    h = h / np.linalg.norm(h, 2)  # vector unitario
    return h


def get_type(fingerprint_path):
    return path.split(path.split(fingerprint_path)[0])[1]
