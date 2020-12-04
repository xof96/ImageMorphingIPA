from utils.utils import imread
import numpy as np
import matplotlib.pyplot as plt

img_o_path = 'assets/shapes/square.jpg'
img_d_path = 'assets/shapes/diamond.jpg'

img_origin = imread(filename=img_o_path)
if img_origin.shape[2] > 3:
    img_origin = img_origin[:, :, 0:3]
img_destination = imread(filename=img_d_path)

ori_shape = img_origin.shape
d_shape = img_destination.shape
# %%
buf_shape = []
n_images = 11  # Total: from origin image to destination image.
for i in range(3):
    buf_shape.append(max(ori_shape[i], d_shape[i]))
buf_images = np.zeros(shape=[n_images] + buf_shape, dtype=np.uint8)

# if __name__ == '__main__':
# %%
dt = 1 / (n_images - 1)  # We subtract 1 not to count the destination image.
with open('data/shape_lines.txt', 'r') as ref_lines:
    lines = ref_lines.readlines()
    if lines[-1] == '':
        lines.pop()

# %%
n_lines = len(lines)
pa = np.zeros(shape=(n_lines, 2))
qa = np.zeros(shape=(n_lines, 2))
pb = np.zeros(shape=(n_lines, 2))
qb = np.zeros(shape=(n_lines, 2))
for i in range(n_lines):
    if lines[i][-1] == '\n':
        lines[i] = lines[i][0: -1]
    coords = lines[i].split(',')
    curr_pa = (int(coords[0]), int(coords[1]))
    curr_qa = (int(coords[2]), int(coords[3]))
    curr_pb = (int(coords[4]), int(coords[5]))
    curr_qb = (int(coords[6]), int(coords[7]))
    pa[i] = curr_pa
    qa[i] = curr_qa
    pb[i] = curr_pb
    qb[i] = curr_qb

# %%
t = np.linspace(0, 1, n_images)
p = np.zeros(shape=(n_lines, n_images, 2))
q = np.zeros(shape=(n_lines, n_images, 2))
for i in range(n_images):
    p[:, i] = pa * (1 - t[i]) + pb * t[i]
    q[:, i] = qa * (1 - t[i]) + qb * t[i]

# %%
# Calculating PQs
pq = q - p

per_pq = np.zeros(shape=(n_lines, n_images, 2))
per_pq[:, :, 0] = pq[:, :, 1]
per_pq[:, :, 1] = -1 * pq[:, :, 0]

len_pq = np.sqrt(np.square(pq[:, :, 0]) + np.square(pq[:, :, 1]))


# %%
def warping(dest_x, dest_p, ori_p, dest_pq, ori_pq, dest_per_pq, ori_per_pq, dest_len_pq, ori_len_pq, n_lines, p, b, a):
    d_sum = np.zeros(2)
    w_sum = 0
    for k in range(n_lines):
        dest_xp_k = dest_x - dest_p[k]
        u = np.abs(np.dot(dest_xp_k, dest_pq[k]) / np.square(dest_len_pq[k]))
        v = np.abs(np.dot(dest_xp_k, dest_per_pq[k]) / dest_len_pq[k])
        ori_x_k = ori_p[k] + u * ori_pq[k] + v * ori_per_pq[k] / ori_len_pq[k]
        dk = ori_x_k - dest_x
        dist = v
        weight = (dest_len_pq[k] ** p / (a + dist)) ** b
        d_sum += dk * weight
        w_sum += weight
    ori_x = dest_x + d_sum / w_sum
    return ori_x


# %%
p_value, b, a = 0.5, 1.5, 0.001
rows, cols, _ = buf_shape

for dest in range(1, n_images):
    if dest == 2:
        break
    ori = dest - 1
    dest_p = p[:, dest]
    ori_p = p[:, ori]
    dest_pq = pq[:, dest]
    ori_pq = pq[:, ori]
    dest_per_pq = per_pq[:, dest]
    ori_per_pq = per_pq[:, ori]
    dest_len_pq = len_pq[:, dest]
    ori_len_pq = len_pq[:, ori]

    for i in range(rows):
        for j in range(cols):
            print(j, i)
            dest_x = np.array([j, i])
            ori_x = warping(dest_x, dest_p, ori_p, dest_pq, ori_pq, dest_per_pq, ori_per_pq, dest_len_pq, ori_len_pq,
                            n_lines, p_value, b, a)

            ori_x_x, ori_x_y = ori_x
            floor_y = int(np.floor(ori_x_y)) # p
            floor_x = int(np.floor(ori_x_x))  # q
            g = ori_x_y - floor_y  # a
            h = ori_x_x - floor_x  # b

            if floor_y < 0 or floor_x < 0 or floor_y + 1 >= ori_shape[0] or floor_x + 1 >= ori_shape[1]:
                buf_images[dest, i, j] = 0
            else:
                buf_images[dest, i, j] = (1 - g) * (
                        (1 - h) * img_origin[floor_y, floor_x] + h * img_origin[floor_y, floor_x + 1]) * g * (
                        (1 - h) * img_origin[floor_y + 1, floor_x] + h * img_origin[floor_y + 1, floor_x + 1])

# %%
plt.imshow(buf_images[1])
