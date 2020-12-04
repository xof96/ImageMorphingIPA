from utils.utils import imread
import numpy as np
import matplotlib.pyplot as plt

img_o_path = 'assets/glass/vaso.jpg'
img_d_path = 'assets/glass/copa.jpg'

img_origin = imread(filename=img_o_path)
if img_origin.shape[2] > 3:
    img_origin = img_origin[:, :, 0:3]
img_destination = imread(filename=img_d_path)

o_shape = img_origin.shape
d_shape = img_destination.shape
# %%
t = 0.5
buf_shape = []
for i in range(3):
    buf_shape.append(max(o_shape[i], d_shape[i]))
buf_img = np.zeros(shape=buf_shape, dtype=np.uint8)

# if __name__ == '__main__':
# %%
n_lines = 0
pa = []
qa = []
pb = []
qb = []
with open('data/glass_lines.txt', 'r') as ref_lines:
    lines = ref_lines.readlines()
    for line in lines:
        n_lines += 1
        if line[-1] == '\n':
            line = line[0: -1]
        coords = line.split(',')
        curr_pa = (int(coords[0]), int(coords[1]))
        curr_qa = (int(coords[2]), int(coords[3]))
        curr_pb = (int(coords[4]), int(coords[5]))
        curr_qb = (int(coords[6]), int(coords[7]))
        pa.append(curr_pa)
        qa.append(curr_qa)
        pb.append(curr_pb)
        qb.append(curr_qb)

# %%
# Calculating PQ and PQ'
pqa = []
pqb = []
for i in range(n_lines):
    pqa.append((qa[i][0] - pa[i][0], qa[i][1] - pa[i][1]))
    pqb.append((qb[i][0] - pb[i][0], qb[i][1] - pb[i][1]))

per_pqa = []
per_pqb = []
for i in range(n_lines):
    per_pqa.append((pqa[i][1], -1 * pqa[i][0]))
    per_pqb.append((pqb[i][1], -1 * pqb[i][0]))

# %%
len_pqa = []
len_pqb = []
for i in range(n_lines):
    len_pqa.append(np.sqrt(pqa[i][0] ** 2 + pqa[i][1] ** 2))
    len_pqb.append(np.sqrt(pqb[i][0] ** 2 + pqb[i][1] ** 2))

# %%
p, b, a = 0.5, 1.5, 0.001
rows, cols, _ = buf_shape
for i in range(rows):
    for j in range(cols):
        d_sum_x = 0
        d_sum_y = 0
        w_sum = 0
        xb_x = j
        xb_y = i
        for k in range(n_lines):
            xpb_k = (xb_x - pb[k][0], xb_y - pb[k][1])
            u = np.dot(xpb_k, pqb[k]) / (len_pqb[k] ** 2)
            v = np.dot(xpb_k, per_pqb[k]) / len_pqb[k]
            xa_x = pa[k][0] + u * pqa[k][0] + v * per_pqa[k][0] / len_pqa[k]
            xa_y = pa[k][1] + u * pqa[k][1] + v * per_pqa[k][1] / len_pqa[k]
            di_x = xb_x - xa_x
            di_y = xb_y - xa_y
            dist = v
            weight = (len_pqb[k] ** p / (a + dist)) ** b
            d_sum_x += di_x * weight
            d_sum_y += di_y * weight
        xa_x = xb_x + d_sum_x / w_sum
        xa_y = xb_y + d_sum_y / w_sum

        fy = np.floor(xa_y)  # p
        fx = np.floor(xa_x)  # q
        g = xa_y - fy  # a
        h = xa_x - fx  # b

        if fy < 0 or fx < 0 or fy + 1 > o_shape[0] or fx + 1 > o_shape[1]:
            buf_img[xb_x, xb_y] = 0
        else:
            buf_img[xb_x, xb_y] = (1 - g) * ((1 - h) * img_origin[fy, fx] + h * img_origin[fy, fx + 1]) * g * (
                    (1 - h) * img_origin[fy + 1, fx] + h * img_origin[fy + 1, fx + 1])
