from utils.utils import imread
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

# Calcuating PQ and PQ'
pqa = []
pqb = []
for i in range(n_lines):
    pqa.append((qa[i][0] - pa[i][0], qa[i][1] - pa[i][1]))
    pqb.append((qb[i][0] - pb[i][0], qb[i][1] - pb[i][1]))

rows, cols, _ = buf_shape
for i in range(rows):
    for j in range(cols):
