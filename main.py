from utils.utils import imread, warping, is_inside_img, bi_linear_interpolation
import numpy as np
import argparse
import cv2
import matplotlib.pyplot as plt

if __name__ == '__main__':
    # We get the arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-image_a_path', type=str)
    parser.add_argument('-image_b_path', type=str)
    parser.add_argument('-ref_lines_path', type=str)
    parser.add_argument('-N', type=int)
    args = parser.parse_args()
    if not args.image_a_path:
        parser.error('-image_a_path is required.')
    if not args.image_b_path:
        parser.error('-image_b_path is required.')
    if not args.ref_lines_path:
        parser.error('-ref_lines_path is required.')
    if not args.N:
        parser.error('-N is required')

    image_a_path = args.image_a_path
    image_b_path = args.image_b_path
    ref_lines_path = args.ref_lines_path

    # Reading the images
    image_a = imread(filename=image_a_path)
    image_b = imread(filename=image_b_path)
    if image_a.shape[2] > 3:
        image_a = image_a[:, :, 0:3]
    if image_b.shape[2] > 3:
        image_a = image_a[:, :, 0:3]

    shape_a = image_a.shape
    shape_b = image_b.shape

    # n_images = args.N
    n_images = 11
    buf_shape = []
    for i in range(3):
        buf_shape.append(max(shape_a[i], shape_b[i]))

    from_a_images = np.zeros(shape=[n_images] + buf_shape, dtype=np.uint8)
    from_b_images = np.zeros(shape=[n_images] + buf_shape, dtype=np.uint8)
    from_a_images[0] = image_a
    from_b_images[-1] = image_b

    dt = 1 / (n_images - 1)  # We subtract 1 not to count the destination image.
    with open(ref_lines_path, 'r') as ref_lines:
        lines = ref_lines.readlines()
        if lines[-1] == '':
            lines.pop()

    n_lines = len(lines)
    pa = np.zeros(shape=(n_lines, 2))
    qa = np.zeros(shape=(n_lines, 2))
    pb = np.zeros(shape=(n_lines, 2))
    qb = np.zeros(shape=(n_lines, 2))

    for i in range(n_lines):
        if lines[i][-1] == '\n':
            lines[i] = lines[i][0: -1]
        coords = lines[i].split(',')
        pa[i] = (int(coords[0]), int(coords[1]))
        qa[i] = (int(coords[2]), int(coords[3]))
        pb[i] = (int(coords[4]), int(coords[5]))
        qb[i] = (int(coords[6]), int(coords[7]))

    t = np.linspace(0, 1, n_images)
    p = np.zeros(shape=(n_images, n_lines, 2))
    q = np.zeros(shape=(n_images, n_lines, 2))

    for i in range(n_images):
        p[i] = (pa * (1 - t[i]) + pb * t[i]).astype('int')
        q[i] = (qa * (1 - t[i]) + qb * t[i]).astype('int')

    # Calculating PQs
    pq = q - p

    # Perpendicular
    per_pq = np.zeros(shape=(n_images, n_lines, 2))
    per_pq[:, :, 0] = pq[:, :, 1]
    per_pq[:, :, 1] = -1 * pq[:, :, 0]

    # LEN
    len_pq = np.sqrt(np.square(pq[:, :, 0]) + np.square(pq[:, :, 1]))

    p_value, b, a = 1, 1, 0.001
    rows, cols, _ = buf_shape

    # Creating the images
    for curr_i in range(1, n_images - 1):
        print("Working on image", curr_i)
        img_a_i = 0
        img_b_i = n_images - 1
        target_p = p[curr_i]
        target_q = q[curr_i]
        ori_p = p[img_a_i]
        dest_p = p[img_b_i]
        target_pq = pq[curr_i]
        ori_pq = pq[img_a_i]
        dest_pq = pq[img_b_i]
        target_per_pq = per_pq[curr_i]
        ori_per_pq = per_pq[img_a_i]
        dest_per_pq = per_pq[img_b_i]
        target_len_pq = len_pq[curr_i]
        ori_len_pq = len_pq[img_a_i]
        dest_len_pq = len_pq[img_b_i]

        for i in range(rows):
            for j in range(cols):
                target_x = np.array([j, i])

                # Warping
                image_a_x = warping(dest_x=target_x, dest_q=target_q, dest_p=target_p, ori_p=ori_p, dest_pq=target_pq,
                                    ori_pq=ori_pq, dest_per_pq=target_per_pq, ori_per_pq=ori_per_pq,
                                    dest_len_pq=target_len_pq, ori_len_pq=ori_len_pq, n_lines=n_lines, p=p_value, b=b,
                                    a=a)
                image_b_x = warping(dest_x=target_x, dest_q=target_q, dest_p=target_p, ori_p=dest_p, dest_pq=target_pq,
                                    ori_pq=dest_pq, dest_per_pq=target_per_pq, ori_per_pq=dest_per_pq,
                                    dest_len_pq=target_len_pq, ori_len_pq=dest_len_pq, n_lines=n_lines, p=p_value, b=b,
                                    a=a)

                # Interpolation
                image_a_x_x, image_a_x_y = image_a_x
                image_b_x_x, image_b_x_y = image_b_x

                image_a_floor_y = int(np.floor(image_a_x_y))  # p
                image_b_floor_y = int(np.floor(image_b_x_y))  # p

                image_a_floor_x = int(np.floor(image_a_x_x))  # q
                image_b_floor_x = int(np.floor(image_b_x_x))  # q

                image_a_g = image_a_x_y - image_a_floor_y  # a
                image_b_g = image_b_x_y - image_b_floor_y  # a

                image_a_h = image_a_x_x - image_a_floor_x  # b
                image_b_h = image_b_x_x - image_b_floor_x  # b

                if is_inside_img(x=image_a_x_x, y=image_a_x_y, shape=buf_shape):
                    o_ix = 0
                else:
                    o_ix = bi_linear_interpolation(p=image_a_floor_y, q=image_a_floor_x, a=image_a_g, b=image_a_h,
                                                   img=from_a_images[img_a_i])

                if is_inside_img(x=image_b_x_x, y=image_b_x_y, shape=buf_shape):
                    d_ix = 0
                else:
                    d_ix = bi_linear_interpolation(p=image_b_floor_y, q=image_b_floor_x, a=image_b_g, b=image_b_h,
                                                   img=from_b_images[img_b_i])

                # Painting pixel
                from_a_images[curr_i, i, j] = o_ix
                from_b_images[curr_i, i, j] = d_ix

    # Making final images
    final_images = np.zeros(shape=[n_images] + buf_shape, dtype=np.uint8)

    for n_im in range(n_images):
        final_images[n_im] = from_a_images[n_im] * (1 - t[n_im]) + from_b_images[n_im] * t[n_im]

    print("Saving images...")
    for i in range(n_images):
        plt.imsave(f'res/im_{i}.jpg', final_images[i])

    frame_size = (buf_shape[0], buf_shape[1])
    out = cv2.VideoWriter('output_video.avi', cv2.VideoWriter_fourcc(*'DIVX'), 10, frame_size)

    for im in final_images:
        out.write(im)

    print("Video saved")
    out.release()
