import numpy as np
import cv2
import math
from numba import njit
import time
import os


# testImg = Image.open('example.jpg')
def gradient_abs_value_mask(image, sobel_kernel=3, axis='x', threshold=120):
    if axis == 'x':
        sobel = np.absolute(cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=sobel_kernel))
    if axis == 'y':
        sobel = np.absolute(cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=sobel_kernel))
    sobel = np.uint8(255 * sobel / np.max(sobel))
    mask = np.zeros_like(sobel)
    mask[(sobel >= threshold)] = 1
    return mask


@njit(parallel=True)
def gradient_magnitude_mask(sobel_x, sobel_y, threshold=120):
    magnitude = np.sqrt(sobel_x ** 2 + sobel_y ** 2)
    magnitude = (magnitude * 255 / np.max(magnitude)).astype(np.uint8)
    mask = np.zeros_like(magnitude)
    for i in range(0, len(magnitude)):
        for j in range(0, len(magnitude[i])):
            if magnitude[i][j] >= threshold:
                mask[i][j] = 1
    return mask


@njit(parallel=True)
def gradient_direction_mask(sobel_x, sobel_y, threshold=(0, np.pi / 2)):
    direction = np.arctan2(sobel_y, sobel_x) + np.pi
    return direction


@njit(parallel=True)
def color_threshold_mask(image, threshold=120):
    mask = np.zeros_like(image)
    for i in range(0, len(mask)):
        for j in range(0, len(mask[i])):
            if image[i][j] < threshold:
                mask[i][j] = 1
    return mask


@njit(parallel=True)
def calc_frame_stat(frame_array):
    average_value = np.average(frame_array)
    print(f'average value: {average_value}')
    median_value = np.median(frame_array)
    print(f'median value: {median_value}')
    mean_value = np.mean(frame_array)
    print(f'mean value: {mean_value}')
    max_value = np.amax(frame_array)
    print(f'max value: {max_value}')
    min_value = np.amin(frame_array)
    print(f'min value: {min_value}')


def get_edges(image, separate_channels=False, main_threshold=90, horizontal_threshold=0, color_threshold=120,
              kernel_size=3):
    hls = cv2.cvtColor(np.copy(image), cv2.COLOR_RGB2HLS).astype(np.float)
    s_channel = hls[:, :, 1]
    sobel_x = cv2.Sobel(s_channel, cv2.CV_64F, 1, 0, ksize=kernel_size)
    sobel_y = cv2.Sobel(s_channel, cv2.CV_64F, 0, 1, ksize=kernel_size)

    # magnitude = gradient_magnitude_mask(sobel_x, sobel_y, threshold=main_threshold)

    direction_mask = gradient_direction_mask(sobel_x, sobel_y)

    # gradient_mask = np.zeros_like(s_channel)
    # # gradient_mask[(magnitude == 1)] = 1
    # color_mask = color_threshold_mask(s_channel, threshold=color_threshold)
    # direction_mask[gradient_mask != 1] = 0
    mask = None
    # if separate_channels:
    #     return np.dstack((np.zeros_like(s_channel), gradient_mask, color_mask))
    # else:
    # mask = np.zeros_like(gradient_mask)
    # mask[(gradient_mask == 1)] = 1
    return mask, direction_mask


@njit(parallel=True, nopython=False)
def create_frame_from_array(array):
    row_numbers = len(array)
    column_numbers = len(array[0])
    frame = np.zeros((row_numbers, column_numbers), dtype='uint8')
    for i in range(0, row_numbers):
        for j in range(0, column_numbers):
            if array[i, j] == 0.0:
                frame[i, j] = 0
            else:
                frame[i, j] = 255
    return frame


@njit(parallel=True)
def create_threshold_variative_video_by_frame():
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = 1.0
    videoReader = cv2.VideoCapture('example.mp4')
    videoWriter = cv2.VideoWriter('frame_test_lightness_1_20_with_frame_buffer.mp4', fourcc, fps, (1280, 400), False)
    frame_number_target = 3570
    frame_number = 0
    success = True
    while success:
        frame_number = frame_number + 1
        success, frame = videoReader.read()
        if frame_number != frame_number_target:
            continue
        if frame_number == frame_number_target:
            for main_threshold in range(1, 20, 1):
                frame_array = get_edges(frame, main_threshold=main_threshold)
                result_frame = create_frame_from_array(frame_array)
                print(main_threshold)
                videoWriter.write(result_frame)
            videoWriter.release()
            break


@njit(parallel=True)
def filter_edges_by_history(frame_history_array, buffer_value=3):
    frame_t = np.zeros_like(frame_history_array[0])
    for i in range(0, len(frame_history_array[0])):
        for j in range(0, len(frame_history_array[0][0])):
            for frame_index in range(len(frame_history_array)):
                if frame_history_array[frame_index][i][j] == 1.0:
                    frame_t[i][j] = frame_t[i][j] + 1
    filtered_frame_from_array_t = np.zeros_like(frame_t)
    for i in range(0, len(frame_t)):
        for j in range(0, len(frame_t[0])):
            if frame_t[i][j] >= buffer_value:
                filtered_frame_from_array_t[i][j] = 1.0
    return filtered_frame_from_array_t


@njit(parallel=True)
def get_projection(frame):
    src = np.float32([[500, 400], [670, 400], [55, 607], [1067, 607]])
    dist = np.float32([[100, 0], [1180, 0], [100, 720], [1180, 720]])
    matrix = cv2.getPerspectiveTransform(src, dist)
    proj_frame = cv2.warpPerspective(frame, matrix, (1280, 720))
    return proj_frame


@njit(parallel=True)
def create_cluster_frame(direction_mask, cluster_count, angle_shift=False):
    cluster_mask = np.zeros_like(direction_mask)
    cluster_size = 2 * np.pi / cluster_count

    if angle_shift:
        start_angle = np.pi / 2 - cluster_size / 2
    else:
        start_angle = np.pi / 2

    cluster_angles = []
    for i in range(0, cluster_count):
        cluster_angles.append([start_angle + cluster_size * i, start_angle + cluster_size * (i + 1)])

    for row_index in range(0, len(direction_mask)):
        for column_index in range(0, len(direction_mask[row_index])):
            for i in range(0, cluster_count):
                angle = direction_mask[row_index][column_index]
                if cluster_angles[i][0] <= angle < cluster_angles[i][1] or \
                        cluster_angles[i][0] <= angle + np.pi * 2 < cluster_angles[i][1]:
                    cluster_mask[row_index][column_index] = i + 1
                    break

    return cluster_mask


@njit(parallel=True)
def create_image_from_cluster(cluster_mask, horizontal_size=0, vertical_size=720, merging=False, input_img=None):
    if not merging:
        img = np.zeros((vertical_size, horizontal_size, 3), np.uint8)
        for row_index in range(0, len(cluster_mask)):
            for column_index in range(0, len(cluster_mask[row_index])):
                value = cluster_mask[row_index][column_index]
                if value == 1.0:
                    img[row_index][column_index] = (0, 0, 255)
                    continue
                if value == 2.0:
                    img[row_index][column_index] = (0, 255, 0)
                    continue
                if value == 3.0:
                    img[row_index][column_index] = (255, 0, 0)
                    continue
                if value == 4.0:
                    img[row_index][column_index] = (0, 255, 255)
                    continue
                if value == 5.0:
                    img[row_index][column_index] = (255, 0, 255)
                    continue
                if value == 6.0:
                    img[row_index][column_index] = (255, 255, 255)
                    continue
                if value == 7.0:
                    img[row_index][column_index] = (255, 255, 0)
                    continue
                if value == 8.0:
                    img[row_index][column_index] = (128, 128, 0)
                    continue
        return img
    else:
        merge_img = np.copy(input_img)
        vertical_parts_count = 16
        horizontal_parts_count = 32
        vertical_step = vertical_size // vertical_parts_count
        vertical_indexes = []
        horizontal_indexes = []
        horizontal_step = horizontal_size // horizontal_parts_count
        for i in range(0, vertical_parts_count):
            vertical_indexes.append((vertical_step * i, vertical_step * (i + 1)))
        vertical_indexes.append((vertical_step * vertical_parts_count,
                                 vertical_step * vertical_parts_count + vertical_size % vertical_step))
        for i in range(0, horizontal_parts_count):
            horizontal_indexes.append((horizontal_step * i, horizontal_step * (i + 1)))
        horizontal_indexes.append((horizontal_step * horizontal_parts_count,
                                   horizontal_step * horizontal_parts_count + horizontal_size % horizontal_step))

        cluster_check_mask = []
        for i in range(0, vertical_parts_count + 1):
            row = []
            for j in range(0, horizontal_parts_count + 1):
                row.append(is_cluster_part_valid(vertical_indexes[i], horizontal_indexes[j],
                                                 cluster_mask, 40))
            cluster_check_mask.append(row)

        for row_index in range(1, len(cluster_mask)):
            for column_index in range(1, len(cluster_mask[row_index]) - 1):

                vertical_mask_index = row_index // vertical_step
                horizontal_mask_index = column_index // horizontal_step
                is_any_cluster_valid = cluster_check_mask[vertical_mask_index][horizontal_mask_index][0] or \
                                       cluster_check_mask[vertical_mask_index][horizontal_mask_index][1]
                if not is_any_cluster_valid:
                    continue
                value = cluster_mask[row_index][column_index]

                if value == 2.0 and cluster_check_mask[vertical_mask_index][horizontal_mask_index][0] \
                        and (cluster_check_mask[vertical_mask_index - 1][horizontal_mask_index - 1][0] \
                             or cluster_check_mask[vertical_mask_index - 1][horizontal_mask_index][0] \
                             or cluster_check_mask[vertical_mask_index - 1][horizontal_mask_index - 1][0]):
                    merge_img[row_index][column_index] = (0, 255, 0)
                    continue
                if value == 4.0 and cluster_check_mask[vertical_mask_index][horizontal_mask_index][1]:
                    merge_img[row_index][column_index] = (0, 255, 255)
                    continue
        return merge_img


@njit(parallel=True)
def is_cluster_part_valid(vertical_indexes, horizontal_indexes, cluster, threshold=1000):
    first_cluster_count = 0
    second_cluster_count = 0
    for i in range(vertical_indexes[0], vertical_indexes[1]):
        for j in range(horizontal_indexes[0], horizontal_indexes[1]):
            cluster_index = cluster[i][j]
            if cluster_index == 2.0:
                first_cluster_count += 1
            if cluster_index == 4.0:
                second_cluster_count += 1
    return (first_cluster_count > threshold and first_cluster_count > second_cluster_count,
            second_cluster_count > threshold and second_cluster_count > first_cluster_count)


# create_threshold_variative_video_by_frame()
# result = get_edges(testImg)
# plt.imshow(result)
# plt.savefig("array")
vidcap = cv2.VideoCapture('example_3.avi')
success = True
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
frame_buffer = []
fps = 29.97
render_seconds_count = 200
render_start_second_number = 0

time_start = time.time()
kernel_size = 7
for clusters_count in range(4, 5):
    frame_number = 0
    videoEdge = cv2.VideoWriter(f'filtration_cluster_by_scanning.mp4', fourcc, fps, (640, 134))

    while success:
        print(f'frames left: {(render_start_second_number + render_seconds_count) * 30 - frame_number}')
        frame_number = frame_number + 1
        success, frame_full = vidcap.read()

        if (render_start_second_number + render_seconds_count) * 30 < frame_number:
            break
        if frame_number < render_start_second_number * 30:
            continue
        frame_time = time.time()
        frame_part = frame_full[366:633]
        frame = cv2.resize(frame_part, (640, 134), cv2.INTER_AREA)
        edges, direction_mask = get_edges(frame, main_threshold=0, kernel_size=kernel_size)
        # # img = create_image_from_cluster(create_cluster_frame(direction_mask))
        # if len(frame_buffer) < 5:
        #     frame_buffer.append(edges)
        #     continue
        # if len(frame_buffer) >= 5:
        #     frame_buffer.append(edges)
        #     frame_buffer.pop(0)
        #     filtered_frame_from_array = filter_edges_by_history(frame_buffer, buffer_value=4)
        #     rgbEdges = create_frame_from_array(filtered_frame_from_array)
        #     direction_mask[rgbEdges == 0] = 0
        cluster_mask = create_cluster_frame(direction_mask, clusters_count, angle_shift=True)
        img = create_image_from_cluster(cluster_mask, vertical_size=134, horizontal_size=640, merging=True,
                                        input_img=frame)
        end_frame_time = time.time() - frame_time
        print(f'frames time{end_frame_time}')
        videoEdge.write(img)
        # projection = get_projection(rgbEdges)
    print(f'average frame time is {(time.time() - time_start) / (render_seconds_count * fps)}')
    print('start releasing')
    videoEdge.release()
print('finished')
