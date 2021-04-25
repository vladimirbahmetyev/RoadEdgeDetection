import math
import time

import cv2
import numpy as np
from numba import njit


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
def gradient_direction_mask(sobel_x, sobel_y, threshold=10):
    direction = np.arctan2(sobel_y, sobel_x) + np.pi
    filter_by_gradient = gradient_magnitude_mask(sobel_x, sobel_y, threshold=threshold)
    for row_index in range(0, len(direction)):
        for column_index in range(0, len(direction[row_index])):
            if filter_by_gradient[row_index][column_index] == 0:
                direction[row_index][column_index] = np.pi / 2
    return direction

def get_edges(image, separate_channels=False, main_threshold=90, horizontal_threshold=0, color_threshold=120,
              kernel_size=3):
    hls = cv2.cvtColor(np.copy(image), cv2.COLOR_RGB2HLS)
    s_channel = hls[:, :, 1]
    sobel_x = cv2.Sobel(s_channel, cv2.CV_64F, 1, 0, ksize=kernel_size)
    sobel_y = cv2.Sobel(s_channel, cv2.CV_64F, 0, 1, ksize=kernel_size)
    direction_mask = gradient_direction_mask(sobel_x, sobel_y, threshold=main_threshold)
    mask = None
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
        checked_cluster = filtrate_angle_cluster_by_group(cluster_mask, threshold=400)
        with_lines = filter_frame_by_lines(checked_cluster, threshold=100)
        merge_img = np.copy(input_img)

        for row_index in range(1, len(cluster_mask)):
            for column_index in range(1, len(cluster_mask[row_index]) - 1):
                value = with_lines[row_index][column_index]
                if value == 2.0:
                    merge_img[row_index][column_index] = (0, 255, 0)
                    continue

                if value == 1.0:
                    merge_img[row_index][column_index] = (0, 0, 255)
                    continue

                if value == 4.0:
                    merge_img[row_index][column_index] = (0, 255, 255)
                    continue
        return merge_img


@njit(parallel=True)
def filtrate_angle_cluster_by_group(matrix, threshold=10):
    matrix_copy = np.copy(matrix)
    current_cluster_index_first = 1
    first_cluster_count = []
    current_cluster_index_second = 1
    second_cluster_count = []
    matrix_with_clusters_first = np.zeros_like(matrix)
    matrix_with_clusters_second = np.zeros_like(matrix)
    for i in range(1, len(matrix) - 1):
        for j in range(1, len(matrix[0]) - 1):
            if matrix[i][j] == 2.0:
                is_cluster_detected = False
                for string_index in range(-1, 2, 1):
                    cluster_index = int(matrix_with_clusters_first[i - 1][j + string_index])
                    if cluster_index > 0:
                        matrix_with_clusters_first[i][j] = cluster_index
                        first_cluster_count[cluster_index - 1] += 1
                        is_cluster_detected = True
                        continue
                    prev_cluster_index = int(matrix_with_clusters_first[i][j - 1])
                    if prev_cluster_index > 0:
                        matrix_with_clusters_first[i][j] = prev_cluster_index
                        first_cluster_count[prev_cluster_index - 1] += 1
                        is_cluster_detected = True
                        continue
                if is_cluster_detected:
                    continue
                first_cluster_count.append(1)
                matrix_with_clusters_first[i][j] = current_cluster_index_first
                current_cluster_index_first += 1

            if matrix[i][j] == 4.0:
                is_cluster_detected = False
                for string_index in range(-1, 2, 1):
                    cluster_index = int(matrix_with_clusters_second[i - 1][j + string_index])
                    prev_cluster_index = int(matrix_with_clusters_second[i][j - 1])
                    if cluster_index > 0:
                        matrix_with_clusters_second[i][j] = cluster_index
                        second_cluster_count[cluster_index - 1] += 1
                        is_cluster_detected = True
                        continue
                    if prev_cluster_index > 0:
                        matrix_with_clusters_second[i][j] = prev_cluster_index
                        second_cluster_count[prev_cluster_index - 1] += 1
                        is_cluster_detected = True
                        continue
                if is_cluster_detected:
                    continue
                second_cluster_count.append(1)
                matrix_with_clusters_second[i][j] = current_cluster_index_second
                current_cluster_index_second += 1
    for i in range(0, len(matrix)):
        for j in range(0, len(matrix[0])):
            if matrix[i][j] == 2.0:
                checking_index = int(matrix_with_clusters_first[i][j])
                if checking_index > 0:
                    if first_cluster_count[checking_index - 1] < threshold:
                        matrix_copy[i][j] = 0
            if matrix[i][j] == 4.0:
                checking_index = int(matrix_with_clusters_second[i][j])
                if checking_index > 0:
                    if second_cluster_count[checking_index - 1] < threshold:
                        matrix_copy[i][j] = 0
    return matrix_copy


@njit(parallel=True)
def filter_frame_by_lines(matrix_with_pixels, threshold=100):
    angle_range = [0, np.pi / 6, 5 * np.pi / 6, np.pi]
    angle_step = np.pi / 32
    rad_range = 5
    angles_number = int((angle_range[1] - angle_range[0]) // angle_step)
    height = len(matrix_with_pixels)
    width = len(matrix_with_pixels[0])
    max_rad = math.sqrt(height ** 2 + width ** 2)
    rad_number = int(max_rad // rad_range)
    green_matrix = np.zeros((angles_number * 2, rad_number + 1))
    yellow_matrix = np.zeros((angles_number * 2, rad_number + 1))
    for i in range(0, height):
        for j in range(0, width):
            checking_value = matrix_with_pixels[i][j]

            if checking_value != 2 and checking_value != 4:
                continue
            for phi_index in range(0, angles_number):
                phi = angle_range[0] + phi_index * angle_step
                rad = (j + 1) * math.cos(phi) + (i + 1) * math.sin(phi)
                rad_class = int(rad // rad_range)
                if checking_value == 2.0:
                    green_matrix[phi_index, rad_class] += 1
                else:
                    yellow_matrix[phi_index, rad_class] += 1
            for phi_index in range(0, angles_number):
                phi = angle_range[2] + phi_index * angle_step
                rad = (j + 1) * math.cos(phi) + (i + 1) * math.sin(phi)
                rad_class = int(rad // -rad_range)
                if checking_value == 2.0:
                    green_matrix[phi_index + angles_number, rad_class] += 1
                else:
                    yellow_matrix[phi_index + angles_number, rad_class] += 1
    filtered_mask = np.zeros_like(matrix_with_pixels)
    for i in range(0, height):
        for j in range(0, width):
            checking_value = matrix_with_pixels[i][j]
            if checking_value != 2 and checking_value != 4 and checking_value != 1:
                continue
            for phi_index in range(0, angles_number):
                phi = angle_range[0] + phi_index * angle_step
                rad = (j + 1) * math.cos(phi) + (i + 1) * math.sin(phi)
                rad_class = int(rad // rad_range)
                if yellow_matrix[phi_index, rad_class] > threshold and checking_value == 4:
                    filtered_mask[i][j] = 4.0
                    break
                elif green_matrix[phi_index, rad_class] > threshold and checking_value == 2:
                    filtered_mask[i][j] = 2.0
                    break
                elif green_matrix[phi_index, rad_class] > threshold or green_matrix[phi_index, rad_class] > threshold:
                    filtered_mask[i][j] = 1.0
                    break
            for phi_index in range(0, angles_number):
                phi = angle_range[2] + phi_index * angle_step
                rad = (j + 1) * math.cos(phi) + (i + 1) * math.sin(phi)
                rad_class = int(rad // -rad_range)
                if yellow_matrix[phi_index + angles_number, rad_class] > threshold and checking_value == 4:
                    filtered_mask[i][j] = 4.0
                    break
                elif green_matrix[phi_index + angles_number, rad_class] > threshold and checking_value == 2:
                    filtered_mask[i][j] = 2.0
                    break
                elif green_matrix[phi_index, rad_class] > threshold or green_matrix[phi_index, rad_class] > threshold:
                    filtered_mask[i][j] = 1.0
                    break
    return filtered_mask

video_reader = cv2.VideoCapture('example_3.avi')
success = True
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
frame_buffer = []
fps = 29.97
render_seconds_count = 500
render_start_second_number = 0
filtered_frame_from_array = []
time_start = time.time()
kernel_size = 7
frame_number = 0
video_writer = cv2.VideoWriter(f'testing_smart_filtration_lines_every_2.mp4', fourcc, fps, (640, 134))
while success:
    print(f'frames left: {(render_start_second_number + render_seconds_count) * 30 - frame_number}')
    frame_number = frame_number + 1
    success, frame_full = video_reader.read()
    if not success:
        break
    if (render_start_second_number + render_seconds_count) * 30 < frame_number:
        break
    if frame_number < render_start_second_number * 30:
        continue
    frame_time = time.time()
    frame_part = frame_full[360:633]
    frame = cv2.resize(frame_part, (640, 134), cv2.INTER_AREA)
    edges, direction_mask = get_edges(frame, main_threshold=0, kernel_size=kernel_size)
    cluster_mask = create_cluster_frame(direction_mask, 4, angle_shift=True)

    img = create_image_from_cluster(cluster_mask, vertical_size=134, horizontal_size=640,
                                    merging=True,
                                    input_img=frame)
    end_frame_time = time.time() - frame_time
    print(f'frames time{end_frame_time}')
    video_writer.write(img)
print(f'average frame time is {(time.time() - time_start) / (render_seconds_count * fps)}')
print('start releasing')
video_writer.release()
print('finished')
