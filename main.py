import math
import time
import matplotlib.pyplot as plt

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
    l_channel = hls[:, :, 1]
    sobel_x = cv2.Sobel(l_channel, cv2.CV_64F, 1, 0, ksize=kernel_size)
    sobel_y = cv2.Sobel(l_channel, cv2.CV_64F, 0, 1, ksize=kernel_size)
    direction_mask = gradient_direction_mask(sobel_x, sobel_y, threshold=main_threshold)
    mask = None
    return mask, direction_mask


@njit(parallel=True)
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
def create_class_frame(direction_mask, cluster_count=4, angle_shift=False, angle_shift_size=0):
    class_mask = np.zeros_like(direction_mask)
    class_size = 2 * np.pi / cluster_count

    if angle_shift:
        start_angle = np.pi / 4 + angle_shift_size
    else:
        start_angle = np.pi / 4

    class_angles = []
    for class_index in range(0, cluster_count):
        class_angles.append(
            [start_angle + class_size * class_index, start_angle + class_size * (class_index + 1)])
    for row_index in range(0, len(direction_mask)):
        for column_index in range(0, len(direction_mask[row_index])):
            for class_index in range(0, cluster_count):
                angle = direction_mask[row_index][column_index]
                if class_angles[class_index][0] <= angle < class_angles[class_index][1] or \
                        class_angles[class_index][0] <= angle + np.pi * 2 < class_angles[class_index][1]:
                    class_mask[row_index][column_index] = class_index + 1
                    break

    return class_mask


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
                    img[row_index][column_index] = (255, 255, 255)
                    continue
        return img
    else:
        # cluster_time = time.time()
        # print('cluster time{}'.format(time.time() - cluster_time))
        # hough_time = time.time()
        with_lines = filter_frame_by_lines(cluster_mask, threshold=100)
        # print('Hough time: {}'.format(time.time() - hough_time))
        merge_img = np.copy(input_img)

        for row_index in range(1, len(cluster_mask)):
            for column_index in range(1, len(cluster_mask[row_index]) - 1):
                value = with_lines[row_index][column_index]
                if value == 2.0:
                    merge_img[row_index][column_index] = (0, 255, 0)
                    continue
                if value == 4.0:
                    merge_img[row_index][column_index] = (0, 255, 255)
                    continue
        return merge_img


@njit(parallel=True)
def filtrate_angle_cluster_by_group(matrix, min_threshold=10, max_threshold=1000, with_neighbour=False):
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
                    if first_cluster_count[checking_index - 1] < min_threshold:
                        # first_cluster_count[checking_index - 1] > max_threshold:
                        matrix_copy[i][j] = 0
            if matrix[i][j] == 4.0:
                checking_index = int(matrix_with_clusters_second[i][j])
                if checking_index > 0:
                    if second_cluster_count[checking_index - 1] < min_threshold:  # or \
                        # second_cluster_count[checking_index - 1] > max_threshold:
                        matrix_copy[i][j] = 0
    if with_neighbour:
        for i in range(10, len(matrix) - 10):
            for j in range(4, len(matrix[0]) - 10):
                if matrix[i][j] == 2.0:
                    is_value_found = False
                    for checking_row in range(-10, 10, 1):
                        for checking_column in range(-10, 10, 1):
                            if matrix_copy[i + checking_row][j + checking_column] == 4:
                                is_value_found = True
                                break
                        if is_value_found:
                            break
                    if not is_value_found:
                        matrix_copy[i][j] = 0
                if matrix[i][j] == 4.0:
                    is_value_found = False
                    for checking_row in range(-10, 10, 1):
                        for checking_column in range(-10, 10, 1):
                            if matrix_copy[i + checking_row][j + checking_column] == 2:
                                is_value_found = True
                                break
                        if is_value_found:
                            break
                    if not is_value_found:
                        matrix_copy[i][j] = 0
    return matrix_copy


@njit(parallel=True)
def filter_frame_by_lines(matrix_with_pixels, threshold=250):
    return matrix_with_pixels
    angles_number = 3
    angles_size = np.pi / 6
    angle_range = [0, angles_size, np.pi - angles_size, np.pi]
    angle_step = angles_size / angles_number
    rad_range = 50

    height = len(matrix_with_pixels)
    width = len(matrix_with_pixels[0])
    max_rad = math.sqrt(height ** 2 + width ** 2)
    rad_number = int(max_rad // rad_range)
    green_lines_matrix = np.zeros((angles_number * 2, rad_number + 1))
    green_pixel2lines_matrix = []

    yellow_lines_matrix = np.zeros((angles_number * 2, rad_number + 1))
    yellow_pixel2lines_matrix = []
    for i in range(0, height):
        green_pixel2lines_row = []
        yellow_pixel2lines_row = []
        for j in range(0, width):
            checking_value = matrix_with_pixels[i][j]

            if checking_value != 2 and checking_value != 4:
                yellow_pixel2lines_row.append(None)
                green_pixel2lines_row.append(None)
                continue
            green_pixel2lines_cell = []
            yellow_pixel2lines_cell = []
            for phi_index in range(0, angles_number):
                phi = angle_range[0] + phi_index * angle_step
                rad = (j + 1) * math.cos(phi) + (i + 1) * math.sin(phi)
                rad_class = int(rad // rad_range)
                if checking_value == 2.0:
                    green_lines_matrix[phi_index, rad_class] += 1
                    green_pixel2lines_cell.append([phi_index, rad_class])
                else:
                    yellow_lines_matrix[phi_index, rad_class] += 1
                    yellow_pixel2lines_cell.append([phi_index, rad_class])
            for phi_index in range(0, angles_number):
                phi = angle_range[2] + phi_index * angle_step
                rad = (j + 1) * math.cos(phi) + (i + 1) * math.sin(phi)
                rad_class = int(rad // -rad_range)
                if checking_value == 2.0:
                    green_lines_matrix[phi_index + angles_number, rad_class] += 1
                    green_pixel2lines_cell.append([phi_index + angles_number, rad_class])
                else:
                    yellow_lines_matrix[phi_index + angles_number, rad_class] += 1
                    yellow_pixel2lines_cell.append([phi_index + angles_number, rad_class])
            green_pixel2lines_row.append(green_pixel2lines_cell)
            yellow_pixel2lines_row.append(yellow_pixel2lines_cell)
        green_pixel2lines_matrix.append(green_pixel2lines_row)
        yellow_pixel2lines_matrix.append(yellow_pixel2lines_row)

    filtered_mask = np.zeros_like(matrix_with_pixels)
    for i in range(0, height):
        for j in range(0, width):
            checking_value = matrix_with_pixels[i][j]
            if checking_value == 2 or checking_value == 4:
                if checking_value == 2:
                    line_params = green_pixel2lines_matrix[i][j]
                    for param in line_params:
                        [angle, rad] = param
                        if green_lines_matrix[angle][rad] > threshold:
                            filtered_mask[i][j] = 2
                            break
                else:
                    line_params = yellow_pixel2lines_matrix[i][j]
                    for param in line_params:
                        [angle, rad] = param
                        if green_lines_matrix[angle][rad] > threshold:
                            filtered_mask[i][j] = 4
                            break
    return filtered_mask


@njit(parallel=True)
def filter_mask_by_trap(mask):
    height = len(mask)
    width = len(mask[0])
    number_of_points = 0
    threshold_slice_max = 0.33
    threshold_slice_low = 0.3
    angle_threshold_min = 0.25
    left_slice = 0.5
    right_slice = 0.5
    slice_step = int(width / 8)
    current_x_left = int(width / 4)
    current_x_right = int(3 * width / 4)
    for i in range(0, height):
        for j in range(0, width):
            if mask[i][j] != 0:
                number_of_points += 1

    while left_slice > threshold_slice_max or left_slice < threshold_slice_low:
        number_in_right_slice = 0
        for i in range(0, height):
            for j in range(0, current_x_left):
                if mask[i][j] != 0:
                    number_in_right_slice += 1
        left_slice = number_in_right_slice / number_of_points
        if left_slice > threshold_slice_max or left_slice < threshold_slice_low:
            if left_slice > threshold_slice_max:
                current_x_left -= slice_step
                slice_step = int(slice_step / 2)
            else:
                current_x_left += slice_step
                slice_step = int(slice_step / 2)

    slice_step = int(width / 8)
    while right_slice > threshold_slice_max or right_slice < threshold_slice_low:
        number_in_right_slice = 0
        for i in range(0, height):
            for j in range(width - 1, current_x_right, -1):
                if mask[i][j] != 0:
                    number_in_right_slice += 1
        right_slice = number_in_right_slice / number_of_points
        if right_slice > threshold_slice_max or right_slice < threshold_slice_low:
            if right_slice > threshold_slice_max:
                current_x_right += slice_step
                slice_step = int(slice_step / 2)
            else:
                current_x_right -= slice_step
                slice_step = int(slice_step / 2)

    rotate_point_left = height - 1
    left_indexes = []
    rotate_point_right = height - 1
    right_indexes = []
    for i in range(0, height):
        if mask[i][current_x_left + 1] > 0:
            left_indexes.append(i)
        if mask[i][current_x_right - 1] > 0:
            right_indexes.append(i)
    if len(left_indexes) > 0:
        left_sum = 0
        for index in left_indexes:
            left_sum += index
        rotate_point_left = int(left_sum / len(left_indexes))
    if len(right_indexes) > 0:
        right_sum = 0
        for index in left_indexes:
            right_sum += index
        rotate_point_right = int(right_sum / len(right_indexes))

    rotate_angle_left = -np.pi / 4
    rotate_angle_step_left = np.pi / 8

    rotate_angle_right = np.pi / 4
    rotate_angle_step_right = np.pi / 8

    for i in range(0, 4):
        number_of_points_left = 0
        k = math.atan(rotate_angle_left)
        b = rotate_point_left - k * current_x_left
        for row_index in range(0, min(int(b), height)):
            for column_index in range(0, int((row_index - b) / k) - 1):
                if mask[row_index][column_index] != 0:
                    number_of_points_left += 1
        if number_of_points_left / number_of_points < angle_threshold_min:
            rotate_angle_left += rotate_angle_step_left
            rotate_angle_step_left /= 2
        else:
            rotate_angle_left -= rotate_angle_step_left
            rotate_angle_step_left /= 2

    for i in range(0, 4):
        number_of_points_right = 0
        k = math.atan(rotate_angle_right)
        b = rotate_point_right - k * current_x_right
        for row_index in range(0, min(int(width * k + b), height)):
            for column_index in range(int((row_index - b) / k), width):
                if mask[row_index][column_index] != 0:
                    number_of_points_right += 1
        if number_of_points_right / number_of_points < angle_threshold_min:
            rotate_angle_right -= rotate_angle_step_right
            rotate_angle_step_right /= 2
        else:
            rotate_angle_right += rotate_angle_step_right
            rotate_angle_step_right /= 2

    left_k = math.atan(rotate_angle_left)
    right_k = math.atan(rotate_angle_right)

    left_b = rotate_point_left - left_k * current_x_left
    right_b = rotate_point_right - right_k * current_x_right

    result_mask = np.copy(mask)
    for i in range(0, height):
        for j in range(0, width):
            if i < left_k * j + left_b or i < right_k * j + right_b:
                result_mask[i][j] = 0
    return result_mask


@njit(parallel=True)
def merge_masks(mask_center, mask_right, mask_left):
    final_mask = np.copy(mask_center)
    for i in range(len(mask_left)):
        for j in range(len(mask_left[0])):
            if mask_right[i][j] == 2 or mask_left[i][j] == 2:
                final_mask[i][j] = 2
            if mask_right[i][j] == 4 or mask_left[i][j] == 4:
                final_mask[i][j] = 4
    return final_mask


@njit(parallel=True)
def calc_accuracy(frame_with_lines, frame_without_lines, frame_ideal):
    true_positive = 0
    true_negative = 0
    false_positive = 0
    false_negative = 0
    height = len(frame_ideal)
    width = len(frame_ideal[0])

    frame_ideal_copy = np.copy(frame_ideal)
    for row_index in range(0, height):
        for column_index in range(0, width):
            blue = frame_ideal[row_index][column_index][0]
            green = frame_ideal[row_index][column_index][1]
            red = frame_ideal[row_index][column_index][2]
            if blue > 100 > green and red < 100:
                frame_ideal_copy[row_index][column_index] = [255, 0, 0]
            else:
                frame_ideal_copy[row_index][column_index] = [0, 0, 0]

    frame_ideal_blue = frame_ideal_copy[:, :, 0]

    for row_index in range(0, height):
        blue_indexes = []
        for column_index in range(0, width):
            if frame_ideal_blue[row_index][column_index] == 255:
                blue_indexes.append(column_index)
        if len(blue_indexes) < 2:
            continue
        for column_index in range(0, width):
            first_blue_index = blue_indexes[0]
            last_blue_index = blue_indexes[len(blue_indexes) - 1]
            if frame_with_lines[row_index][column_index] != 0:
                if first_blue_index <= column_index <= last_blue_index:
                    true_positive += 1
                else:
                    false_positive += 1
            if frame_without_lines[row_index][column_index] != 0 and frame_with_lines[row_index][column_index] == 0:
                if first_blue_index <= column_index <= last_blue_index:
                    false_negative += 1
                else:
                    true_negative += 1
    if true_positive == 0 or false_negative == 0:
        return None, None, None, None
    precision = true_positive / (true_positive + false_positive)
    recall = true_positive / (true_positive + false_negative)
    tpr = true_positive / (true_positive + false_negative)
    fpr = false_positive / (true_negative + false_positive)
    f = precision * recall / (precision + recall)
    return precision, recall, f, fpr


video_reader = cv2.VideoCapture('example_2.mp4')
example_reader = cv2.VideoCapture('example_2_ideal_2.mp4')
success = True
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
frame_buffer = []
fps = 29.97
render_seconds_count = 300
render_start_second_number = 30
filtered_frame_from_array = []
time_start = time.time()
kernel_size = 7
frame_number = 0
frame_produced_number = 0
prec_acc = 0
fpr_acc = 0
fpr_acc_array = []
recall_acc = 0
f_acc = 0
prec_acc_array = []
recall_acc_array = []
f_acc_array = []
frame_index_array = []
video_writer = cv2.VideoWriter(f'example_2_final'
                               f'.mp4', fourcc, fps, (640, 134))
while success:
    print(f'frames left: {(render_start_second_number + render_seconds_count) * 30 - frame_number}')
    frame_number = frame_number + 1
    success, frame_full = video_reader.read()
    success_final, frame_full_ideal = example_reader.read()
    if not success:
        break
    if (render_start_second_number + render_seconds_count) * 30 < frame_number:
        break
    if frame_number < render_start_second_number * 30:
        continue
    frame_time = time.time()
    # preprocessing_time = time.time()
    frame_part = frame_full[360:633]
    frame_part_final = frame_full_ideal[360:633]
    frame = cv2.resize(frame_part, (640, 134), cv2.INTER_AREA)
    frame_ideal_blue = cv2.resize(frame_part_final, (640, 134), cv2.INTER_AREA)
    # print(f'preprocessing time: {time.time() - preprocessing_time}')
    edges, direction_mask = get_edges(frame, main_threshold=0, kernel_size=kernel_size)
    # class_time = time.time()
    class_mask_center = create_class_frame(direction_mask, 4, angle_shift=False)
    class_mask_center_filtered = filtrate_angle_cluster_by_group(class_mask_center, min_threshold=500,
                                                                 max_threshold=2000)
    class_mask_right = create_class_frame(direction_mask, 4, angle_shift=True, angle_shift_size=-np.pi / 3)
    class_mask_right_filtered = filtrate_angle_cluster_by_group(class_mask_right, min_threshold=1000,
                                                                max_threshold=2000, with_neighbour=False)
    class_mask_left = create_class_frame(direction_mask, 4, angle_shift=True, angle_shift_size=np.pi / 3)
    class_mask_left_filtered = filtrate_angle_cluster_by_group(class_mask_left, min_threshold=1000, max_threshold=2000,
                                                               with_neighbour=False)
    cluster_mask = merge_masks(class_mask_center_filtered, class_mask_left_filtered, class_mask_right_filtered)
    cluster_mask_filtered = filter_mask_by_trap(cluster_mask)
    # print(f'class time: {time.time() - class_time}')
    prec, rec, f, fpr = calc_accuracy(cluster_mask_filtered, cluster_mask, frame_ideal_blue)
    if prec is not None:
        prec_acc += prec
        recall_acc += rec
        f_acc += f
        fpr_acc += fpr
        frame_produced_number += 1
        prec_acc_array.append(prec_acc / frame_produced_number)
        recall_acc_array.append(recall_acc / frame_produced_number)
        f_acc_array.append(f_acc / frame_produced_number)
        fpr_acc_array.append(fpr_acc / frame_produced_number)
        frame_index_array.append(frame_produced_number)
    img = create_image_from_cluster(cluster_mask_filtered
                                    , vertical_size=134, horizontal_size=640,
                                    merging=True,
                                    input_img=frame)
    end_frame_time = time.time() - frame_time
    print(f'frames time{end_frame_time}')
    video_writer.write(img)
print(f'average frame time is {(time.time() - time_start) / (render_seconds_count * fps)}')
print('start releasing')
print(f'Average Precision: {prec_acc / frame_produced_number}')
print(f'Average Recall: {recall_acc / frame_produced_number}')
print(f'Average F-мера: {f_acc / frame_produced_number}')
plt.title('Precision')
plt.plot(frame_index_array, prec_acc_array)
plt.show()
plt.title('Recall/TPR')
plt.plot(frame_index_array, recall_acc_array)
plt.show()
plt.title('F')
plt.plot(frame_index_array, f_acc_array)
plt.show()
plt.title('FPR')
plt.plot(frame_index_array, fpr_acc_array)
plt.show()
video_writer.release()
print('finished')
