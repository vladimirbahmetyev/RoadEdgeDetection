import numpy as np
import cv2
import math
from numba import njit
import time
from PIL import Image, ImageDraw
# import Image  as SimpleImage
import matplotlib.pyplot as plt


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


def get_edges(image, separate_channels=False, main_threshold=90, horizontal_threshold=0, color_threshold=120):
    hls = cv2.cvtColor(np.copy(image), cv2.COLOR_RGB2HLS).astype(np.float)
    s_channel = hls[:, :, 1]
    # calc_frame_stat(s_channel)
    sobel_x = cv2.Sobel(s_channel, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(s_channel, cv2.CV_64F, 0, 1, ksize=3)
    magnitude = gradient_magnitude_mask(sobel_x, sobel_y, threshold=main_threshold)
    direction_mask = gradient_direction_mask(sobel_x, sobel_y)
    gradient_mask = np.zeros_like(s_channel)
    gradient_mask[(magnitude == 1)] = 1
    color_mask = color_threshold_mask(s_channel, threshold=color_threshold)
    direction_mask[gradient_mask != 1] = 0

    if separate_channels:
        return np.dstack((np.zeros_like(s_channel), gradient_mask, color_mask))
    else:
        mask = np.zeros_like(gradient_mask)
        mask[(gradient_mask == 1)] = 1
        return mask, direction_mask


@njit(parallel=True, nopython=False)
def create_frame_from_array(array):
    frame = np.zeros((720, 1280), dtype='uint8')
    for i in range(0, len(array)):
        for j in range(0, len(array[0])):
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
    videoWriter = cv2.VideoWriter('frame_test_lightness_1_20_with_frame_buffer.mp4', fourcc, fps, (1280, 720), False)
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
def create_cluster_frame(direction_mask):
    cluster_mask = np.zeros_like(direction_mask)
    for row_index in range(len(direction_mask)):
        for column_index in range(len(direction_mask[row_index])):
            if direction_mask[row_index][column_index] == 0:
                cluster_mask[row_index][column_index] = 0
                continue
            cluster_mask[row_index][column_index] = direction_mask[row_index][column_index] // (2 * np.pi / 8) + 1
    return cluster_mask


@njit(parallel=True)
def create_image_from_cluster(cluster_mask):
    img = np.zeros((720, 1280, 3), np.uint8)
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


@njit(parallel=True)
def create_clusters_from_frame(frame, threshold=10):
    vertical_start = 360
    pixels_list_with_coords = []
    for i in range(vertical_start, len(frame)):
        for j in range(0, len(frame)):

            if frame[i][j] > 0.0:
                pixels_list_with_coords.append([i, j])
    pixels_length = len(pixels_list_with_coords)
    matrix = [[] for i in range(0, pixels_length)]
    print('start to cound dist')
    for i in range(0, pixels_length):
        for j in range(i, pixels_length):
            dist = math.sqrt((pixels_list_with_coords[i][0] - pixels_list_with_coords[j][0]) ** 2 + (
                    pixels_list_with_coords[i][1] - pixels_list_with_coords[j][1]) ** 2)
            if dist < threshold and i != j:
                matrix[i].append(j)
    clusters = []
    print('counted dist')
    checked_pixels = [False for i in range(0, pixels_length)]

    for i in range(0, len(pixels_list_with_coords)):
        if checked_pixels[i]:
            continue
        new_cluster = []
        cluster_queue = [i]
        checked_pixels[i] = True
        while len(cluster_queue) > 0:
            current_pixel = cluster_queue.pop(0)
            new_cluster.append(current_pixel)
            checked_pixels[current_pixel] = True
            for neighbor_pixel_index in range(0, len(matrix[i])):
                if not checked_pixels[neighbor_pixel_index]:
                    cluster_queue.append(neighbor_pixel_index)
        clusters.append(new_cluster)

    result_clusters = []
    for cluster in clusters:
        cluster_with_coords = []
        for pixel_index in cluster:
            cluster_with_coords.append(pixels_list_with_coords[pixel_index])
        result_clusters.append(cluster_with_coords)
    return result_clusters


# create_threshold_variative_video_by_frame()
# result = get_edges(testImg)
# plt.imshow(result)
# plt.savefig("array")
vidcap = cv2.VideoCapture('example_2.mp4')
success = True
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
frame_buffer = []
fps = 29.97
videoEdge = cv2.VideoWriter('example_out_2_with_numba.mp4', fourcc, fps, (1280, 720))
render_seconds_count = 30
render_start_second_number = 25
frame_number = 0
while success:

    print(f'frames left: {(render_start_second_number + render_seconds_count) * 30 - frame_number}')
    frame_number = frame_number + 1
    success, frame = vidcap.read()
    if (render_start_second_number + render_seconds_count) * 30 < frame_number:
        break
    if frame_number < render_start_second_number * 30:
        continue
    edges, direction_mask = get_edges(frame, main_threshold=5)
    # img = create_image_from_cluster(create_cluster_frame(direction_mask))
    if len(frame_buffer) < 5:
        frame_buffer.append(edges)
        continue
    if len(frame_buffer) >= 5:
        frame_buffer.append(edges)
        frame_buffer.pop(0)
        filtered_frame_from_array = filter_edges_by_history(frame_buffer, buffer_value=4)
        rgbEdges = create_frame_from_array(filtered_frame_from_array)
        direction_mask[rgbEdges == 0] = 0
        cluster_mask = create_cluster_frame(direction_mask)
        img = create_image_from_cluster(cluster_mask)
        videoEdge.write(img)
    # projection = get_projection(rgbEdges)
print('start releasing')
videoEdge.release()
print('finished')
