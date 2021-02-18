import numpy as np
import cv2
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


def gradient_magnitude_mask(image, sobel_kernel=3, threshold=120):
    sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    magnitude = np.sqrt(sobel_x ** 2 + sobel_y ** 2)
    magnitude = (magnitude * 255 / np.max(magnitude)).astype(np.uint8)
    mask = np.zeros_like(magnitude)
    mask[(magnitude >= threshold)] = 1
    return mask


def gradient_direction_mask(image, sobel_kernel=3, threshold=(0, np.pi / 2)):
    sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    direction = np.arctan2(np.absolute(sobel_y), np.absolute(sobel_x))
    mask = np.zeros_like(direction)
    mask[(direction >= threshold[0]) & (direction <= threshold[1])] = 1
    return mask


def color_threshold_mask(image, threshold=120):
    mask = np.zeros_like(image)
    mask[(image < threshold)] = 1
    return mask


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
    gradient_x = gradient_abs_value_mask(s_channel, axis='x', sobel_kernel=3, threshold=horizontal_threshold)
    # gradient_y = gradient_abs_value_mask(s_channel, axis='y', sobel_kernel=3, threshold=200)
    magnitude = gradient_magnitude_mask(s_channel, sobel_kernel=3, threshold=main_threshold)
    # direction = gradient_direction_mask(s_channel, sobel_kernel=3, threshold=(0.7, 1.3))
    gradient_mask = np.zeros_like(s_channel)
    gradient_mask[((gradient_x == 1) & (magnitude == 1))] = 1
    color_mask = color_threshold_mask(s_channel, threshold=color_threshold)

    if separate_channels:
        return np.dstack((np.zeros_like(s_channel), gradient_mask, color_mask))
    else:
        mask = np.zeros_like(gradient_mask)
        mask[(gradient_mask == 1)] = 1
        return mask


def create_frame_from_array(array):
    frame = np.zeros((720, 1280), dtype='uint8')
    for i in range(0, len(array)):
        for j in range(0, len(array[0])):
            if array[i, j] == 0.0:
                frame[i, j] = 0
            else:
                frame[i, j] = 255
    return frame


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


def filter_edges_by_history(frame_history_array, buffer_value=3):
    frame_t = np.zeros_like(frame_history_array[0])
    for i in range(0, len(frame_history_array[0])):
        for j in range(0, len(frame_history_array[0][0])):
            for frame_index in range(len(frame_history_array)):
                if frame_history_array[frame_index][i][j] == 1.0:
                    frame_t[i][j] = frame_t[i][j] + 1
    filtered_frame_from_array_t = np.zeros_like(frame_t)
    filtered_frame_from_array_t[(frame_t >= buffer_value)] = 1.0
    return filtered_frame_from_array_t

# create_threshold_variative_video_by_frame()
# result = get_edges(testImg)
# plt.imshow(result)
# plt.savefig("array")
vidcap = cv2.VideoCapture('example_2.mp4')
success = True
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
frame_buffer = []
fps = 29.97
videoEdge = cv2.VideoWriter('example_out_2_with_buffer.mp4', fourcc, fps, (1280, 720), False)
render_seconds_count = 60
render_start_second_number = 20
frame_number = 0
while success:

    print(f'frames left: {(render_start_second_number + render_seconds_count) * 30 - frame_number}')
    frame_number = frame_number + 1
    success, frame = vidcap.read()
    if (render_start_second_number + render_seconds_count) * 30 < frame_number:
        break
    if frame_number < render_start_second_number * 30:
        continue
    edges = get_edges(frame, main_threshold=5)
    if len(frame_buffer) < 5:
        frame_buffer.append(edges)
        continue
    if len(frame_buffer) >= 5:
        frame_buffer.append(edges)
        frame_buffer.pop(0)
        filtered_frame_from_array = filter_edges_by_history(frame_buffer, buffer_value=4)
        rgbEdges = create_frame_from_array(filtered_frame_from_array)
        videoEdge.write(rgbEdges)
print('start releasing')
videoEdge.release()
print('finished')
