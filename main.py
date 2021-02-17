import numpy as np
import cv2
from PIL import Image, ImageDraw
# import Image  as SimpleImage
import matplotlib.pyplot as plt

vidcap = cv2.VideoCapture('example.mp4')


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
    mask[(image > threshold)] = 1
    return mask


def get_edges(image, separate_channels=False, main_threshold=90, horizontal_threshold=10):
    hls = cv2.cvtColor(np.copy(image), cv2.COLOR_RGB2HLS).astype(np.float)
    s_channel = hls[:, :, 1]
    gradient_x = gradient_abs_value_mask(s_channel, axis='x', sobel_kernel=3, threshold=horizontal_threshold)
    # gradient_y = gradient_abs_value_mask(s_channel, axis='y', sobel_kernel=3, threshold=200)
    magnitude = gradient_magnitude_mask(s_channel, sobel_kernel=3, threshold=main_threshold)
    # direction = gradient_direction_mask(s_channel, sobel_kernel=3, threshold=(0.7, 1.3))
    gradient_mask = np.zeros_like(s_channel)
    gradient_mask[((gradient_x == 1) | (magnitude == 1))] = 1
    # color_mask = color_threshold_mask(s_channel, threshold=130)

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
    fps = 29.97
    videoReader = cv2.VideoCapture('example.mp4')
    videoWriter = cv2.VideoWriter('frame_test.mp4', fourcc, fps, (1280, 720), False)
    frame_number_target = 3570
    frame_number = 0
    success = True
    while success:
        frame_number = frame_number + 1
        success, frame = videoReader.read()
        if frame_number != frame_number_target:
            continue
        if frame_number == frame_number_target:
            for main_threshold in range(0, 255, 10):
                for horizontal_threshold in range(0, 255, 10):
                    frame_array = get_edges(frame, main_threshold=main_threshold, horizontal_threshold=horizontal_threshold)
                    result_frame = create_frame_from_array(frame_array)
                    print(horizontal_threshold + main_threshold)
                    videoWriter.write(result_frame)
            videoWriter.release()
            break


create_threshold_variative_video_by_frame()
# result = get_edges(testImg)
# plt.imshow(result)
# plt.savefig("array")

# success = True
# fourcc = cv2.VideoWriter_fourcc(*'mp4v')
# fps = 29.97
# videoEdge = cv2.VideoWriter('example_out.mp4', fourcc, fps, (1280, 720), False)
# render_seconds_count = 120
# render_start_second_number = 20
# frame_number = 0;
# while success:
#     print(frame_number)
#     frame_number = frame_number + 1
#     success, image = vidcap.read()
#     if (render_start_second_number + render_seconds_count) * 30 < frame_number:
#         break
#     if frame_number < render_start_second_number * 30:
#         continue
#     edges = get_edges(image)
#     rgbEdges = np.zeros((720, 1280), dtype='uint8')
#     for i in range(0, len(edges)):
#         for j in range(0, len(edges[0])):
#             if edges[i, j] == 0.0:
#                 rgbEdges[i, j] = 0
#             else:
#                 rgbEdges[i, j] = 255
#     videoEdge.write(rgbEdges)
# print('start releasing')
# videoEdge.release()
# print('finished')
