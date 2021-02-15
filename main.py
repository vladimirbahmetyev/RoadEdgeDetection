import numpy as np
import time
import cv2
from PIL import Image, ImageDraw
# import Image  as SimpleImage
import matplotlib.pyplot as plt

vidcap = cv2.VideoCapture('example.mp4')


# testImg = Image.open('example.jpg')

def gradient_abs_value_mask(image, sobel_kernel=3, axis='x', threshold=(0, 255)):
    if axis == 'x':
        sobel = np.absolute(cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=sobel_kernel))
    if axis == 'y':
        sobel = np.absolute(cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=sobel_kernel))
    sobel = np.uint8(255 * sobel / np.max(sobel))
    mask = np.zeros_like(sobel)
    mask[(sobel >= threshold[0]) & (sobel <= threshold[1])] = 1
    return mask


def gradient_magnitude_mask(image, sobel_kernel=3, threshold=(0, 255)):
    sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    magnitude = np.sqrt(sobel_x ** 2 + sobel_y ** 2)
    magnitude = (magnitude * 255 / np.max(magnitude)).astype(np.uint8)
    mask = np.zeros_like(magnitude)
    mask[(magnitude >= threshold[0]) & (magnitude <= threshold[1])] = 1
    return mask


def gradient_direction_mask(image, sobel_kernel=3, threshold=(0, np.pi / 2)):
    sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    direction = np.arctan2(np.absolute(sobel_y), np.absolute(sobel_x))
    mask = np.zeros_like(direction)
    mask[(direction >= threshold[0]) & (direction <= threshold[1])] = 1
    return mask


def color_threshold_mask(image, threshold=(0, 255)):
    mask = np.zeros_like(image)
    mask[(image > threshold[0]) & (image <= threshold[1])] = 1
    return mask


def get_edges(image, separate_channels=False):
    hls = cv2.cvtColor(np.copy(image), cv2.COLOR_RGB2HLS).astype(np.float)
    s_channel = hls[:, :, 1]
    gradient_x = gradient_abs_value_mask(s_channel, axis='x', sobel_kernel=3, threshold=(20, 100))
    gradient_y = gradient_abs_value_mask(s_channel, axis='y', sobel_kernel=3, threshold=(20, 100))
    magnitude = gradient_magnitude_mask(s_channel, sobel_kernel=3, threshold=(20, 100))
    direction = gradient_direction_mask(s_channel, sobel_kernel=3, threshold=(0.7, 1.3))
    gradient_mask = np.zeros_like(s_channel)
    gradient_mask[((gradient_x == 1) & (gradient_y == 1)) | ((magnitude == 1) & (direction == 1))] = 1
    color_mask = color_threshold_mask(s_channel, threshold=(80, 150))

    if separate_channels:
        return np.dstack((np.zeros_like(s_channel), gradient_mask, color_mask))
    else:
        mask = np.zeros_like(gradient_mask)
        mask[(gradient_mask == 1) | (color_mask == 1)] = 1
        return mask


# result = get_edges(testImg)
# plt.imshow(result)
# plt.savefig("array")

success = True
frame_array = []
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
fps = 29.97
videoEdge = cv2.VideoWriter('example_out_v2.mp4', fourcc, fps, (1280, 720), False)
# iterationCount = 0
start_time = time.time()
while success:
    # print(iterationCount)
    # iterationCount = iterationCount + 1
    success, image = vidcap.read()
    # if iterationCount > 1580:
    #     break
    # if iterationCount < 780:
    #     continue
    # frame_time_start = time.time()

    edges = get_edges(image)
    # frame_time_end = time.time()
    # print(frame_time_end - frame_time_start)
    rgbEdges = np.zeros((720, 1280), dtype='uint8')
    for i in range(0, len(edges)):
        for j in range(0, len(edges[0])):
            if edges[i, j] == 0.0:
                rgbEdges[i, j] = 0
            else:
                rgbEdges[i, j] = 255
    videoEdge.write(rgbEdges)
end_time = time.time()
print(end_time - start_time)
print('start releasing')

videoEdge.release()
print('finished')
