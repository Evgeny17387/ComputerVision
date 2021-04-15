import cv2
from matplotlib import pyplot as plt
from scipy import signal
import numpy as np
import math

SQRT_3_DIV_6 = math.sqrt(3) / 6
SQRT_3_DIV_2 = math.sqrt(3) / 2

COS_60 = math.cos(math.pi / 3)
SIN_60 = math.sin(math.pi / 3)

COS_30 = math.cos(math.pi / 6)
SIN_30 = math.sin(math.pi / 6)


def plot_graphs():

    plt.subplot(141)
    plt.imshow(image, cmap='gray')
    plt.title('Image')
    plt.xticks([])
    plt.yticks([])

    plt.subplot(142)
    plt.imshow(edges, cmap='gray')
    plt.title('Edge - Canny')
    plt.xticks([])
    plt.yticks([])

    plt.subplot(143)
    plt.imshow(gradient * binary_mask, cmap='gray')
    plt.title('Gradient')
    plt.xticks([])
    plt.yticks([])

    plt.subplot(144)
    plt.imshow(gradient_direction * binary_mask, cmap='gray')
    plt.title('Gradient Direction')
    plt.xticks([])
    plt.yticks([])

    plt.get_current_fig_manager().window.state('zoomed')
    plt.show()


def calculate_angles():

    angles = dict()
    for x in range(gradient_direction.shape[0]):
        for y in range(gradient_direction.shape[1]):

            if binary_mask[x, y] == 0:
                continue

            angle = gradient_direction[x, y]

            if angle in angles:
                angles[angle] += 1
            else:
                angles[angle] = 1

    print({k: v for k, v in sorted(angles.items(), key=lambda item: item[1], reverse=True)})

    plt.scatter(angles.keys(), angles.values())

    plt.get_current_fig_manager().window.state('zoomed')
    plt.show()


def hough_transfom():

    # given the canny image, we do the following:
    # 1. iterate over all points of edge image
    # 2. assume triangle angle theta = 0
    # 3. check all 3 possible edges
    # 4. given the triangle edge lenght, iterate over all possible triangles centers given the image point

    possible_triangles = np.zeros((image_length_y, image_length_x))

    for x in range(image_length_x):
        for y in range(image_length_y):

            if binary_mask[y, x] == 0:
                continue

            for i in range(-edge_length_half, edge_length_half):

                # Horizontal edge

                center_y = int(y + edge_to_center)
                center_x = int(x + i)
                if not ((center_x < 0) or (center_x >= image_length_x) or (center_y < 0) or (center_y >= image_length_y)):
                    possible_triangles[center_y, center_x] += 1

                # left edge

                center_y = int(y + i * COS_30 - edge_to_center * COS_60)
                center_x = int(x + i * SIN_30 + edge_to_center * SIN_60)
                if not ((center_x < 0) or (center_x >= image_length_x) or (center_y < 0) or (center_y >= image_length_y)):
                    possible_triangles[center_y, center_x] += 1

                # right edge

                center_y = int(y + i * COS_30 - edge_to_center * COS_60)
                center_x = int(x - i * SIN_30 - edge_to_center * SIN_60)
                if not ((center_x < 0) or (center_x >= image_length_x) or (center_y < 0) or (center_y >= image_length_y)):
                    possible_triangles[center_y, center_x] += 1

    center_y, center_x = np.unravel_index(np.argmax(possible_triangles), possible_triangles.shape)

    x1 = center_x - edge_length_half
    y1 = center_y - edge_to_center

    x2 = center_x + edge_length_half
    y2 = y1

    x3 = center_x
    y3 = y1 + edge_height

    plt.subplot(121)
    plt.imshow(image, cmap='gray')
    plt.title('Image')
    plt.xticks([])
    plt.yticks([])
    plt.gca().add_patch(plt.Polygon([[x1, y1], [x2, y2], [x3, y3]], facecolor="none", edgecolor='blue'))

    plt.subplot(122)
    plt.imshow(possible_triangles, cmap='gray')
    plt.title('Hough Transform')
    plt.xticks([])
    plt.yticks([])

    plt.get_current_fig_manager().window.state('zoomed')
    plt.show()


if __name__ == '__main__':

    image = cv2.imread('test_1.jpg', 0)
    edge_length = 500
    threshold_1 = 100
    threshold_2 = 200

    image_length_x = image.shape[1]
    image_length_y = image.shape[0]

    edge_length_half = int(edge_length / 2)
    edge_to_center = SQRT_3_DIV_6 * edge_length
    edge_height = SQRT_3_DIV_2 * edge_length

    edges = cv2.Canny(image, threshold_1, threshold_2)

    binary_mask = np.int8(edges / 255)

    dx = signal.convolve2d(image, [[1, -1]], 'same')
    dy = signal.convolve2d(image, [[1], [-1]], 'same')

    gradient = np.sqrt((dx ** 2) + (dy ** 2))

    gradient_direction = np.int16(np.arctan2(dy, dx) * 180 / math.pi)

    # plot_graphs()
    # calculate_angles()
    hough_transfom()
