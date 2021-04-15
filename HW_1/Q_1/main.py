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

TRIANGLE_ANGLES = 120


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


def hough_transfom():

    # given the canny image, we do the following:
    # 1. iterate over all points of edge image
    # 2. edge angle is accordingly to graident direction
    # 3. given the triangle edge length, iterate over all possible triangles centers given the image point

    possible_triangles = np.zeros((image_length_y, image_length_x, TRIANGLE_ANGLES))

    for x in range(image_length_x):
        for y in range(image_length_y):

            if binary_mask[y, x] == 0:
                continue

            for i in range(-edge_length_half, edge_length_half):

                # Horizontal edge

                direction = gradient_direction[y, x]

                cos_direction_normal = math.cos(direction * math.pi / 180)
                sin_direction_normal = math.sin(direction * math.pi / 180)

                cos_direction_edge = math.cos((direction + 90) * math.pi / 180)
                sin_direction_edge = math.sin((direction + 90) * math.pi / 180)

                center_y = int(y + i * sin_direction_edge + edge_to_center * sin_direction_normal)

                center_x = int(x + i * cos_direction_edge + edge_to_center * cos_direction_normal)

                if not ((center_x < 0) or (center_x >= image_length_x) or (center_y < 0) or (center_y >= image_length_y)):
                    possible_triangles[center_y, center_x, direction % TRIANGLE_ANGLES] += 1

    center_y, center_x, triangle_angel = np.unravel_index(np.argmax(possible_triangles), possible_triangles.shape)

    angle_1 = triangle_angel
    angle_2 = triangle_angel + 120
    angle_3 = triangle_angel + 240

    x1 = center_x + (edge_height - edge_to_center) * math.cos(angle_1 * math.pi / 180)
    y1 = center_y + (edge_height - edge_to_center) * math.sin(angle_1 * math.pi / 180)

    x2 = center_x + (edge_height - edge_to_center) * math.cos(angle_2 * math.pi / 180)
    y2 = center_y + (edge_height - edge_to_center) * math.sin(angle_2 * math.pi / 180)

    x3 = center_x + (edge_height - edge_to_center) * math.cos(angle_3 * math.pi / 180)
    y3 = center_y + (edge_height - edge_to_center) * math.sin(angle_3 * math.pi / 180)

    triangle = np.array([[x1, y1], [x2, y2], [x3, y3]])

    plt.subplot(121)
    plt.imshow(image, cmap='gray')
    plt.title('Image')
    plt.xticks([])
    plt.yticks([])
    plt.gca().add_patch(plt.Polygon(triangle, facecolor="none", edgecolor='blue'))

    plt.subplot(122)
    plt.imshow(possible_triangles[:, :, triangle_angel], cmap='gray')
    plt.title(f'Hough Transform\nAngle: {triangle_angel}')
    plt.xticks([])
    plt.yticks([])

    plt.get_current_fig_manager().window.state('zoomed')
    plt.show()


if __name__ == '__main__':

    # image_name = 'test_1.jpg'
    # image_name = 'test_2.jpg'
    image_name = 'test_3.jpg'
    edge_length = 180

    threshold_1 = 100
    threshold_2 = 200

    image = cv2.imread(image_name, 0)

    image_length_x = image.shape[1]
    image_length_y = image.shape[0]

    edge_length_half = int(edge_length / 2)
    edge_to_center = SQRT_3_DIV_6 * edge_length
    edge_height = SQRT_3_DIV_2 * edge_length

    edges = cv2.Canny(image, threshold_1, threshold_2)

    binary_mask = np.int8(edges / 255)

    dx = signal.convolve2d(image, [[-1, 1]], 'same')
    dy = signal.convolve2d(image, [[-1], [1]], 'same')

    gradient = np.sqrt((dx ** 2) + (dy ** 2))

    gradient_direction = np.int16(np.arctan2(dy, dx) * 180 / math.pi)

    # plot_graphs()
    hough_transfom()
