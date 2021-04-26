import cv2
from matplotlib import pyplot as plt
import numpy as np


def plot_images():

    plt.imshow(image_output_transform, cmap='gray')
    plt.title('Image Output Transform')
    plt.xticks([])
    plt.yticks([])

    plt.get_current_fig_manager().window.state('zoomed')
    plt.show()


if __name__ == '__main__':

    image_filename_source = 'Part_1\\Dylan.jpg'
    image_filename_destination = 'Part_1\\frames.jpg'

    image_source = cv2.imread(image_filename_source)
    image_destination = cv2.imread(image_filename_destination)

    image_source_gray = cv2.cvtColor(image_source, cv2.COLOR_BGR2GRAY)
    image_destination_gray = cv2.cvtColor(image_destination, cv2.COLOR_BGR2GRAY)

    # Source Points

    y1_source = 0
    x1_source = 0

    y2_source = 0
    x2_source = image_source_gray.shape[1] - 1

    y3_source = image_source_gray.shape[0] - 1
    x3_source = image_source_gray.shape[1] - 1

    y4_source = image_source_gray.shape[0] - 1
    x4_source = 0

    # Transform 1

    xy_source_transform_1 = np.float32([[x1_source, y1_source], [x2_source, y2_source], [x3_source, y3_source], [x4_source, y4_source]])

    # xy_destination_transform_1 = plt.ginput(4)
    xy_destination_transform_1 = np.float32([[40, 182], [195, 54], [495, 158], [429, 491]])

    h_transform_1, _ = cv2.findHomography(xy_source_transform_1, xy_destination_transform_1)

    image_output_transform_1 = cv2.warpPerspective(image_source_gray, h_transform_1, (image_destination_gray.shape[1], image_destination_gray.shape[0]))

    # Transform 2

    xy_source_transform_2 = np.float32([[x1_source, y1_source], [x2_source, y2_source], [x3_source, y3_source]])

    # xy_destination_transform_2 = plt.ginput(4)
    xy_destination_transform_2 = np.float32([[551, 222], [840, 69], [899, 295]])

    h_transform_2 = cv2.getAffineTransform(xy_source_transform_2, xy_destination_transform_2)

    image_output_transform_2 = cv2.warpAffine(image_source_gray, h_transform_2, (image_destination_gray.shape[1], image_destination_gray.shape[0]))

    image_output_transform = image_output_transform_1 + image_output_transform_2

    plot_images()
