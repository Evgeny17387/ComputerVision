import cv2
from matplotlib import pyplot as plt
import numpy as np


def plot_key_points():

    image_a_with_keypoints = image_a.copy()
    image_b_with_keypoints = image_b.copy()

    cv2.drawKeypoints(image_a_gray, image_a_keypoints, image_a_with_keypoints)
    cv2.drawKeypoints(image_b_gray, image_b_keypoints, image_b_with_keypoints)

    plt.subplot(121)
    plt.imshow(image_a_with_keypoints)
    plt.title('Image A - KP')
    plt.xticks([])
    plt.yticks([])

    plt.subplot(122)
    plt.imshow(image_b_with_keypoints)
    plt.title('Image B - KP')
    plt.xticks([])
    plt.yticks([])


def plot_match_image():

    plt.imshow(image_c)
    plt.title('Match Image')
    plt.xticks([])
    plt.yticks([])

    plt.get_current_fig_manager().window.state('zoomed')
    plt.show()


if __name__ == '__main__':

    image_a_filename = 'pair1_imageA.jpg'
    image_b_filename = 'pair1_imageB.jpg'

    image_a = cv2.imread(image_a_filename)
    image_b = cv2.imread(image_b_filename)

    image_a_gray = cv2.cvtColor(image_a, cv2.COLOR_BGR2GRAY)
    image_b_gray = cv2.cvtColor(image_b, cv2.COLOR_BGR2GRAY)

    sift = cv2.SIFT_create()
    image_a_keypoints, image_a_descriptor = sift.detectAndCompute(image_a_gray, None)
    image_b_keypoints, image_b_descriptor = sift.detectAndCompute(image_b_gray, None)

    # Debug Code
    image_a_descriptor = image_a_descriptor[:100, :]
    image_b_descriptor = image_b_descriptor[:100, :]

    n = image_a_descriptor.shape[0]
    m = image_b_descriptor.shape[0]

    distances = np.zeros((n, m))
    for i in range(n):
        for j in range(m):
            distances[i, j] = sum(abs(image_a_descriptor[i] - image_b_descriptor[j]))

    # Bi Directional Match

    test_1 = False
    test_2 = True

    matches = list()
    for i in range(n):

        min_distance_1 = 0xFFFFFFFF
        match_index_1 = -1

        min_distance_2 = 0xFFFFFFFF
        match_index_2 = -1

        for j in range(m):

            distance = distances[i, j]

            if distance < min_distance_1:
                min_distance_1 = distance
                match_index_1 = j
            elif distance < min_distance_2:
                min_distance_2 = distance
                match_index_2 = j

        if test_1:

            if min_distance_1 < 0.8 * min_distance_2:
                matches.append([cv2.DMatch(i, match_index_1, 0, min_distance_1)])

        if test_2:

            min_distance_3 = 0xFFFFFFFF
            match_index_3 = -1

            for k in range(n):

                distance = distances[k, match_index_1]

                if distance < min_distance_3:
                    min_distance_3 = distance
                    match_index_3 = k

            if match_index_3 == i:
                matches.append([cv2.DMatch(i, match_index_1, 0, min_distance_1)])

        else:
            matches.append([cv2.DMatch(i, match_index_1, 0, min_distance_1)])

    image_c = cv2.drawMatchesKnn(image_a, image_a_keypoints, image_b, image_b_keypoints, matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    # plot_key_points()
    plot_match_image()
