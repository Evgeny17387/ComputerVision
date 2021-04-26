import cv2
from matplotlib import pyplot as plt
import numpy as np
import time
import random


MATCHES_TO_PLOT = 100


def plot_match_image():

    plt.imshow(image_c)
    plt.title('Match Image')
    plt.xticks([])
    plt.yticks([])

    plt.get_current_fig_manager().window.state('zoomed')
    plt.show()


if __name__ == '__main__':

    time_start = time.time()

    pair = 1

    image_a_filename = f'Part_2\\pair{pair}_imageA.jpg'
    image_b_filename = f'Part_2\\pair{pair}_imageB.jpg'

    image_a = cv2.imread(image_a_filename)
    image_b = cv2.imread(image_b_filename)

    image_a_gray = cv2.cvtColor(image_a, cv2.COLOR_BGR2GRAY)
    image_b_gray = cv2.cvtColor(image_b, cv2.COLOR_BGR2GRAY)

    sift = cv2.SIFT_create()
    image_a_keypoints, image_a_descriptor = sift.detectAndCompute(image_a_gray, None)
    image_b_keypoints, image_b_descriptor = sift.detectAndCompute(image_b_gray, None)

    n = image_a_descriptor.shape[0]
    m = image_b_descriptor.shape[0]

    distances = np.zeros((n, m))
    for i in range(n):
        for j in range(m):
            distances[i, j] = sum(abs(image_a_descriptor[i] - image_b_descriptor[j]))

    # test_1 = False
    test_1 = True
    # test_2 = False
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

    matches_indices = random.sample(range(0, len(matches)), MATCHES_TO_PLOT)

    matches_to_plot = [matches[index] for index in matches_indices]

    image_c = cv2.drawMatchesKnn(image_a, image_a_keypoints, image_b, image_b_keypoints, matches_to_plot, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    print(f"Run Time: {time.time() - time_start}")

    plot_match_image()
