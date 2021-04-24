import cv2
from matplotlib import pyplot as plt
import numpy as np
import random
import time


def plot_images():

    plt.subplot(321)
    plt.imshow(image_a, cmap='gray')
    plt.title('Image A')
    plt.xticks([])
    plt.yticks([])

    plt.subplot(322)
    plt.imshow(image_b, cmap='gray')
    plt.title('Image B')
    plt.xticks([])
    plt.yticks([])

    plt.subplot(323)
    plt.imshow(image_c, cmap='gray')
    plt.title('Image C')
    plt.xticks([])
    plt.yticks([])

    plt.subplot(324)
    plt.imshow(image_d, cmap='gray')
    plt.title('Image D')
    plt.xticks([])
    plt.yticks([])

    plt.subplot(325)
    plt.imshow(image_e, cmap='gray')
    plt.title('Image E')
    plt.xticks([])
    plt.yticks([])

    plt.subplot(326)
    plt.imshow(image_f, cmap='gray')
    plt.title('Image F')
    plt.xticks([])
    plt.yticks([])

    plt.get_current_fig_manager().window.state('zoomed')
    plt.show()


def run_matches():

    matches = list()

    if not manual_matches:

        bf = cv2.BFMatcher()
        matches = bf.knnMatch(image_a_descriptor, image_b_descriptor, k=1)

    else:

        n = image_a_descriptor.shape[0]
        m = image_b_descriptor.shape[0]

        distances = np.zeros((n, m))
        for i in range(n):
            for j in range(m):
                distances[i, j] = sum(abs(image_a_descriptor[i] - image_b_descriptor[j]))

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

    return matches


def run_ransac():

    image_a_points = np.float32([image_a_keypoints[match[0].queryIdx].pt for match in matches]).reshape(-1, 1, 2)
    image_b_points = np.float32([image_b_keypoints[match[0].trainIdx].pt for match in matches]).reshape(-1, 1, 2)

    M = np.zeros((3, 3))
    matchesMask = list()

    if not manual_ransac:

        M, mask = cv2.findHomography(image_a_points, image_b_points, cv2.RANSAC, 5.0)
        matchesMask = mask.ravel().tolist()

    else:

        inlieners = 0

        while inlieners < len(matches) / 3:

            match_indices = random.sample(range(0, len(matches)), 4)

            image_a_random_seed_points = np.float32([image_a_keypoints[matches[match_index][0].queryIdx].pt for match_index in match_indices]).reshape(-1, 1, 2)
            image_b_random_seed_points = np.float32([image_b_keypoints[matches[match_index][0].trainIdx].pt for match_index in match_indices]).reshape(-1, 1, 2)

            M, _ = cv2.findHomography(image_a_random_seed_points, image_b_random_seed_points)
            print(M)

            image_a_points_transformed = cv2.perspectiveTransform(image_a_points, M)

            transform_error = np.linalg.norm((image_a_points_transformed - image_b_points).reshape(-1, 2), axis=1)

            error_threshold = 5

            matchesMask = [1 if x < error_threshold else 0 for x in transform_error]

            inlieners = sum(matchesMask)

    return M, matchesMask


if __name__ == '__main__':

    time_start = time.time()

    pair = 1

    # manual_matches = False
    manual_matches = True

    # test_1 = False
    test_1 = True
    # test_2 = False
    test_2 = True

    manual_ransac = False
    # manual_ransac = True

    image_pair_extension = {
        1: [0, 0],
        2: [100, 100],
        3: [100, 300]
    }

    image_a_filename = f'pair{pair}_imageA.jpg'
    image_b_filename = f'pair{pair}_imageB.jpg'

    image_a = cv2.imread(image_a_filename, 0)
    image_b = cv2.imread(image_b_filename, 0)

    h, w = image_a.shape

    sift = cv2.SIFT_create()
    image_a_keypoints, image_a_descriptor = sift.detectAndCompute(image_a, None)
    image_b_keypoints, image_b_descriptor = sift.detectAndCompute(image_b, None)

    matches = run_matches()

    M, matchesMask = run_ransac()

    image_a_corners = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
    image_b_corners = cv2.perspectiveTransform(image_a_corners, M)
    image_e = np.copy(image_b)
    image_e = cv2.polylines(image_e, [np.int32(image_b_corners)], True, 255, 3, cv2.LINE_AA)

    image_c = cv2.drawMatchesKnn(image_a, image_a_keypoints, image_b, image_b_keypoints, matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    draw_params = dict(matchColor=(0, 255, 0), singlePointColor=None, matchesMask=matchesMask, flags=2)
    image_d = cv2.drawMatches(image_a, image_a_keypoints, image_b, image_b_keypoints, [m[0] for m in matches], None, **draw_params)

    image_f = cv2.warpPerspective(image_a, M, (w + image_pair_extension[pair][0], h + image_pair_extension[pair][1]))

    for i in range(image_f.shape[0]):
        for j in range(image_f.shape[1]):
            if (i >= image_b.shape[0]) or (j >= image_b.shape[1]):
                continue
            if image_f[i, j] == 0:
                image_f[i, j] = image_b[i, j]

    print(f"Run Time: {time.time() - time_start}")

    plot_images()
