import cv2
from matplotlib import pyplot as plt


def plot_graphs():

    plt.subplot(221)
    plt.imshow(image_a)
    plt.title('Image A')
    plt.xticks([])
    plt.yticks([])

    plt.subplot(222)
    plt.imshow(image_b)
    plt.title('Image B')
    plt.xticks([])
    plt.yticks([])

    plt.subplot(223)
    plt.imshow(image_a_with_keypoints)
    plt.title('Image A - KP')
    plt.xticks([])
    plt.yticks([])

    plt.subplot(224)
    plt.imshow(image_b_with_keypoints)
    plt.title('Image B - KP')
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

    image_a_with_keypoints = image_a.copy()
    image_b_with_keypoints = image_b.copy()

    cv2.drawKeypoints(image_a_gray, image_a_keypoints, image_a_with_keypoints)
    cv2.drawKeypoints(image_b_gray, image_b_keypoints, image_b_with_keypoints)

    plot_graphs()
