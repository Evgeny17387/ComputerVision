import cv2
from matplotlib import pyplot as plt
import math


if __name__ == '__main__':

    image_filename = 'UoH.jpg'

    size_threshold = 8

    image = cv2.imread(image_filename)

    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    sift = cv2.SIFT_create()
    image_keypoints = sift.detect(image_gray, None)

    for image_keypoint in image_keypoints:

        size = image_keypoint.size

        if size > size_threshold:

            x_start = image_keypoint.pt[0]
            y_start = image_keypoint.pt[1]

            angle = image_keypoint.angle

            x_end = x_start + size * math.cos(angle * math.pi / 180)
            y_end = y_start + size * math.sin(angle * math.pi / 180)

            image = cv2.arrowedLine(img=image, pt1=(round(x_start), round(y_start)), pt2=(round(x_end), round(y_end)), color=(0, 255, 0), thickness=1, line_type=8, shift=0, tipLength=0.1)

    plt.imshow(image)
    plt.title('Image - UoH')
    plt.xticks([])
    plt.yticks([])

    plt.get_current_fig_manager().window.state('zoomed')
    plt.show()
