# ComputerVision
## Description
This repo contains homeworks done as part of the ImageProcessing course in Haifa University, Israel</br>
In order to receive the results for each HW, simply run script.m file

## HW 1
### Hough-Transform
Using <b>Hough-Transform</b> and <b>Voting algorithm</b>, we detect equaliteral triangles by voating for the center and the angle of the triangle</br>
<img src="HW_1/Q_1/tests/test_3_results.png" width="1000">
Detecting many triangles we receive many false-positive detection, which could be avoided by applying non-maximal-supression algorithm:
<img src="/HW_1/Q_1/triangles_1/image003_results.png" width="1000">
### SIFT
Using <b>SIFT</b> we can find <b>Keypoints</b> and their <b>Descriptors</b> in images, and find matches between keypoints by descriptors distance using the following tests:</br>
1. Second closest match test</br>
2. Dual match test</br>
<img src="/HW_1/Q_2/Part_2/Results.png" width="1000"></br>
### Warping and RANSAC
For <b>Warping</b> of image into another, we search for that best fitting homographic transformation matrix, which we calculate from the match between keypoints in the two images, but since the match is not 100% ideal, we use <b>RANSAC</b> to find the outlier matches and calculate the matrix upon the inliers only
<img src="/HW_1/Q_3/Part_2/Results.png" width="1000">
