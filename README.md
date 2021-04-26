# ComputerVision
## Description
This repo contains homeworks done as part of the ImageProcessing course in Haifa University, Israel</br>
In order to receive the results for each HW, simply run script.m file

### HW 1
#### Hough-Transform
Using <b>Hough-Transform</b> and <b>Voting algorithm</b>, we detect equaliteral triangles by voating for the center and the angle of the triangle</br>
<img src="HW_1/Q_1/tests/test_3_results.png" width="1000">
Detecting many triangles we receive many false-positive detection, which could be avoided by applying non-maximal-supression algorithm:
<img src="/HW_1/Q_1/triangles_1/image003_results.png" width="1000">
#### SIFT
Using <b>SIFT</b> we can find <b>Keypoints</b> and their <b>Descriptors</b> in images, and find matches between keypoints by descriptors distance using the following tests:</br>
1. Second closest match test
2. Dual match test
<img src="/HW_1/Q_2/Part_1/Results.png" width="1000">
