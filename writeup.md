---
typora-root-url: ./
---

## Advanced Lane Finding Project

---

****

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

## Camera Calibration

#### Camera intrinsic matrix and distortion coefficients

The code that calculates the focal distances, principal points and the distortion coefficients can be found in *calculate_camera_matrices()* in **camera_calibration.py**, along with the code to save and reload these parameters to/from a pickle file.

For each chessboard pattern image, the image is converted to grayscale and subsequently the internal chessboard corners are searched for, by the OpenCV function *findChessboardCorners()*. If the chessboard corners were successfully detected, then chessboard corner coordinates ("image points") are appended to the array *image_points*. For each successful detection, the "object points" that are the real world coordinates (we assume the white wall is the z=0 plane) is stored as well: this a regular grid of 9 x 6 (the grid size is a parameter, dependent on the size of the chessboard). They are appended to the array *object_points*.

![chessboard_corners](/writeup_images/chessboard_corners.jpg)

The arrays *image_points* and *object_points* are provided to the OpenCV function *calibrateCamera()* that returns the camera intrinsic matrix and distortion coefficients (we are not interested in the rotation and translation vectors as they only apply for the chessboard pattern images).

If the module is run as a stand-alone program, then the camera calibration is run using the provided chessboard patterns from the **camera_cal** folder, with grid size 9 x 6. If the calibration was successful, the results are stored in the folder **camera_cal/calibration.data.py**. As a cross check, the first chessboard pattern from the folder is undistorted by OpenCV's *undistort()* function and the result is displayed. 

![camera_calibration](/writeup_images/camera_calibration.jpg)

#### Perspective matrix for top view of the road

The curvature of a lane is derived easier when the road is viewed upon from above (90°). We need a perspective matrix that transforms the road images from inside the car to images that have a bird's eye view on the road.

It is assumed that the position and orientation of the camera in the car is the same in all the provided clips. Therefore, it suffices to calculate the perspective matrix once and apply this perspective matrix on all the road images.

In order to calculate this perspective matrix, it is necessary to find a collection of source points in the original image and be able to define their respective image coordinates in the bird's eye view image ("destination points").

I loaded the image **test_images/straight_lines2.jpg** and undistorted it using the camera intrinsic matrix and distortion coefficients given back by *load_calibration_params()* from **camera_calibration.py**. The resulting image is particularly useful since it depicts a straight lane on a flat road surface. It is much easier to pick good destination points, because we know that given previous situation, the trapezoid lane will be converted to a rectangular shape in the warped image.

The code to calculate this perspective matrix - given the source points and the destination points defined by an offset from the side of the image - is provided in *calculate_perspective_transform()* in **unwarp.py**. The transformation matrix is derived using OpenCV's *getPerspectiveTransform()* function. 

If the module is run as a stand-alone program, then the perspective matrix is calculated using the source points and destination points in the table below:

|  Source   | Destination |
| :-------: | :---------: |
| 585, 456  |   300, 0    |
|  699,456  |   980, 0    |
| 1055, 685 |  980, 720   |
| 266, 685  |  300, 720   |

I determined the source and destination points manually (using Irfanview). Note that I made sure the original shape was a trapezoid (with the hood of the car left out) and the destination shape is a rectangle.

The trapezoid shape in the original, undistorted image and the rectangular shape in the warped image are shown here:

![perspective_matrix](/writeup_images/perspective_matrix.jpg)

The perspective matrix and its inverse is stored in a pickle file **camera_cal/perspective_matrix.p** with the function *save_perspective_matrices()* and finally a cross check validation is made with the image **test_images/straight_lines1.jpg** using the OpenCV function *warp_perspective()*

![perspective_matrix_cross_check](/writeup_images/perspective_matrix_cross_check.jpg)

### Pipeline for single images

#### 1. Overview

The pipeline for each image can be found in *process_frame()* in **process_frame.py**

First off, each image is undistorted and warped to bird's eye view with the provided camera parameters (see **CameraParameters.py**). We discussed already how this is done in the previous sections.

The resulting image is then filtered so that only points are selected that are good candidates for lane line points. These points are stored in a binary image of similar shape as the undistorted, warped image.

![filtering](/writeup_images/filtering.jpg)

The next step tries to find out which of these candidates belong to the left lane, the right lane and no lane at all. Once the points are grouped, a polynomial regression is calculated that fits these point clouds best. 

![lane_fitting](/writeup_images/lane_fitting.jpg)

Next, the candidate fits are checked if they are good candidates or not, dependent on their individual coefficients, but also if the two candidates together make sense or not. If they don't, they are replaced with a previous best fit. The fits are also smoothened a bit, taking the previous fit into account.

Once it is decided which fits to use, the lane is drawn on the undistorted, non-warped image using the lane line fits and the inverse perspective matrix.

Finally, also the radius of curvature and the position of the vehicle towards the centre of the detected lane is calculated and displayed on the image.

Some of the calculations that were made for each lane line fit during a pipeline cycle are stored in *LaneLineHistory* instances (see **LaneLineHistory.py**) and bundled into a *LaneHistory* instance (e.g. the previous best fit).

#### 2. How parameters were identified in the pipeline

The filtering, lane fitting and lane candidate validation steps use quite some parameters. These parameters were fine-tuned on a series of selected test images. All frames of the three videos were saved as JPEG files (**save_movie_images.py**) and then a selection was made of images where it was visible upon first sight that detection of the lane would experience some difficulties. These images were stored (next to the ones provided already by Udacity) in the **test_images** folder.

The three aforementioned algorithms contain parameters that allow to visualize the end results of the algorithms and thus deduce the consequence of these parameter values. 

**process_frame.py** runs the pipeline on each of these images in sequence when started as a stand-alone program.

#### 3. Colour and gradient filtering 

The undistorted, warped image is converted to a binary image (black/white pixels only) where pixels are selected based on their colour and gradient properties. 

##### 3.1 Colour spaces

Three colour spaces were experimented with:

- RGB
- HSV
- HLS

The different channels from these colour spaces were visualized for each of the generated test images. The three channels from HLS, the saturation channel from HSV and the red channel from RGB looked promising to be able to linearly discern lane pixels from non-lane pixels. 

##### 3.2 Histogram equalization

I investigated how histogram equalization could help me to be able to tackle different light conditions (bridges, tunnels, shadows, overexposure). 

The site https://docs.opencv.org/3.1.0/d5/daf/tutorial_py_histogram_equalization.html was an interesting read and showed that CLAHE (Contrast Limited Adaptive Histogram Equalization) could help us in situations where there are different light conditions in the same image. This technique increases contrast locally (in tiles of a given size) and with a maximum clipping size. OpenCV offers the *createCLAHE()* factory to create a CLAHE instance and this can then be applied on an image.

CLAHE was applied on the red channel from the RGB colour space and on the saturation channels of the HLS and HSV colour spaces.

##### 3.3 Colour filtering

Pixels were filtered according to their colour channel value with the following rule:

A pixel is a candidate lane pixel if:

- its locally equalized red channel value (from RGB) is between 210 and 255
- or: its hue channel value (from HLS) is between 0 and 40, its lightness value is between 80 and 255 and its locally equalized saturation value is between 150 and 255

The functions *color_filter()* and *color_filter3()* from **image_filter.py** perform the filtering.

##### 3.4 Gradient filtering

Pixels were filtered according to their local gradient in selected colour channels with the help of a Sobel filter with kernel size 15.

First off, gradients in the X and Y directions were calculated and then these values are used to derive the gradient magnitude and direction. Note that since the pixels are not square entities in the real world due to the warping, so direction and magnitude are relative. 

Pixels are selected with a given threshold to produce a binary image. The produced binary images from the gradient filters on different colour channels are then combined to produce a final binary image.

Due to the fact that only edges are detected (even if we apply a bigger Sobel kernel) and that we want full lane lines in our binary images, there is also an option to apply morphological operations on the binary images (see https://docs.opencv.org/trunk/d9/d61/tutorial_py_morphological_ops.html): first an opening operation is applied to remove noise (edges caused by dirt or sudden local colour differences on the asphalt) and then a closing operation is applied to fill up the lane lines.

Following gradient filters were applied:

- Equalized saturation channel (from HSV) with Sobel kernel size 15, gradient direction threshold in between 0 and 1.0, gradient magnitude minimum of 30 and final morphological closing with (square) kernel size 11 (opening is fixed at kernel size 3)
- Equalized saturation channel (from HLS) with Sobel kernel size 15, gradient direction threshold in between 0 and 1.0, gradient magnitude minimum of 30 and no morphological operations
- Equalized red channel (from RGB) with Sobel kernel size 15, gradient direction threshold in between 0 and 1.0, gradient magnitude minimum of 30 and final morphological closing with (square) kernel size 11 (opening is fixed at kernel size 3)

##### 3.5 Colour and gradient filters combined

The colour filters usually produced better (clearer) results, so in many cases only the results of the (combined) colour filters were used.

The image below shows such as an example. The "All masks combined" subplot shows the final binary image that is used in the next step. It has to be noted that some gradient filters (the saturation channel from the HLS colour space in this example) also produces good results. The algorithm might be extended to detect which gradient filters produce noisy results and which ones not and based on that, choose only a subset of the binary images to combine.

![color_filtering_only](/writeup_images/color_filtering_only.jpg)

When the colour filtered binary image produces noisy results, then also the gradient filter is used. The final binary image is then an OR-combination of the colour and the gradient filters. The check for "noisy results" is currently implemented in a (too) simple way: if the colour filtered binary image contains more than 20000 pixels, then the gradient filters are used as well.

![color_and_gradient_filtering](/writeup_images/color_and_gradient_filtering.jpg)



#### 4. Pixel selection per lane line 

The next step tries to find which of the pixels in the binary image are candidates for the left and the right lane line.

A histogram is made of the bottom half of the binary image on the X axis (so all pixel values for a given column are summed up and this for each column in the bottom half). If the lane lines are well visible on the binary image and their slope is quite steep (so no big turns), then the histogram should show peaks where the lanes are.

The lanes can be found in the histogram below around column 350 and column 1000.  

![histogram_peaks](/writeup_images/histogram_peaks.jpg)

The peaks can be the starting point for the sliding window fit operation explained in the project materials. A window is centred around the histogram peaks with a given size. I have chosen a window height of 1/9th of the image and a width of 200 pixels. Note that the choice for a good window size is dependent on the perspective transform to the bird's eye view.

The first window for each lane starts on the bottom of the image (although I assume it actually would be better to start off in the middle of the bottom half of the image). There the points that fall in the window are selected. We filter out the points that are more than two standard deviations away from the mean of the point cloud (in X direction). Also, if the number of points is smaller or bigger than certain thresholds (minimum: 80 and maximum: 5000), then all the points in the current window are not used at all.

Principal component analysis (PCA) is used to find the first principal component of the point cloud inside the window (the API I used is based on https://jakevdp.github.io/PythonDataScienceHandbook/05.09-principal-component-analysis.html). 

The first principal component should ideally give us the direction of the lane line. We check if the principal component is well suited by applying a threshold (0.7) on the explained variance ratio of the point cloud. If the first principal component doesn't meet this standard, then the point cloud is too noisy and the entire point cloud inside the window is dismissed (and thus not regarded as part of a lane line). 

![pca_sliding_window](/writeup_images/pca.jpg)

The next search window for each line is then shifted up (each time 1/9th of the image), but also shifted horizontally with the following rule:

- If the first principal component has enough explained variance ratio and its slope is bigger than 0.4 (so not too flat, since this would be strange for a lane line), then the sliding window is shifted horizontally according to the principal component vector
- Else if the previous search window didn't contain too many or too little pixels, then the mean in the X direction of the point cloud of the selected points (selection based on standard deviation from the mean) is used as the new centre for the next search window
- Else: don't shift the sliding window horizontally

An example of this sliding window technique is shown below:

![pca_sliding_window](/writeup_images/pca_sliding_window.jpg)

We clearly see that some points are not selected due to the fact that they are more than 2 standard deviations away from the mean (left lane). We also see that some of the point clouds in the sliding windows are dismissed thoroughly, due to the constraint on the minimum of pixels in the second lowest right lane window or due to the constraint on the explained variance ratio of the first principal component (lowest sliding window of the right lane).

The following example also shows the effect of the first principal component on the horizontal shifting of the sliding window:

![pca_sliding_window2](/writeup_images/pca_sliding_window2.jpg)

The left sliding window shifts better than by just taking the mean. The given example shows also clearly where it still could go wrong in case there is too much random noise: the mean was chosen for the horizontal shift on the middle sliding window of the right lane and from then off the wrong direction was chosen.

All the selected points of all vertically sliding windows for each line are appended and stored for the next step (candidate lane fitting).

#### 5. Lane line fitting 

The candidate points for each lane line were selected in the previous step. We try to fit two curves on these two point sets by second degree polynomial regression. This is achieved by Numpy's *polyfit()* function.

![lane_fitting](/writeup_images/lane_fitting.jpg)

An important condition to fit a curve is that at least points were selected in three sliding windows for each respective lane line. Else, no fit is performed.

The code and documentation for steps 4 and 5 can be found in *fit_polynomial_sliding_window()* in **sliding_window_fit.py**. 

#### 6. Selection of lane line fits

The next step checks if the two lane line candidates (if any) are good candidates at all. 

If no fit was found for a particular lane line, then we select the previous best fit for that lane line as the current lane line candidate. If the lane candidate has a fit so that its X coordinate on the bottom row differs more than 100 pixels from the previous fit, then the lane line candidate is rejected.

The two lane line candidates are both rejected if:

- their slopes differ too much
- the distance in between the lane lines on 2/3 of the height of the warped image is smaller or larger than pre-defined thresholds (400 and 800 respectively)
- if the lane lines intersect in the image or just outside the image (with a margin of 200 pixels)

When both lane line candidates are rejected, then they are replaced with their previous best fits. If any of the lane lines were rejected in 20 subsequent frames, then no lane lines will be accepted and no lane will be drawn on the end result.

The final lane line candidates are then smoothened a bit (taking the current candidate's polynomial coefficients into account for 80% and the previous best fit for 20%) for more stable lane drawing (this could improve a lot !)

The selection of the lane line candidates has also a consequence for the placement of the first (bottom) sliding window in the next frame (step 4). If a lane line candidate was approved, then its centre is the X coordinate on the bottom row of the current fit and no histogram will be calculated anymore to estimate the centre. If on the other hand, the lane line candidate is not selected for three subsequent frames, then the histogram procedure is used again to find the centre of the bottom sliding window.

*process_lanes()* in **process_frame.py** contains the code for aforementioned checks. 

#### 7. Radius of curvature of the lane and the vehicle position

The calculation for the radius of curvature of the lane and the position of the vehicle with respect to the lane centre is calculated in *calculate_radius_of_curvature_and_lane_offset()* in **process_frame.py**. The code is heavily based on the code already provided by Udacity, but adapted such that it makes use of the *LaneLineHistory* instances that were already used in steps 4 - 7.

#### 8. Drawing of the lanes and calculated data

The lanes are drawn on the undistorted, non-warped image in *draw_lanes_on_image()* in **process_frame.py**

The radius of curvature and the vehicle position is shown as text in the *process_frame()* function itself.

Here are a couple of results of this final step:

![lane_drawing](/writeup_images/lane_drawing.jpg)

![lane_drawing2](/writeup_images/lane_drawing2.jpg)

---

### Pipeline (video)

The pipeline for the video make use of the *VideoFileClip* class of the *moviepy.editor* library and can be found in *process_road_movie()* in **process_movie.py**

Before the video processing starts, the camera intrinsic matrix, distortion coefficients and the perspective matrix is loaded from their respective pickle files. The video pipeline calls *process_frame()* for each MPEG-4 frame along with the loaded matrices and vectors (through a lambda function). 

Here's a [link to my video result for the project video](./project_video_with_filtered_lanes.mp4)

Here's a [link to my video result for the challenge video](./challenge_video_with_filtered_lanes.mp4)

Here's a [link to my video result for the harder challenge video](./harder_challenge_video_with_filtered_lanes.mp4)

---

### Project files

- **camera_calibration.py**: Functions that perform camera calibration with chessboard images and loads and stores this data
- **camera_parameters.py**: Class definition to store all important matrices and vectors that undistort and warp camera images
- **color_transform.py**: Helper module to visualize test images in different colour spaces
- **image_filters.py**: Functions for colour and gradient filtering and combining the results
- **lane_line_history.py**: Class definition to store temporary calculations during the lane line point selection, curve fitting and lane line detection algorithms
- **process_frame.py**: Functions that processes a single frame (pipeline for a single frame)
- **process_movie.py**: Function that processes an entire movie
- **save_movie_images.py**: Helper module that stores all MPEG-4 frames from a video in JPEG format
- **sliding_windows_fit.py**: Functions that perform the lane line point selection and curve fitting, based on the sliding window technique
- **unwarp.py**: Functions that calculate, load and store the transformation matrices that warp undistorted images to bird's eye view and vice versa

------

### Discussion on shortcomings and possible improvements

The project only scratches the surface of what is necessary to make the pipeline really robust.

The perspective matrix is fixed at the moment. This gives problems for example on the harder challenge video where in my opinion, the height and the width of the source trapezoid needs to be different: too much of the forest is inside the warped image and also the bends are too sharp so that much of the lane lines fall outside of the picture (a bigger margin needs to be taken for the destination rectangular shape). The transform could be adapted using some kind of tilt sensor inside the car, so that the height of the source trapezoid could be adapted.

The perspective transformation also doesn't result in square pixels, so that for example gradient direction and principal component angles do not match with the real world. An extra calibration step could resolve this (already using the camera's position and orientation in the car).

Another problem is that the parameters used in the sliding window algorithm are also dependent on the perspective matrix that is used. Therefore, they should also be adjusted automatically.

The filtering step is certainly something that still could be improved a lot, since a lot of noise is still produced (for example on the hand picked test images). The filtering could be improved by applying a region of interest, so that a lot of noise is already removed on the sides of the image. Another idea would be to only accept lane line points that are in the vicinity of "asphalt" colours. Finally, it would be interesting to see how well convolutional neural networks work in the detection of lane lines. 

It has to be noted though that one of the basic rules of computer vision, is that the detection algorithms can only be as good as the source image. Even though countermeasures have been taken in the filtering step to tackle different light conditions, I assume that a camera with AGC (automatic gain control) would improve some situations. If an image experiences overexposure (like in the harder challenge video), then no detection algorithm can recover this lost data.

The sliding window technique still misses some important lane pixels sometimes. If the binary images would be cleaner, it would be interesting to shift the sliding window with the same offset as the previous shift if no good point data was found. This would result in better sliding window shifts in case of a broken lane line in bigger turns. Unfortunately, since the binary image contained too much noise, this was disabled so that noise could not steer the sliding window too much in the wrong direction.

The lane fitting could also improve: higher order polynomials could be used, especially in situations where multiple bends in different directions are in one single image. 

The lane candidate selection and smoothening of the lanes is still quite primitive and could be improved a lot (as explained in these sections)

Finally, the image pipeline is quite slow. There are a couple of factors for that: for some algorithms CUDA could be used as they are highly parallellizable (e.g. the filters and histogram calculation). Also, PCA is not particularly CPU friendly.