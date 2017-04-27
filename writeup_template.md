**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./examples/car_not_car.png
[image2]: ./examples/HOG_example.jpg
[image3]: ./examples/slidingWindow_1.png
[image4]: ./examples/slidingWindow_2.png
[image5]: ./examples/bboxes_and_heat.png
[image6]: ./examples/labels_map.png
[image7]: ./examples/output_bboxes.png
[video1]: ./project_video.mp4
[image8]: ./examples/hog_car_ch1.png
[image9]: ./examples/hog_car_ch2.png
[image10]: ./examples/hog_car_ch3.png
[image11]: ./examples/hog_NOTcar_ch1.png
[image12]: ./examples/hog_NOTcar_ch2.png
[image13]: ./examples/hog_NOTcar_ch3.png
[image14]: ./examples/orignalAndPostColorConversion.png
[image15]: ./examples/orignalAndPostColorConversionNotcar.png
[image16]: ./examples/slidingWindow_3.png
[image17]: ./examples/test_image_example1.png
[image18]: ./examples/test_image_example2.png
[image19]: ./examples/test_image_example3.png

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

###Histogram of Oriented Gradients (HOG)

####1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step is contained in lines 17 through 127 of the file called `ExtractFeaturesAndTrain.py`.  

I started by reading in all the `vehicle` and `non-vehicle` images.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![alt text][image1]

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here is an example using the `YUV` color space and HOG parameters of `orientations=11`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:

Car

![alt text][image14]

Channel 1

![alt text][image8]

Channel 2

![alt text][image9]

Channel 3

![alt text][image10]

Not Car

![alt text][image15]

Channel 1

![alt text][image11]

Channel 2

![alt text][image12]

Channel 3

![alt text][image13]

####2. Explain how you settled on your final choice of HOG parameters.

I tried various combinations of parameters and found that using the YUV color space improved detection on concrete (white) sections of road and helped detect the white vehicle. Previously, only the black vehicle was being detected. 

####3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

The code for this step is contained in lines 128 through 167 of the file called `ExtractFeaturesAndTrain.py`.  I trained a linear SVM using the extracted features of cars and not cars. I split up the features into a training set (80%) and test set (20%). After training my svc, I evaluated the accuracy of my model using the test set. The accuracy was greater than 99%. 

###Sliding Window Search

####1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

The code for this step is contained in lines 24 through 89 of the file called `VehicleDetection.py`. Rather than creating two separate functions for a sliding window search and second function to get features, I combined the two using a Hog Sub-sampling Window Search. Towards the bottom of the image, I chose larger scales because the vehicles appear larger in the image. Towards the horizon (middle of the image), I used smaller scales because vehicles appear smaller at a distance. I used three different sliding windows, each with a unique scale and vertical span. 

![alt text][image3]

![alt text][image4]

![alt text][image16]

####2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched on three scales using YUV 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result.  Here are some example images:

![alt text][image17]
![alt text][image18]
![alt text][image19]
---

### Video Implementation

####1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./output.mp4)


####2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  

Here's an example result showing the heatmap from a single frame of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the video:

### Here are all of the bounding boxes from the three sliding window passes, the corresponding heatmap, and the final bounding box selection:

![alt text][image5]
---

###Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

A challenge I faced initially was trying to understand why I was seeing different behavior in the video than in still images. My pipeline was working correctly on images, but the bounding boxes in the video would start out looking good, and then eventually grow to cover the lower half of the screen. It turned out I was not zero-ing the bounding box list in between frames--an issue I would not and did not see when testing on single images. 

Another challege was properly tuning the scale factor and vertical limits for the sliding window approach. This took some trial and error. 

Initally my alogorithm was not very robust. I ultimately traced this to my SVM only being trained on 500 samples. Once I increased the samples to 2800, I saw much better performance. The limit was intially imposed in the training lesssons to prevent a timeout. 

To make my pipeline more robust, I could threshold the heat map across consecutive frames. 

