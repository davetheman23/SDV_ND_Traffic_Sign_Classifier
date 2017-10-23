# **Traffic Sign Recognition** 

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./example_training_images.png "Example Training Data"
[image2]: ./hist_of_training_dataset.png "Histogram (Training)"
[image3]: ./hist_of_validation_dataset.png "Histogram (Validation)"
[image4]: ./hist_of_testing_dataset.png "Histogram (Testing)"
[image5]: ./preprocessing_image_examples.png "Preprocessed Image"
[image6]: ./distruction_of_pixel_values_before_after.png "Distribution of Pixel Values"
[image7]: ./image1_ahead_only.jpg "Ahead Only with Occlusion"
[image8]: ./image2_children_cross.jpg "Children Crossing"
[image9]: ./image3_stop_sign.jpg "Stop Sign"
[image10]: ./image4_120_speed_limit.jpg "120 speed limit"
[image11]: ./image5_road_work.jpg "Road Work"
[image12]: ./test_images_german_traffic_signs_preprocessed.png "Processed Web Images"

---
### Writeup / README

#### 1. Project Code

Here is a link to my [project code](https://github.com/davetheman23/SDV_ND_Traffic_Sign_Classifier/blob/master/Traffic_Sign_Classifier.ipynb) in github

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the basic numpy functionalities to calculate summary statistics of the traffic signs data set:

* The size of training set is **34799**
* The size of the validation set is **4410**
* The size of test set is **12630**
* The shape of a traffic sign image is **(32, 32, 3)**
* The number of unique classes/labels in the data set is **43**

#### 2. Include an exploratory visualization of the dataset.

First, I have examined some of the images via randomly selecting them from the training dataset. And here is a list of 10 randomly picked images, with their labels.

![image1]

Second, I have looked at the distribution of each of the label classes in the training, validation and testing datasets as below. Which, I found, that the distribution is pretty similar over the three datasets. 

![image2] ![image3] ![image4]

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. 

I have implemented three methods of preprocessing of the image data:
* grayscaling
* normalization
* sharpening

As a first step, I decided to convert the images to grayscale because there are fewer channels in grayscale comparing to color images. This means a reduction in the input dimension, so the number of weight parameters in the input and 1st convolutional layer are only about 1/3 of that without the grayscale. From the traffic sign images (unlike traffic signal images), color doesn't seems to be the main distingushing factor to determine the class of the signs. 
Secondly, I used normalization to bring the scale of the image pixel values to the range between -1.0 to +1.0. And this reduced range aims to keep the gradient small.
Thirdly, I used a histogram equalization technique to sharpen the image, i.e. increase the contrast of the image, which make the image to be recognizable. 

Here is an example of a traffic sign image after applying these preprocessing.

![image5]

Furthermore, I plotted out the distribution of the pixel values before and after the preprocessing, and see that the distribution is nicely uniform. 

![image6]

#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer                 |     Description                               | 
|:---------------------:|:---------------------------------------------:| 
| Input                 | 32x32x1 grayscale image                       | 
| Convolution 5x5       | 10 filters, 1x1 stride, same padding, outputs 32x32x10    |
| RELU                  |                                               |
| Max pooling           | 2x2 stride,  outputs 16x16x10                 |
| Convolution 5x5       | 20 filters, 1x1 stride, same padding, outputs 16x16x20    |
| RELU                  |                                               |
| Max pooling           | 2x2 stride,  outputs 8x8x20                 |
| Fully connected       | input 1280 and output 200 neurons, dropout enabled                                          |
| RELU                  |                                               |
| Fully connected       | input 200 and output 100 neurons, dropout enabled                                          |
| RELU                  |                                               |
 | Fully connected       | input 100 and output 43 neurons                                          |


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used an the following hyperparameters:

| Hyper-Parameters      |     Description                               | 
|:---------------------:|:---------------------------------------------:| 
| Optimizer             | tf.train.AdamOptimizer                        |
| learning rate         | 0.001 constant                        |
| batch size            | 128                            |
| epochs                | 30                        |
| weight initialization | truncated_normal, mu = 0, sigma = 0.1         |
| bias initialization   | zeros         |

####4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 99.891%
* validation set accuracy of 95.488%
* test set accuracy of 93.618%

To start, the LeNet architecture was chosen with a relatively higher learning rate (i.e. 0.1), no dropout was used. Choosing LeNet was because in the previous udacity lab, it was proven to be useful in recognizing the handwriting numerals. 

In particular, the LeNet architecture has a couple of convolutional layers that learns different low level image features, such as lines, circles, retangular shapes. The convolutional layer uses filters to focus on a small region of the picture and use the same weights to get activations across the spatial dimension of the image. This way, whatever the filter learned in one part of the image help recognize similar features in aonther part of the image. And this spatial abstraction ability makes the convolution layer very powerful in image recognition tasks. 

With high learning rate, it shoots up to a high accuracy relatively quickly, but it oscilate between good and no so good, didn't seem to quick converge. Then I tried a lower rate (0.001), it converges slower, but it does seem to converge. A better approach which was not implemented was to have a dynamically adjustable learning rate, which should start high, then decade over EPOCHs. 

I have also been printing out training accuracy along with the validation accuracy. And I found patterns that the training accuracy is high 99.4% but the validation accuracy is just 88%. That's an indication that the generalization ability of the model is low. So dropout can help that since it is a very effective techniques to prevent overfitting. 
 
### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that is included in the testset images.

![image7] | ![image8] | ![image9] | ![image10] | | ![image11]

And these images are preprocessed into the following images

![image12]

I am picking some of the very hard images, at least I think maybe hard for the CNN to figure out. These are some of the difficulties I see in these images:
* The 1st image is occluded about 20%
* the 2nd image is distorted slightly, and has a little character overlay in the bottom of the image
* the 3rd image is rotated
* the 4th image is not frontfacing the camera
* the 5th image is very blurry and pixelated.


#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set.

Here are the results of the prediction:

| Image                 |     Prediction                                | 
|:---------------------:|:---------------------------------------------:| 
| Road work             | Road work                                     | 
| Ahead only                | Ahead only                                |
| Speed limit (120km/h)      | Speed limit (120km/h)                  |
| Children crossing     | Children crossing                              |
| Stop                   | Stop                                 |


The model was able to correctly guess 4 out of 5 times (80%), and missing only the one sign that is very tricky to predict, which is not forward facing the camera. And comparing to the testset that has > 93% accuracy, they are very comparable to the testset. 

####3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. 

The code for making predictions on my final model is located in the 124th cell of the Ipython notebook.

Let's examine closer on the image that the model predicted wrong, the one sign of speed limit (120km/h). For this image, the model predicted with 82% of confidence that the class is End of all speed and passing limits. I think it is getting the circle correctly, but hard to actually see the character inside, since there is an angle to the number 120. And to be fair that this is not something that the CNN network has see before, almost all of the training data are directly facing the camera. 

All the rest of the prediction are of the speed limits. So with more training of those sideview images, I believe the network should be able to predict it correctly. 


| Probability           |     Prediction                                | 
|:---------------------:|:---------------------------------------------:| 
| 0.820 | End of all speed and passing limits |
| 0.097 | Speed limit (20km/h) |
| 0.048 | End of speed limit (80km/h) |
| 0.013 | Speed limit (60km/h) |
| 0.010 | Speed limit (120km/h) |


For the second image ... 

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
####1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?


