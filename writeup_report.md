# **Behavioral Cloning** 

## Writeup Report


---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/model_visualization.jpg "Model Visualization"
[image2]: ./examples/center_2018_03_28_21_40_10_134.jpg "Center image"
[image3]: ./examples/center_2018_03_28_21_43_53_108.jpg "Recovery Image"
[image4]: ./examples/center_2018_03_28_21_43_55_377.jpg "Recovery Image"
[image5]: ./examples/center_2018_03_28_21_43_56_010.jpg "Recovery Image"
[image6]: ./examples/placeholder_small.png "Normal Image"
[image7]: ./examples/placeholder_small.png "Flipped Image"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

In the beginning, I choosed to use LeNet as my model, but it was difficult to keep the car inside the drivable road, so I used the nVidia Autonomous Car Group model in the end. This model consists of 3 convolution neural network with 5x5 filter sizes, 3 convolution neural network with 3x3 filter sizes and 4 fully-connected layers. 

The model includes RELU layers to introduce nonlinearity, and the data is normalized in the model using a Keras lambda layer. Besides, the data is cropped in the model using a cropping2D layer to remove the region in the images that's not relevent. 

#### 2. Attempts to reduce overfitting in the model

In order to reduce overfitting, the epochs was set less than 7 during training, and the model was trained and validated on different data sets to ensure that the model was not overfitting. The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually. 

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road, and driving smoothly around curves.

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

My first step was to use a convolution neural network model similar to the LeNet, I thought this model might be appropriate because it had good performance at traffic sign classification application. By using the LeNet model, the car did pretty good at the beginning, but went straight to the lake around the first big curve.

So I tried the nVidia Autonomous Car Group model instead , because this model was more complicated and might learn better features than LeNet. 

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that this model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting, and the car passed through the bridge, but ran outside of the road during the next curve in the simulator.So I recordeded more training data by driving the car smoothly around curves in the simulator, besides, I added left and right cameras into the training set by adding/substracting a steering offset. 

The final step was to run the simulator to see how well the car was driving around track one. After tuning the hyperparameters of the model, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

Here is a visualization of the nVidia Autonomous Car Group model architecture I used, which consists a convolution neural network with the following layers and layer sizes.

![alt text][image1]

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image2]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to recover when it's not in the center of the drivable area.These images show what a recovery looks like starting from the right side :

![alt text][image3]
![alt text][image4]
![alt text][image5]


To augment the data sat, I also added left and right cameras into the training set by adding/substracting a steering offset thinking that this would help the vehicle to learn to recover from the left and right sides. The steering angle offset I used was 0.32.

After the collection process, I had 29184 number of data points. I then preprocessed this data by lambder layers in order to monalize the images and to crop irrelevent pixels.

I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 5 as evidenced by the model mean squared error loss figure. I used an adam optimizer so that manually training the learning rate wasn't necessary.
