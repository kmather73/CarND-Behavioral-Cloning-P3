#**Behavioral Cloning** 


**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[image1]: ./examples/placeholder.png "Model Visualization"
[image2]: ./examples/placeholder.png "Grayscaling"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
###Files Submitted & Code Quality

####1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* commaModel.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model_final.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results

####2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model_final.h5
```

####3. Submission code is usable and readable

The commaModel.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model.

###Model Architecture and Training Strategy

####1. An appropriate model architecture has been employed
My model is a convolutional neural networks with three convolutional layers followed by two fully connected layers. Its build from the steering angle model built by [comma.ai](https://github.com/commaai/research/blob/master/train_steering_model.py). 


The model includes ELU layers and dropout to introduce nonlinearity, and the data is normalized in the model using a Keras lambda layer. 

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 160x320x3 Colour image    					| 
| Convolution 8x8     	| 4x4 subsampling 								|
| ELU			      	| 							 					|
| 						|												|
| Convolution 5x5	    | 2x2 subsampling								|
| ELU					|												|
| 						|												|
| Convolution 5x5	    | 2x2 subsampling								|
| Flatten 				| 												|
| Dropout				| .2 dropout probability						|
| ELU					|												|
|						|												|
| Fully connected		| output 512   									|
| Dropout				| .5 dropout probability						|
| ELU					|												|
|						|												|
| Fully connected		| output 1   									|

####2. Attempts to reduce overfitting in the model

The model contains ELU and dropout layers in order to reduce overfitting. 

The model was trained and validated on different data sets to ensure that the model was not overfitting (commaModel.py line 273-288). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

####3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (commaModel.py line 268).

####4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road 

For details about how I created the training data, see the next section. 

###Model Architecture and Training Strategy

####1. Solution Design Approach

The overall strategy for deriving a model architecture was to trail and error.

My first step was to use a Nvidia convolution neural network model that was suggested, this model took a long time to train even on a AWS instance and did not prefrom well. Next I went the comma.ai model and reduced the size of the image which reduced traning time and prefromed much better.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my final model had a low mean squared error on the training set and a low mean squared error on the validation set. This implied that the model was not overfitting. 

The final step was to run the simulator to see how well the car was driving around track one. At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

####2. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving once going clockwise and the other going counter clockwise.
I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to react if it ever encountered these kinda of situations. Then I also captured recordings of on track two in order to get more data points. 

I then merged my data with the udacity data set.


To augment the data set I split up the images into three categories turing left, turning right, and going straight. I then supersampled each class so that there would be roughly the same number of instance of each class. I also flipped the left, right, and center camera images to increase the data set more.

After the collection process, I had 150987 number of data points. I then after the preprocessed this data set grow to a size of 724608

I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was between 5-8 as evidenced by the fact that the validation loss started to fluctuate. I used an adam optimizer so that manually training the learning rate wasn't necessary.
