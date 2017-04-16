import os
import csv
import cv2
import random
import json
import numpy as np
import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split
from pathlib import Path

def filter_df(df, x):
	left_angle = -0.125
	right_angle = 0.125
	stright_angle = 0.000125

	df_left = []
	df_right = []
	df_stright = []

	super_sample_right_rate = 11
	super_sample_left_rate = 11
	super_sample_stright_rate = 2
	for i in range(len(df)):
		center_img = df["center_image"][i]
		left_img = df["left_image"][i]
		right_img = df["right_image"][i]
		angle = df["steering_angle"][i]

		# Turning Left
		if angle < left_angle:
			df_left.append([center_img, left_img, right_img, angle, x])
			for i in range(super_sample_left_rate):
				new_angle = angle * (1.0 + np.random.uniform(-1, 1)/30.0)
				df_left.append([center_img, left_img, right_img, new_angle, x])
		
		# Turning Right
		elif angle > right_angle:
			df_right.append([center_img, left_img, right_img, angle, x])
			for i in range(super_sample_right_rate):
				new_angle = angle * (1.0 + np.random.uniform(-1, 1)/30.0)
				df_right.append([center_img, left_img, right_img, new_angle, x])

		# Going stright
		#elif True or angle != 0 or abs(angle) > stright_angle or np.random.choice(np.arange(0, 2), p=[0.5, 0.5]) == 10:
		elif True or angle != 0:
			df_stright.append([center_img, left_img, right_img, angle, x])
			for i in range(super_sample_stright_rate):
				new_angle = angle * (1.0 + np.random.uniform(-1, 1)/30.0)
				df_stright.append([center_img, left_img, right_img, new_angle, x])
	return df_left, df_right, df_stright
	
def load_data():
	# Reading my data
	df = pd.read_csv('../driving_log.csv', header=0)
	df.columns = ["center_image", "left_image", "right_image", "steering_angle", "throttle", "break", "speed"]
	img_center = cv2.imread('../IMG/' + df["center_image"][0].strip().split('\\')[-1])

	height, width, channels = img_center.shape
	df_left, df_right, df_stright = filter_df(df, False)


	# Reading Udacity's data
	df = pd.read_csv('../data/driving_log.csv', header=0)
	df.columns = ["center_image", "left_image", "right_image", "steering_angle", "throttle", "break", "speed"]
	df_left2, df_right2, df_stright2 = filter_df(df, True)
	
	# Merge the two datasets
	df_left = df_left + df_left2
	df_right = df_right + df_right2
	df_stright = df_stright + df_stright2

	print("WE have", len(df_left), "left images")
	print("WE have", len(df_right), "right images")
	print("WE have", len(df_stright), "stright images")

	random.shuffle(df_stright)
	random.shuffle(df_left)
	random.shuffle(df_right)
	
	
	df_stright = pd.DataFrame(df_stright, columns=["center_image", "left_image", "right_image", "steering_angle", "Udacity"])
	df_left = pd.DataFrame(df_left, columns=["center_image", "left_image", "right_image", "steering_angle", "Udacity"])
	df_right = pd.DataFrame(df_right, columns=["center_image", "left_image", "right_image", "steering_angle", "Udacity"])


	# Put the data together
	data_list = [df_stright, df_left, df_right]
	df_master = pd.concat(data_list, ignore_index=True)

	X_data = df_master[["center_image","left_image","right_image","steering_angle", "Udacity"]]
	y_data = df_master["steering_angle"]
	
	return X_data, y_data

def getImg(source_path):
	filename = source_path.split('\\')[-1]
	current_path = '../IMG/' + filename
	image = cv2.imread(current_path)
	image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
	image = image[56:160,:,:]
	image = cv2.resize(image, (64,64))
	return image

def getUdacityImg(source_path):
	filename = source_path.split('\\')[-1]
	current_path = '../data/' + filename.strip()
	img = cv2.imread(current_path)
	img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
	img = img[56:160,:,:]
	img = cv2.resize(img, (64,64))
	return img
	
def random_flip_image(img, angle):
	# randomly flip the image with probability 1/2
	if np.random.randint(2) == 1:
		img = cv2.flip(img, 1)
		angle = -angle
	return img, angle

def flip_image(img, angle):
	img = cv2.flip(img, 1)
	angle = -angle
	return img, angle

def train_generator_df(df, batch_size=256):
	num_samples = len(df)
	ch, row, col = 3, 160, 320
	while 1: # Loop forever so the generator never terminates
		images = np.zeros((3*batch_size, row, col, ch))
		angles = np.zeros(3*batch_size)

		for i in range(batch_size):
			idx = np.random.randint(num_samples)
			data_row = df.iloc[[idx]].reset_index()
			if not data_row["Udacity"][0] :
				centerImg = getImg(data_row["center_image"][0])
				leftImg = getImg(data_row["left_image"][0])
				rightImg = getImg(data_row["right_image"][0])
			else:
				centerImg = getUdacityImg(data_row["center_image"][0])
				leftImg = getUdacityImg(data_row["left_image"][0])
				rightImg = getUdacityImg(data_row["right_image"][0])

			correction = 0.25
			center_angle = data_row["steering_angle"][0]
			left_angle = center_angle + correction
			right_angle = center_angle - correction


			centerImg, center_angle = random_flip_image(centerImg, center_angle)
			leftImg, left_angle = random_flip_image(leftImg, left_angle)
			rightImg, right_angle = random_flip_image(rightImg, right_angle)

			#images[i] = centerImg
			#angles[i] = center_angle			
			images[3*i] = centerImg
			images[3*i + 1] = leftImg
			images[3*i + 2] = rightImg

			angles[3*i] = center_angle
			angles[3*i + 1] = left_angle
			angles[3*i + 2] = right_angle

		yield images, angles

def train_generator_2_df(df, batch_size=64, ch=3, row=64, col=64):
	num_samples = len(df)
	
	while 1: # Loop forever so the generator never terminates
		images = np.zeros((6*batch_size, row, col, ch))
		angles = np.zeros(6*batch_size)

		for i in range(batch_size):
			idx = np.random.randint(num_samples)
			data_row = df.iloc[[idx]].reset_index()
			if not data_row["Udacity"][0] :
				centerImg = getImg(data_row["center_image"][0])
				leftImg = getImg(data_row["left_image"][0])
				rightImg = getImg(data_row["right_image"][0])
			else:
				centerImg = getUdacityImg(data_row["center_image"][0])
				leftImg = getUdacityImg(data_row["left_image"][0])
				rightImg = getUdacityImg(data_row["right_image"][0])

			correction = 0.3
			center_angle = data_row["steering_angle"][0]
			left_angle = center_angle + correction
			right_angle = center_angle - correction


			centerImgF, center_angleF = flip_image(centerImg, center_angle)
			leftImgF, left_angleF = flip_image(leftImg, left_angle)
			rightImgF, right_angleF = flip_image(rightImg, right_angle)

					
			images[6*i] = centerImg
			images[6*i + 1] = leftImg
			images[6*i + 2] = rightImg
			images[6*i + 3] = centerImgF
			images[6*i + 4] = leftImgF
			images[6*i + 5] = rightImgF

			angles[6*i] = center_angle
			angles[6*i + 1] = left_angle
			angles[6*i + 2] = right_angle
			angles[6*i + 3] = center_angleF
			angles[6*i + 4] = left_angleF
			angles[6*i + 5] = right_angleF

		yield images, angles

def valid_generator_df(df):
	while True:
		ch, row, col = 3, 160, 320

		for idx in range(len(df)):
			images = np.zeros((1, row, col, ch))
			data_row = df.iloc[[idx]].reset_index()
			if not data_row["Udacity"][0]:
				img, angle = getImg(data_row["center_image"][0]), data_row["steering_angle"][0]
			else:
				img, angle = getUdacityImg(data_row["center_image"][0]), data_row["steering_angle"][0]
			images[0] = img
			angle = np.array([[angle]])
			yield images, angle

def valid_generator_2_df(df):
	while True:
		ch, row, col = 3, 64, 64

		for idx in range(len(df)):
			images = np.zeros((1, row, col, ch))
			data_row = df.iloc[[idx]].reset_index()
			if not data_row["Udacity"][0]:
				img, angle = getImg(data_row["center_image"][0]), data_row["steering_angle"][0]
			else:
				img, angle = getUdacityImg(data_row["center_image"][0]), data_row["steering_angle"][0]
			images[0] = img
			angle = np.array([[angle]])
			yield images, angle


from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Flatten, Lambda, ELU
from keras.layers.convolutional import Convolution2D, Cropping2D
from keras.layers.pooling import MaxPooling2D


def get_commaai_model():
	#ch, row, col = 3, 160, 320
	ch, row, col = 3, 64, 64
	model = Sequential()
	model.add(Lambda(lambda x: x/127.5 - 1., input_shape=(row, col, ch), output_shape=(row, col, ch)))
	model.add(Convolution2D(16, 8, 8, subsample=(4, 4), border_mode="same"))
	model.add(ELU())
	model.add(Convolution2D(32, 5, 5, subsample=(2, 2), border_mode="same"))
	model.add(ELU())
	model.add(Convolution2D(64, 5, 5, subsample=(2, 2), border_mode="same"))
	model.add(Flatten())
	model.add(Dropout(.2))
	model.add(ELU())
	model.add(Dense(512))
	model.add(Dropout(.5))
	model.add(ELU())
	model.add(Dense(1))

	model.compile(optimizer="adam", loss="mse")

	return model


X_data, y_data = load_data()
print(len(X_data))
X_train_data, X_valid_data, y_train_data, y_valid_data = train_test_split(X_data, y_data, test_size=0.2)
X_train_data = X_train_data.reset_index(drop=True)
X_valid_data = X_valid_data.reset_index(drop=True)

train_generator = train_generator_2_df(X_train_data)
valid_generator = valid_generator_2_df(X_valid_data)

model = get_commaai_model()

batch_size = 64
numCameras = 6
spe = batch_size * ((numCameras*len(X_train_data)) // batch_size)

history = model.fit_generator(train_generator, samples_per_epoch=spe, \
			nb_epoch=8, validation_data=valid_generator, \
			nb_val_samples=len(X_valid_data))
    
model.save('model.h5')   
val_loss = history.history['val_loss'][8]

print("Validation score: ", val_loss)
