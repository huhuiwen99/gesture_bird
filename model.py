#CNN + openCV： gesture recognition
import tensorflow as tf
import keras
import numpy as np
import pandas as pd
import cv2
from matplotlib import pyplot as plt
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
from keras.datasets import mnist
from keras.utils import np_utils
from keras.optimizers import SGD
from sklearn.model_selection import train_test_split

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
y_train = train['label'].values
y_test = test['label'].values
x_train = train.drop(['label'],axis=1)
x_test = test.drop(['label'],axis=1)

#np.display(x_train.info())
#np.display(x_train.head(n=2))

# pre-processing
x_train = np.array(x_train.iloc[:,:])
x_train = np.array([np.reshape(i,(28,28)) for i in x_train])
x_test = np.array(x_test.iloc[:,:])
x_test = np.array([np.reshape(i,(28,28)) for i in x_test])
num_classes = 26
y_train = np.array(y_train).reshape(-1)
y_test = np.array(y_test).reshape(-1)
y_train = np.eye(num_classes)[y_train]
y_test = np.eye(num_classes)[y_test]
x_train = x_train.reshape((27455,28,28,1))
x_test = x_test.reshape((7172,28,28,1))
x_train, x_val, y_test, y_val = train_test_split(x_train, y_train, test_size = 0.1)

classifier = Sequential()
classifier.add(Conv2D(filters=8, kernel_size=(3,3),strides=(1,1), padding="same",input_shape=(28,28,1),activation='relu',data_format='channels_last'))
classifier.add(MaxPooling2D(pool_size=(2,2)))
classifier.add(Conv2D(filters=16, kernel_size=(3,3),strides=(1,1), padding="same",activation='relu'))
classifier.add(Dropout(0,5))
classifier.add(MaxPooling2D(pool_size=(4,4)))
classifier.add(Dense(128,activation='relu'))
classifier.add(Flatten())
classifier.add(Dense(26,activation='softmax'))

classifier.compile(optimizer='SGD',loss='categorical_crossentropy',metrics=['accuracy'])
history  = classifier.fit(x_train, y_train, epochs=30, batch_size=128, validation_data = (x_val, y_val), verbose=2)

res = classifier.evaluate(x=x_test, y=y_test, batch_size=32)
print("Accuracy",res[1])
classifier.save('CNNmodel.h5')
