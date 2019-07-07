import numpy as np
import pandas as pd
import cv2
import os
from keras import layers
from keras.layers import Input, Dense, Activation, Flatten, Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Dense, Dropout
from keras.layers.normalization import BatchNormalization
from keras.utils import np_utils, print_summary
from keras.models import Sequential
from keras.callbacks import ModelCheckpoint
import keras.backend as K
from keras import regularizers
from keras import optimizers

training_images = 11000
testing_images = 2200

train_datadir = "data/train"
test_datadir = "data/test"
categories = {1:"1", 2:"2", 3:"3", 4:"4", 5:"5", 6:"6", 7:"7", 8:"8", 9:"9", 10:"10", 11:"11"}
train_dataset = np.zeros(shape=(training_images,2501))
test_dataset = np.zeros(shape=(testing_images,2501))

counter = 0
for category in categories:
    path = os.path.join(train_datadir,categories[category])
    label = np.array([category])
    for img in os.listdir(path):
        try:
            image = cv2.imread(os.path.join(path,img), cv2.IMREAD_GRAYSCALE)
            image = cv2.resize(image, (50,50))
            te = image.flatten()
            te = np.array(te)
            te = np.concatenate((te, label), axis=0)
            te = np.array(te)[np.newaxis]
            #cv2.imwrite("newfan"+str(i)+".jpg",image)
            train_dataset[counter] = te
            counter = counter + 1
        except Exception as e:
            pass

counter = 0
for category in categories:
    path = os.path.join(test_datadir,categories[category])
    label = np.array([category])
    for img in os.listdir(path):
        try:
            image = cv2.imread(os.path.join(path,img), cv2.IMREAD_GRAYSCALE)
            image = cv2.resize(image, (50,50))
            te = image.flatten()
            te = np.array(te)
            te = np.concatenate((te, label), axis=0)
            te = np.array(te)[np.newaxis]
            #cv2.imwrite("newfan"+str(i)+".jpg",image)
            test_dataset[counter] = te
            counter = counter + 1
        except Exception as e:
            pass

print("training dataset shape",train_dataset.shape)
print("testing dataset shape",test_dataset.shape)

np.random.shuffle(train_dataset)
np.random.shuffle(test_dataset)

X_train = train_dataset[:, 0:2500]
Y_train = train_dataset[:, 2500]

x_test = test_dataset[:, 0:2500]
Y_test = test_dataset[:, 2500]

x_train = X_train/255

Y_train = Y_train.reshape(Y_train.shape[0], 1)
y_train = Y_train.T

Y_test = Y_test.reshape(Y_test.shape[0], 1)
y_test = Y_test.T

#we are taking an image of 50x50 for training purpose
image_x = 50
image_y = 50
#to_categorical is an function which converts a matrix to a vector; in our case it basically gets the label which is an integer
#and changes it to form of one_hot_encoding.
#Check this link for more details->https://keras.io/utils/
train_y = np_utils.to_categorical(y_train, dtype='int32')
test_y = np_utils.to_categorical(y_test, dtype='int32')
#adjusting the shape
train_y = train_y.reshape(train_y.shape[1], train_y.shape[2])
test_y = test_y.reshape(test_y.shape[1], test_y.shape[2])
x_train = x_train.reshape(x_train.shape[0], image_x, image_y, 1)
x_test = x_test.reshape(x_test.shape[0], image_x, image_y, 1)

adam = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
def keras_model(image_x, image_y):
    num_of_classes = 12
    model = Sequential()
    model.add(Conv2D(filters=8, kernel_size=(3,3), input_shape=(image_x, image_y, 1), activation='relu'))
    model.add(MaxPooling2D(pool_size=(3,3), strides=(3,3), padding='same'))
    model.add(Conv2D(64, (5,5), activation='relu'))
    model.add(MaxPooling2D(pool_size=(3,3), strides=(3,3), padding='same'))
    model.add(Dropout(0.5))
    model.add(BatchNormalization())
    model.add(Flatten())
    model.add(Dropout(0.5))
    model.add(Dense(512, activation='relu', kernel_regularizer=regularizers.l2(0.007)))
    model.add(BatchNormalization())
    model.add(Dense(256, activation='relu', kernel_regularizer=regularizers.l2(0.002)))
    model.add(BatchNormalization())
    model.add(Dense(num_of_classes, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
    #saving the trained data is saved here
    filepath='hand_emoji_v1.h5'
    checkpoint1 = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
    callbacks_list = [checkpoint1]
    #returning the model and the checkpoints of the saved(learned) data
    return model, callbacks_list

model, callbacks_list = keras_model(image_x, image_y)
model.fit(x_train, train_y, validation_data=(x_test, test_y), epochs=10, batch_size=64, callbacks=callbacks_list)
scores = model.evaluate(x_test, test_y, verbose=0)
print("CNN error: %.2f%%" %(100 - scores[1] * 100))
print_summary(model)
model.save('hand_emoji_v1.h5')
