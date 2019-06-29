# Hand-Emoji-Recognition

## Introduction
This project is on Hand emojis recognition, where there are 11 types of different hand emojis consisting of 1000 images per class of 50x50 resolution. Total of 11000 training images and 2200 of testing images are fed to the network.

## The Data
Data was created by shooting a video of hand doing different emojis, video was 30 fps which gave 30 frames per second so a footage of 40 seconds would give me 1200 images, of which i used 1000 for training and 200 for testing, this i did for every 11 classes. Data folder can be found at https://drive.google.com/file/d/1g_zwgvrWSCXvgwxFXz-4Qc7NSt1GRF0E/view?usp=sharing

Sample input images after processing are:

![105](https://user-images.githubusercontent.com/35074988/60382475-20026f80-9a81-11e9-88f4-bc6aae16dbff.jpg)
![103](https://user-images.githubusercontent.com/35074988/60382496-5e982a00-9a81-11e9-9761-692059cf68a6.jpg)
![101](https://user-images.githubusercontent.com/35074988/60382510-956e4000-9a81-11e9-8c5e-185ab391aa0d.jpg)

After training the images i did manual testing of the model for that i just click an image through my phone and just passed it to the processing function in *classify_result.ipnb*.

The sample image i used for manual testing is:

![IMG_3](https://user-images.githubusercontent.com/35074988/60382537-e2521680-9a81-11e9-95a2-4c654d7bd705.jpg)

## Process
Firstly use *create database.ipnb* to create images from video while creating the database split training data and testing data.
Then use *emoji_train.ipnb* to train the model on the images which creates *hand_emoji_v5.h5* file which is just a file where the weights of the network are saved which can be used to load the model. Now, to manually test the images use *classify_result.ipnb* it would show to images with its predicted label.

## Problems faced
In the following project some of the problems i encountered were:

**1:** I tried to re-use the code from my last project which was Devanagari Handwriting recognition(https://github.com/ajeenkkya/Devanagari-handwriting-recognition) but the images was 32x32 which used 1024 columns in spreadsheet but in this case the image had to be minimum 50x50 for the data to be relevant which needed 2500 columns which was not possible. So I had to redo the data storing part and i just got rid of storing the images in a csv file instead i directly saved the data in a numpy array.

**2:** The model was overfitting and my best guess is because of less data. I used a batch notmalization, l2 regularizarion and dropout layers to overcome overfitting.

**3:** A lot of trial and error had to be done to perfect the model.
