# Hand-Emoji-Recognition

## Introduction
This project is on Hand emojis recognition, where there are 11 types of different hand emojis consisting of 1000 images per class of 50x50 resolution. Total of 11000 training images and 2200 of testing images are fed to the network.

## The Data
Data was created by shooting a video of hand doing different emojis, video was 30 fps which gave 30 frames per second so a footage of 40 seconds would give me 1200 images, of which i used 1000 for training and 200 for testing, this i did for every 11 classes.

Sample input image after processing is:

After training the images i did manual testing of the model for that i just click an image through my phone and just passed it to the processing function in *classify_result.ipnb*.

The sample image i used for manual testing is:

## Problems faced
In the following project some of the problems i encountered were:

**1:** I tried to re-use the code from my last project which was Devanagari Handwriting recognition(https://github.com/ajeenkkya/Devanagari-handwriting-recognition) but the images was 32x32 which used 1024 columns in spreadsheet but in this case the image had to be minimum 50x50 for the data to be relevant which needed 2500 columns which was not possible. So I had to redo the data storing part and i just got rid of storing the images in a csv file instead i directly saved the data in a numpy array.

**2:** The model was overfitting and my best guess is because of less data. I used a batch notmalization, l2 regularizarion and dropout layers to overcome overfitting.

**3:** A lot of trial and error had to be done to perfect the model.
