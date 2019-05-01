# Capstone Project

## Overview
For my capstone project, I will be developing a tool that will utilize deep
learning techniques to model a very diverse sample of labelled sounds, resulting 
in an ability to predict what type of sounds the tool hears in the future.

I am starting out with two labelled datasets downloaded directly from Kaggle.
The first is a curated set of 4,970 sound files with 213 unique labels. The
second is a noisy set of 19,815 sound files with 1,168 unique labels. I have a
graph of two wave files below as an example.

![](./images/000b6cfb_Motorcycle_(noisy).png)

![](./images/0019adae_Raindrop_(noisy).png)

The test set includes 1,120 unlabelled sounds files. Part of the way I plan to
test the accuracy of my model's predictions is by simply listening to the files
to determine if they are correct or at least in the ballpark.

## Process

Due to the size of the combined files (~23gb), I am doing all of my processing
via AWS. I am storing the files in an EFS, and processing my scripts via an
EC2 instance.

I am still researching which tools I am going to use to process the wave files,
but so far I am looking at the Librosa library for sound processing, and keras
for machine learning. The curated sounds will only have one output and should be
relatively easy to process, while the noisy sound files will have multiple
outputs, making modeling significantly more difficult.