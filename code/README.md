# Code Documentation

Since I completed all of my coding in pure python files instead of Jupyter
notebooks, I will comment on each script and what it is doing in this README.

## 01_EDA.py

This module conducts some simple EDA functions. It loads the labelled CSVs
to determine how many files each set contains and how many/what percentage of
each label exists.

I also load the *wave* and *librosa* packages to study the parameters and
information available to me from each package.

## 02_plot_wave_file.py

This module continues the EDA by plotting what the wave signals look like in
their raw form.

It uses the *wave* package to:
1. extract raw audio from WAV file,
2. split the data into channels, and
3. get time component from indices.

I then added the label from the corresponding CSV and used *matplotlib* to
plot the sound signal.

## 03_MFCC_sound_parser.py

Here I build a function that uses the  *librosa* library to parse each sound
file into its MFCC transformation. It takes as an argument how many many MFCCs
you want to create per file, with a default of 40, and runs through the entire
data set to create and save a JSON file with all of the relative MFCCs. 

## 04_NN_single-label.py

This module performs the first modeling on the transformed set. It loads the 
JSON files with the MFCC transformations, loads the unsplit labels, encodes
those labels into categorical features, gets rid of labels without duplicates
(so the list can be stratified), performs a train-test-split with 
stratification, and then scales the data using the Standard Scalar.

Finally, it runs a Sequential Neural Network (SNN) model using 'softmax' as its
output layer activation function, and 'categorical_crossentropy' for its loss
function. It runs 250 epochs and then prints out the summary and train/test
accuracies, and plots the train/test accuracies and losses by epoch.

Those plots are then saved, using a save_fig module I created in an imported
file.

## 05_multi_dummy_maker.py

This module separates the labels into a sparse matrix. It splits the labels 
on commas and drills down to get all 80 unique labels. It then runs a for loop
through the original labels, and then through each column, replacing the 0 with 
a 1 if one of its labels matches that column.

Finally, it drops the original columns and saves a CSV with the multi-labelled 
dummies as a sparse matrix.

## 06_NN_multi-label.py

This module is my first attempt at running an SNN model with the sparse
multi-labelled matrix. I changed the output layer activation to sigmoid since
I am now requiring binary outputs for each label.

I print out the first two rows of predictions to see what they look like and 
again plot and save the train/test accuracies and losses across epochs.

## 07_NN_multi-label_custom.py

since the last model run was a failure, I resorted to building a custom function
that runs the SNN for 10 epochs across each label separately, using 
binary_crossentropy for the loss function and sigmoid for the output layer's
activation.

I had to run this model in batches of 10 because running the full model broke
my kernel.

I saved the predictions for each label to a dataframe and printed out the 
precision score for each label. The final prediction matrices are saved to a CSV
file on the EFS.

## 08_multi-label_predict_combine.py

This file simply joins all 8 of the prediction matrices into one large sparse
dataframe and saves that back out to a CSV.

## 09_NN_multi-label_metrics.py

I now load the true CSV and the predicted CSV and load them into dataframes. I 
loop through each label and use sklearn's precision_score to measure the 
precision of each label's predictions. I save those to a list which then allows
me to plot the top and worst 10 performing labels in the dataset.

I also determine the average precision score using the relative weights of each
label in the data set.