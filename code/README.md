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



## 04_NN_single-label.py



## 05_multi_dummy_maker.py



## 06_NN_multi-label.py



## 07_NN_multi-label_custom.py



## 08_multi-label_predict_combine.py



## 09_NN_multi-label_metrics.py

