from imports.save_figures import save_fig

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import wave

curated_df = pd.read_csv('../data/train_curated.csv')


def plot_signal_wave(sample):
    """

    :param sample: sample sound file to process
    :return: cignal wave plot of file
    """

    file = os.path.join('../data/train_curated', sample)

    with wave.open(file, 'r') as wav_file:
        # Extract Raw Audio from Wav File
        signal = wav_file.readframes(-1)
        signal = np.fromstring(signal, 'Int16')

        # Split the data into channels
        channels = [[] for channel in range(wav_file.getnchannels())]
        for index, datum in enumerate(signal):
            channels[index%len(channels)].append(datum)

        # Get time from indices
        fs = wav_file.getframerate()
        time = np.linspace(0, len(signal)/len(channels)/fs,
                           num=len(signal)/len(channels))

        # Get label from dataframe
        name = curated_df[curated_df['fname'] == sample]['labels'].values[0]

        # Plot
        plt.figure(1)
        plt.title(f'Signal Wave: {name}')
        for channel in channels:
            plt.plot(time, channel)
        # save_fig(name)

    return plt.show()


plot_signal_wave('0019ef41.wav')
