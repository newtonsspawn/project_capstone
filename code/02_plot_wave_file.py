from imports.save_figures import save_fig

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import wave

curated_df = pd.read_csv('../data/train_curated.csv')
noisy_df = pd.read_csv('../data/train_noisy.csv')
combined_df = curated_df.append(noisy_df, ignore_index=True)


def plot_signal_wave(sample, sample_set='curated'):
    """
    This function will take a sound file from either the curated or noisy set
    and display its waveform graphically. 
    
    :param sample: name of the file
    :param sample_set: which sample set to find file (default: curated) 
    :return: signal wave plot of file
    """

    file = os.path.join(f'../data/train_{sample_set}', sample)
    file_name = os.path.splitext(sample)[0]

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
        label = combined_df[combined_df['fname'] == sample]['labels'].values[0]

        # Plot
        plt.figure(1)
        plt.title(f'Signal Wave: {label} | (file: {sample})')
        for channel in channels:
            plt.plot(time, channel)
        save_fig(f'{file_name}_{label}_({sample_set})', folder=wave_plots)

    return plt.show()


plot_signal_wave('0019adae.wav', 'noisy')

# for file in curated_df[curated_df['labels'] == 'Fart']['fname'].head(10).values:
#     plot_signal_wave(file)
