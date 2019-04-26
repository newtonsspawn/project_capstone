import wave
import matplotlib.pyplot as plt
import pandas as pd
import librosa

curated_df = pd.read_csv('../data/train_curated.csv')
print(curated_df.head())
print(f'Shape: {curated_df.shape}')
print('\n')

print(f"{len(curated_df['labels'].unique())} unique labels")
print('\n')


noisy_df = pd.read_csv('../data/train_noisy.csv')
print(noisy_df.head())
print(f'Shape: {noisy_df.shape}')
print('\n')

print(f"{len(noisy_df['labels'].unique())} unique labels")
print('\n')

f = wave.open('../data/train_curated/0006ae4e.wav')
print(f.getparams())
f.close()

plt.plot([1, 2, 3, 4, 5], [6, 7, 8, 9, 10])

data, sampling_rate = librosa.load('../data/train_curated/0006ae4e.wav')

plt.figure(figsize=(12, 4))
librosa.display.waveplot(data, sr=sampling_rate)
