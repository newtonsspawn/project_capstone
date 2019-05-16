import pandas as pd
import librosa
import wave

curated_df = pd.read_csv('../data/train_curated.csv')
print(curated_df.head())
print(f'Shape: {curated_df.shape}')
print('\n')

print(f"{len(curated_df['labels'].unique())} unique labels")
print('\n')

print(f"{curated_df['labels'].value_counts(normalize=False)}")
print('\n')


noisy_df = pd.read_csv('../data/train_noisy.csv')
print(noisy_df.head())
print(f'Shape: {noisy_df.shape}')
print('\n')

print(f"{len(noisy_df['labels'].unique())} unique labels")
print('\n')

print(f"{noisy_df['labels'].value_counts(normalize=False)}")
print('\n')

f = wave.open('../data/train_curated/0019ef41.wav')
print(f.getparams())
f.close()

print('\n')

data, sampling_rate = librosa.load('../data/train_curated/0019ef41.wav')
print(len(data))
