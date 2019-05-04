import os
import numpy as np
import pandas as pd
import librosa


def mfcc_parser(row, n_mfcc=40, sample_set='curated'):
    """
    This function will load a sound file and extract its features for parsing
    through the labelled CSVs.

    :param row: row with sound filename and associated label
    :param n_mfcc: integer with number of MFCCs to run (default: 40)
    :param sample_set: which sample set to find file (default: curated) 
    :return: [mfcc feature, label]
    """

    file = os.path.join(f'../data/train_{sample_set}', row.fname)

    # Handle exception to check if there isn't a file which is corrupted
    try:
        # Here kaiser_fast is a technique used for faster extraction
        X, sample_rate = librosa.load(file, res_type='kaiser_fast')

        # We extract mfcc feature from data
        mfccs = np.mean(librosa.feature.mfcc(y=X,
                                             sr=sample_rate,
                                             n_mfcc=n_mfcc).T, axis=0)

    except Exception as e:
        print("Error encountered while parsing file: ", file)
        return None, None

    feature = mfccs
    label = row['labels']

    return {'feature': feature.tolist(), 'labels': label}


curated_df = pd.read_csv('../data/train_curated.csv')
noisy_df = pd.read_csv('../data/train_noisy.csv')


curated_parsed_mfcc_df = pd.DataFrame(columns=['feature', 'labels'])

for row in range(len(curated_df)):
    curated_parsed_mfcc_df = curated_parsed_mfcc_df.append(mfcc_parser(curated_df.iloc[row],
                                                                       n_mfcc=60),
                                                           ignore_index=True)
    if (row % 100) == 0:
        print(f'...processed {row} files')

curated_parsed_mfcc_df.to_json('../assets/curated_parsed_mfcc_60.json')


noisy_parsed_mfcc_df = pd.DataFrame(columns=['feature', 'labels'])

for row in range(len(noisy_df)):
    noisy_parsed_mfcc_df = noisy_parsed_mfcc_df.append(mfcc_parser(noisy_df.iloc[row],
                                                                   n_mfcc=60,
                                                                   sample_set='noisy'),
                                                       ignore_index=True)
    if (row % 100) == 0:
        print(f'...processed {row} files')

noisy_parsed_mfcc_df.to_json('../assets/noisy_parsed_mfcc_60.json')