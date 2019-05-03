import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from keras.utils.np_utils import to_categorical




X = np.array(curated_parsed_mfcc_df.feature.tolist())
y = np.array(curated_parsed_mfcc_df.labels.tolist())

lb = LabelEncoder()
y = to_categorical(lb.fit_transform(y))

print(y)