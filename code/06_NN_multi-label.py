from imports.save_figures import save_fig

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras import metrics
import tensorflow as tf

import matplotlib.pyplot as plt


curated_parsed_mfcc_df = pd.read_json('../assets/curated_parsed_mfcc_40.json')
noisy_parsed_mfcc_df = pd.read_json('../assets/noisy_parsed_mfcc_40.json')

combined_parsed_mfcc_df = curated_parsed_mfcc_df.append(noisy_parsed_mfcc_df,
                                                       ignore_index=True)

combined_df = pd.read_csv('../assets/combined_dummies.csv')
labels = list(combined_df.columns)
labels = labels[3:]

df = combined_parsed_mfcc_df.merge(combined_df, left_index=True, right_index=True)
# df = df[df.duplicated(subset=labels, keep=False)]

X = np.array(df['feature'].tolist())
y = np.array(df[labels])

X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    # stratify=y,
                                                    random_state=42)

ss = StandardScaler()
X_train_sc = ss.fit_transform(X_train)
X_test_sc = ss.transform(X_test)


num_labels = y.shape[1]


# Build model
model = Sequential()


model.add(Dense(256, activation='relu', input_shape=(40,)))

model.add(Dense(256, activation='relu',))
model.add(Dropout(0.5))

model.add(Dense(64, activation='relu',))
model.add(Dropout(0.5))

model.add(Dense(64, activation='relu',))
model.add(Dropout(0.5))

model.add(Dense(256, activation='relu',))

model.add(Dense(num_labels, activation='sigmoid'))


model.compile(loss='sigmoid',
              metrics=['accuracy'],
              optimizer='adam')


history = model.fit(X_train, y_train,
                    batch_size=32,
                    epochs=100,
                    validation_data=(X_test, y_test))

model.summary()

print(f"Train accuracy: {history.history['acc']}")
print(f"Test accuracy: {history.history['val_acc']}")
print('\n')

pred = model.predict_proba(X_test_sc)
print(pred[:2])
print('\n')

print(history.history)

plt.figure(1)
plt.plot(history.history['binary_crossentropy'], label='Train accuracy')
plt.plot(history.history['val_binary_crossentropy'], label='Test accuracy')
plt.title('NN Model Training & Testing Accuracy by Epoch')
plt.legend()
save_fig('NN_multi_40_acc_plot', folder='NN_metrics')

plt.figure(2)
plt.plot(history.history['loss'], label='Train loss')
plt.plot(history.history['val_loss'], label='Test loss')
plt.title('NN Model Training & Testing Loss by Epoch')
plt.legend()
save_fig('NN_multi_40_loss_plot', folder='NN_metrics')
