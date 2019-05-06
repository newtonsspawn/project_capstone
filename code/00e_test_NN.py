from imports.save_figures import save_fig

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Dropout

import matplotlib.pyplot as plt


curated_parsed_mfcc_df = pd.read_json('../assets/curated_parsed_mfcc_60.json')

df = curated_parsed_mfcc_df[curated_parsed_mfcc_df.duplicated(subset=['labels'],
                                                              keep=False)]

X = np.array(df['feature'].tolist())
y = np.array(df['labels'].tolist())

lb = LabelEncoder()
y = to_categorical(lb.fit_transform(y))

X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    stratify=y,
                                                    random_state=42)

ss = StandardScaler()
X_train_sc = ss.fit_transform(X_train)
X_test_sc = ss.transform(X_test)


num_labels = y.shape[1]


# Build model
model = Sequential()


model.add(Dense(256, activation='relu', input_shape=(60,)))

model.add(Dense(256, activation='relu',))
model.add(Dropout(0.5))

model.add(Dense(64, activation='relu',))
model.add(Dropout(0.5))

model.add(Dense(64, activation='relu',))
model.add(Dropout(0.5))

model.add(Dense(256, activation='relu',))

model.add(Dense(num_labels, activation='softmax'))


model.compile(loss='categorical_crossentropy',
              metrics=['accuracy'],
              optimizer='adam')


history = model.fit(X_train, y_train, 
                    batch_size=32,
                    epochs=250,
                    validation_data=(X_test, y_test))


model.summary()


print(f"Train accuracy: {history.history['acc']}")
print(f"Test accuracy: {history.history['val_acc']}")
print('\n')


plt.figure(1)
plt.plot(history.history['acc'], label='Train accuracy')
plt.plot(history.history['val_acc'], label='Test accuracy')
plt.title('NN Model Training & Testing Accuracy by Epoch')
plt.legend()
save_fig('curated_NN_60_acc_plot', folder='NN_metrics')

plt.figure(2)
plt.plot(history.history['loss'], label='Train loss')
plt.plot(history.history['val_loss'], label='Test loss')
plt.title('NN Model Training & Testing Loss by Epoch')
plt.legend()
save_fig('curated_NN_60_loss_plot', folder='NN_metrics')
