import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dense, Dropout
from sklearn.metrics import precision_score


curated_parsed_mfcc_df = pd.read_json('../assets/curated_parsed_mfcc_40.json')
noisy_parsed_mfcc_df = pd.read_json('../assets/noisy_parsed_mfcc_40.json')

combined_parsed_mfcc_df = curated_parsed_mfcc_df.append(noisy_parsed_mfcc_df,
                                                       ignore_index=True)

combined_df = pd.read_csv('../assets/combined_dummies.csv')

df = combined_parsed_mfcc_df.merge(combined_df,
                                   left_index=True,
                                   right_index=True)

predictions_df = combined_df[['fname', 'set']]

X = np.array(df['feature'].tolist())

ss = StandardScaler()
X_sc = ss.fit_transform(X)


labels = list(combined_df.columns)[3:]

for label in labels[70:80]:
    
    y = np.array(df[label])
    
    num_labels = 1
    
    
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
    
    
    model.compile(loss='binary_crossentropy',
                  metrics=['accuracy'],
                  optimizer='adam')
    
    
    history = model.fit(X_sc, y,
                        batch_size=32,
                        epochs=10,
                        # validation_data=(X_test_sc, y_test)
                        )
    
    
    # model.summary()
    
    print(f"Train accuracy: {history.history['acc']}")
    
    
    pred = model.predict_proba(X_sc)
    pred_flat = [item for sublist in pred for item in sublist]
    
    selected = sorted(pred_flat, reverse=True)
    val = selected[sum(y)]
    
    preds = [1 if x > val else 0 for x in pred_flat]
    
    # print(len(preds), len(y), sum(preds), sum(y))
    
    print(f'Precision Score <{label}>: {precision_score(y, preds)}')
    
    predictions_df[label] = preds

predictions_df.to_csv('../assets/predictions_08.csv')

