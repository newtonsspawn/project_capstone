import pandas as pd


pred_01 = pd.read_csv('../assets/predictions_01.csv', index_col=0)
pred_02 = pd.read_csv('../assets/predictions_02.csv', index_col=0)\
    .drop(['fname', 'set'], axis=1)
pred_03 = pd.read_csv('../assets/predictions_03.csv', index_col=0)\
    .drop(['fname', 'set'], axis=1)
pred_04 = pd.read_csv('../assets/predictions_04.csv', index_col=0)\
    .drop(['fname', 'set'], axis=1)
pred_05 = pd.read_csv('../assets/predictions_05.csv', index_col=0)\
    .drop(['fname', 'set'], axis=1)
pred_06 = pd.read_csv('../assets/predictions_06.csv', index_col=0)\
    .drop(['fname', 'set'], axis=1)
pred_07 = pd.read_csv('../assets/predictions_07.csv', index_col=0)\
    .drop(['fname', 'set'], axis=1)
pred_08 = pd.read_csv('../assets/predictions_08.csv', index_col=0)\
    .drop(['fname', 'set'], axis=1)


df = pred_01.join(pred_02).join(pred_03).join(pred_04).join(pred_05)\
    .join(pred_06).join(pred_07).join(pred_08)

print(df.head())

df.to_csv('../assets/combined_predictions.csv')