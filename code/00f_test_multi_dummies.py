import pandas as pd


curated_df = pd.read_csv('../data/train_curated.csv')
noisy_df = pd.read_csv('../data/train_noisy.csv')

curated_df.insert(1, 'set', 'curated')
noisy_df.insert(1, 'set', 'noisy')

combined_df = curated_df.append(noisy_df, ignore_index=True)


df = pd.concat([combined_df['fname'],
                combined_df['set'],
                combined_df['labels'].str.split(',', expand=True)],
          axis=1)

print(df.head())
print('\n')

labels = df[0].dropna().unique().tolist() + \
         df[1].dropna().unique().tolist() + \
         df[2].dropna().unique().tolist() + \
         df[3].dropna().unique().tolist() + \
         df[4].dropna().unique().tolist() + \
         df[5].dropna().unique().tolist() + \
         df[6].dropna().unique().tolist()

labels = sorted(list(set(labels)))
print(labels)

dummy_cols = pd.DataFrame(columns=labels)

df = pd.concat([df, dummy_cols], ignore_index=True, sort=False)
df.fillna(0, inplace=True)

for col in dummy_cols:
    for i in range(7):
        df[col][df[i] == col] = 1

df.drop([0, 1, 2, 3, 4, 5, 6], axis=1, inplace=True)

print(df.head())

df.to_csv('../assets/combined_dummies.csv')

