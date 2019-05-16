from imports.save_figures import save_fig

import numpy as np
import pandas as pd
import operator
from sklearn.metrics import precision_score

import matplotlib.pyplot as plt


combined_dummies = pd.read_csv('../assets/combined_dummies.csv',
                               index_col=0)
combined_predictions = pd.read_csv('../assets/combined_predictions.csv',
                                   index_col=0)

labels = combined_dummies.columns[2:]

precision_dict = {}
precision_weights = []

for label in labels:
    y_true = combined_dummies[label]
    y_pred = combined_predictions[label]

    # print(f'Precision Score <{label}>: {precision_score(y_true, y_pred)}')

    precision_dict[label] = precision_score(y_true, y_pred)

    precision_weights.append(np.sum(y_true))


precisions = sorted(precision_dict.items(),
                    key=operator.itemgetter(1),
                    reverse=True)

average_precision = np.sum(
    np.multiply(
        [item for item in precision_dict.values()],
        precision_weights)) / np.sum(precision_weights)
print(average_precision)



print(precision_dict['Fart'])

best_X = [item[1] for item in precisions[:10]][::-1]
best_Y = [item[0] for item in precisions[:10]][::-1]

plt.figure(1, figsize=(8, 5))
plt.barh(best_Y, best_X)
plt.title('Top 10 Performing Sounds by Precision Metric')
# save_fig('best_performers_plot', tight_layout=True)


worst_X = [item[1] for item in precisions[-10:]]
worst_Y = [item[0] for item in precisions[-10:]]

plt.figure(2, figsize=(8, 5))
plt.barh(worst_Y, worst_X)
plt.title('Worst 10 Performing Sounds by Precision Metric')
# save_fig('worst_performers_plot', tight_layout=True)