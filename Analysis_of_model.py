import matplotlib.pyplot as plt
import numpy as np


data = {
    '80% Training': {'TP': 3133, 'TN': 3324, 'FP': 2243, 'FN': 2402},
    '50% Training': {'TP': 3057, 'TN': 3300, 'FP': 2267, 'FN': 2478},
    '30% Training': {'TP': 2945, 'TN': 3440, 'FP': 2127, 'FN': 2590},
}


fig, axes = plt.subplots(1, 3, figsize=(18, 6))
titles = ['80% Training', '50% Training', '30% Training']
for ax, title in zip(axes, titles):
    TP, TN, FP, FN = data[title].values()
    matrix = np.array([[TP, FN], [FP, TN]])
    im = ax.imshow(matrix, cmap=plt.cm.Blues)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, matrix[i, j], ha="center",va="center", color="black")
    ax.set_title(title)
    ax.set_xlabel('Predicted labels')
    ax.set_ylabel('True labels')
    ax.set_xticks(np.arange(len(['Positive', 'Negative'])))
    ax.set_yticks(np.arange(len(['Positive', 'Negative'])))
    ax.set_xticklabels(['Positive', 'Negative'])
    ax.set_yticklabels(['Positive', 'Negative'])
plt.tight_layout()
fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.5)


for key in data:
    data[key]['TPR'] = data[key]['TP'] / (data[key]['TP'] + data[key]['FN'])
    data[key]['FPR'] = data[key]['FP'] / (data[key]['FP'] + data[key]['TN'])


plt.figure(figsize=(7, 7))
colors = {'80% Training': 'blue',
          '50% Training': 'red', '30% Training': 'green'}
for key, values in data.items():
    plt.plot(values['FPR'], values['TPR'], 'o', color=colors[key], label=key)
plt.plot([0, 1], [0, 1], 'r--', label='Random Classifier')
plt.title('ROC Curve for Different Training Sizes')
plt.xlabel('False Positive Rate (FPR)')
plt.ylabel('True Positive Rate (TPR)')
plt.legend()
plt.grid(True)
plt.show()
