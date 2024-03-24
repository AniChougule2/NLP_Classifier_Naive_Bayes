import matplotlib.pyplot as plt

data = {
    '80% Training': {'TP': 3133, 'TN': 3324, 'FP': 2243, 'FN': 2402},
    '50% Training': {'TP': 3057, 'TN': 3300, 'FP': 2267, 'FN': 2478},
    '30% Training': {'TP': 2945, 'TN': 3440, 'FP': 2127, 'FN': 2590},
}


for key in data:
    data[key]['TPR'] = data[key]['TP'] / (data[key]['TP'] + data[key]['FN'])
    data[key]['FPR'] = data[key]['FP'] / (data[key]['FP'] + data[key]['TN'])


plt.figure(figsize=(7, 7))


colors = {'80% Training': 'b', '50% Training': 'r', '30% Training': 'g'}


for key, values in data.items():
    plt.plot(values['FPR'], values['TPR'], 'o', color=colors[key], label=key)

plt.plot([0, 1], [0, 1], 'r--', label='Random Classifier')

# Final touches
plt.title('ROC Curve for Naive Bayes Classifier')
plt.xlabel('False Positive Rate (FPR)')
plt.ylabel('True Positive Rate (TPR)')
plt.legend()
plt.grid(True)
plt.show()
