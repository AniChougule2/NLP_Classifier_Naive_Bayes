import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px

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


twitter_data = pd.read_csv("Tweet_data_for_gender_guessing/merged_data.csv")

training_size = 80
num_training_samples = int(len(twitter_data) * 0.8)
twitter_data_train = twitter_data.iloc[:num_training_samples]
twitter_data_test = twitter_data.iloc[num_training_samples:]

print("Total data = ",len(twitter_data))
print("Total data males = ",len(twitter_data[twitter_data['male']== True]))
print("Total data not_males = ",len(twitter_data[twitter_data['male']== False]))
print("Train data = ", len(twitter_data_train))
print("Train data males = ", len(twitter_data_train[twitter_data_train['male'] == True]))
print("Train data not_males = ", len(twitter_data_train[twitter_data_train['male'] == False]))
print("Test data = ", len(twitter_data_test))
print("Total data males = ", len(twitter_data_test[twitter_data_test['male'] == True]))
print("Total data not_males = ", len(twitter_data_test[twitter_data_test['male'] == False]))

total_data = len(twitter_data)
total_data_males = len(twitter_data[twitter_data['male'] == True])
total_data_not_males = len(twitter_data[twitter_data['male'] == False])
train_data = len(twitter_data_train)
train_data_males = len(twitter_data_train[twitter_data_train['male'] == True])
train_data_not_males = len(
    twitter_data_train[twitter_data_train['male'] == False])
test_data = len(twitter_data_test)
test_data_males = len(twitter_data_test[twitter_data_test['male'] == True])
test_data_not_males = len(
    twitter_data_test[twitter_data_test['male'] == False])


# Creating a DataFrame to hold the data for plotting
data = {
    'Dataset': ['Total', 'Training', 'Test'],
    'Males': [total_data_males, train_data_males, test_data_males],
    'Not Males': [total_data_not_males, train_data_not_males, test_data_not_males]
}

df = pd.DataFrame(data)


df_melted = df.melt(id_vars=['Dataset'], var_name='Gender', value_name='Count')

fig = px.bar(df_melted, x='Dataset', y='Count', color='Gender', barmode='group',
             title="Label Distribution Across Datasets",
             labels={'Count': 'Number of Samples', 'Gender': 'Label'},
             text='Count')


fig.update_traces(texttemplate='%{text}', textposition='outside')


fig.update_layout(xaxis_title='Dataset',
                  yaxis_title='Number of Samples',
                  legend_title='Label',
                  uniformtext_minsize=8)
fig.show()
