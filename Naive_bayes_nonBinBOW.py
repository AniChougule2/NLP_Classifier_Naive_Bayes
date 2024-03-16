import pandas as pd
import numpy as np
import spacy
import re
import emoji


twitter_data_clean=pd.read_csv("Tweet_data_for_gender_guessing/twitter_train_data_clean.csv")

eng_corpus=spacy.load("en_core_web_sm")

 
male={}
not_male={}
for id,row in twitter_data_clean.iterrows():
    if row['male']:
        for text in str(row['text']).split(" "):
            if text in male:
                male[text]=male[text]+1
            else:
                male[text]=1
    else:
        for text in str(row['text']).split(" "):
            if text in not_male:
                not_male[text]=not_male[text]+1
            else:
                not_male[text]=1


for text in not_male.keys():
    if text not in male:
        male[text]=0

for text in male.keys():
    if text not in not_male:
        not_male[text]=0

male_df = pd.DataFrame(list(male.items()), columns=['word', 'count'])
not_male_df = pd.DataFrame(list(not_male.items()), columns=['word', 'count'])

male_df.drop(index=0,inplace=True)
not_male_df.drop(index=0,inplace=True)

print(twitter_data_clean.shape[0])
print(male_df.shape[0],not_male_df.shape[0])
print(male_df.sort_values(by="count",ascending=False))
print(not_male_df.sort_values(by="count",ascending=False))

merge_df = pd.merge(male_df,not_male_df, on = "word",suffixes=("_male","_not_male"))
print(merge_df.head())
print(merge_df.shape[0])

total_male_count = male_df['count'].sum()
total_not_male_count = not_male_df['count'].sum()

merge_df['male_probability'] = (merge_df['count_male'] + 1) / (total_male_count + merge_df.shape[0])
merge_df['not_male_probability'] = (merge_df['count_not_male'] + 1) / (total_not_male_count + merge_df.shape[0])

df_train_probabilty = merge_df
print(df_train_probabilty.sort_values(by="male_probability",ascending=False))

male_prop = twitter_data_clean['male'].mean()
not_male_prop = 1 - male_prop

print(male_prop)
print(not_male_prop)


twitter_test_data_clean = pd.read_csv(
    "Tweet_data_for_gender_guessing/twitter_test_data_clean.csv")

pred_dict = {}

for id, row in twitter_test_data_clean.iterrows():

    pred_male = np.log(male_prop)
    pred_not_male = np.log(not_male_prop)
    for text in str(row['text']).split(" "):
        matching_rows = df_train_probabilty[df_train_probabilty['word'] == text]
        if not matching_rows.empty:
            pred_male += np.log(matching_rows['male_probability'].iloc[0])
            pred_not_male += np.log(matching_rows['not_male_probability'].iloc[0])

    pred_dict[id] = pred_male > pred_not_male

pred_df = pd.DataFrame(list(pred_dict.items()), columns=['id', 'Prediction'])

print(pred_df.head())
print(twitter_test_data_clean.head())

twitter_test_data_clean['Prediction'] = pred_df['Prediction'].values

print(twitter_test_data_clean.head(20))

TP = ((twitter_test_data_clean['Prediction'] == True) & (
    twitter_test_data_clean['male'] == True)).sum()
TN = ((twitter_test_data_clean['Prediction'] == False) & (
    twitter_test_data_clean['male'] == False)).sum()
FP = ((twitter_test_data_clean['Prediction'] == True) & (
    twitter_test_data_clean['male'] == False)).sum()
FN = ((twitter_test_data_clean['Prediction'] == False) & (
    twitter_test_data_clean['male'] == True)).sum()

print(f"True Positives (TP): {TP}")
print(f"True Negatives (TN): {TN}")
print(f"False Positives (FP): {FP}")
print(f"False Negatives (FN): {FN}")

sensitivity = TP / (TP + FN)
print(f"Sensitivity (Recall): {sensitivity}")

specificity = TN / (TN + FP)
print(f"Specificity: {specificity}")

precision = TP / (TP + FP)
print(f"Precision: {precision}")

negative_predictive_value = TN / (TN + FN)
print(f"Negative Predictive Value: {negative_predictive_value}")

accuracy = (TP + TN) / (TP + TN + FP + FN)
print(f"Accuracy: {accuracy}")

f_score = 2 * (precision * sensitivity) / (precision + sensitivity)
print(f"F-score: {f_score}")
