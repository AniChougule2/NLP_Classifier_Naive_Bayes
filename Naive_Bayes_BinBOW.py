import pandas as pd
import numpy as np
import spacy
import re
import emoji

twitter_data_clean = pd.read_csv("Tweet_data_for_gender_guessing/twitter_train_data_clean.csv")

eng_corpus = spacy.load("en_core_web_sm")


male = {}
not_male = {}
for id, row in twitter_data_clean.iterrows():
    if row['male']:
        for text in str(row['text']).split(" "):
            if text  not in male:
                male[text] = 2
    else:
        for text in str(row['text']).split(" "):
            if text not in not_male:
                not_male[text] = 2


for text in not_male.keys():
    if text not in male:
        male[text] = 1

for text in male.keys():
    if text not in not_male:
        not_male[text] = 1

male_df = pd.DataFrame(list(male.items()), columns=['word', 'present'])
not_male_df = pd.DataFrame(list(not_male.items()), columns=['word', 'present'])

male_df.drop(index=0, inplace=True)
not_male_df.drop(index=0, inplace=True)

print(twitter_data_clean.shape[0])
print(male_df.shape[0], not_male_df.shape[0])
print(male_df.head())
print(not_male_df.head())

merge_df = pd.merge(male_df, not_male_df, on="word",
                    suffixes=("_male", "_not_male"))
print(merge_df.head())
print(merge_df.shape[0])

total_male_count = male_df['present'].sum()
total_not_male_count = not_male_df['present'].sum()

merge_df['male_probability'] = merge_df['present_male'] / total_male_count
merge_df['not_male_probability'] = merge_df['present_not_male'] / total_not_male_count

df_train_probabilty = merge_df
print(df_train_probabilty.sort_values(by="male_probability", ascending=False))
