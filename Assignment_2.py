import pandas as pd
import numpy as np
import spacy
import re
import emoji

##read the train and valid datasets
twitter_data_clean=pd.read_csv("Tweet_data_for_gender_guessing/twitter_train_data_clean.csv")

# print(twitter_data.head())
## loaded english corpus from spacy
eng_corpus=spacy.load("en_core_web_sm")

## cleaning the data using lemmation and converting the emojis into text
# twitter_data_clean=pd.DataFrame()
# for index,row in twitter_data.iterrows():
#     sentence=eng_corpus(row['text'])
#     lemmas=[]
#     for token in sentence:
#         demojized_token = emoji.demojize(token.text)
#         if demojized_token != token.text:
#             lemmas.append(demojized_token)
#         else:
#             lemmas.append(token.lemma_)
#     lemmas = [lem.replace(':', '').replace('_', ' ') for lem in lemmas]
#     clean_lemmas = re.sub(r'[\n\t]', ' ', " ".join(lemmas))
#     clean_lemmas = re.sub(r'[ ]+', ' ', clean_lemmas)

#     new_row = pd.Series({
#         'id': row['id'],
#         'time': row['time'],
#         'text': clean_lemmas,
#         'male': row['male']
#     })
#     twitter_data_clean=twitter_data_clean.append(new_row, ignore_index=True)
 
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
        male[text]=1

for text in male.keys():
    if text not in not_male:
        not_male[text]=1

male_df = pd.DataFrame(list(male.items()), columns=['word', 'count'])
not_male_df = pd.DataFrame(list(not_male.items()), columns=['word', 'count'])

male_df.drop(index=0,inplace=True)
not_male_df.drop(index=0,inplace=True)

print(twitter_data_clean.shape[0])
print(male_df.shape[0],not_male_df.shape[0])
print(male_df.sort_values(by="count",ascending=False))
print(not_male_df.sort_values(by="count",ascending=False))

