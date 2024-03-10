import pandas as pd
import numpy as np
import spacy
import re
import emoji

##read the train and valid datasets
twitter_data=pd.read_csv("/Users/anichougule/Masters/Code/Python/Sem_2/NLP/Assignment_2/NLP_Classifier_Naive_Bayes/Tweet_data_for_gender_guessing/twitgen_train_201906011956.csv")
twitter_data=twitter_data._append(pd.read_csv("/Users/anichougule/Masters/Code/Python/Sem_2/NLP/Assignment_2/NLP_Classifier_Naive_Bayes/Tweet_data_for_gender_guessing/twitgen_valid_201906011956.csv"))

# print(twitter_data.head())
## loaded english corpus from spacy
eng_corpus=spacy.load("en_core_web_sm")

## cleaning the data using lemmation and converting the emojis into text
twitter_data_clean=pd.DataFrame()
for index,row in twitter_data.iterrows():
    sentence=eng_corpus(row['text'])
    lemmas=[]
    for token in sentence:
        demojized_token = emoji.demojize(token.text)
        if demojized_token != token.text:
            lemmas.append(demojized_token)
        else:
            lemmas.append(token.lemma_)
    lemmas = [lem.replace(':', '').replace('_', ' ') for lem in lemmas]
## removing unwanted tabs and new line charcter along with multipal spaces
    clean_lemmas = re.sub(r'[\n\t]', ' ', " ".join(lemmas))
    clean_lemmas = re.sub(r'[ ]+', ' ', clean_lemmas)

    new_row = pd.Series({
        'id': row['id'],
        'time': row['time'],
        'text': clean_lemmas,
        'male': row['male']
    })
    twitter_data_clean=twitter_data_clean._append(new_row, ignore_index=True)

## creating dicts to count the occurance of words 
male={}
not_male={}
for id,row in twitter_data_clean.iterrows():
    if row['male']:
        for text in row['text'].split(" "):
            if text in male:
                male[text]=male[text]+1
            else:
                male[text]=1
    else:
        for text in row['text'].split(" "):
            if text in not_male:
                not_male[text]=not_male[text]+1
            else:
                not_male[text]=1


###smoothing 1
for text in not_male.keys():
    if text not in male:
        male[text]=1

for text in male.keys():
    if text not in not_male:
        not_male[text]=1

print(len(twitter_data))
print(len(male),len(not_male))