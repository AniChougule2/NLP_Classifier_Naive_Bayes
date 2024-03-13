import pandas as pd
import spacy
import emoji
import re


def Preprocess(twitter_data):
    eng_corpus = spacy.load("en_core_web_sm")
    data_clean = [] 

    for _, row in twitter_data.iterrows():
        sentence = eng_corpus(row['text'])
        lemmas = [emoji.demojize(str(token.text)) if emoji.demojize(token.text) != token.text else token.lemma_ for token in sentence if not token.is_stop and not token.is_punct]
        lemmas = [lem.replace(':', '').replace('_', ' ') for lem in lemmas]
        clean_lemmas = re.sub(r'[\n\t]', ' ', " ".join(lemmas))
        clean_lemmas = re.sub(r'[ ]+', ' ', clean_lemmas)

        data_clean.append({
            'id': row['id'],
            'time': row['time'],
            'text': clean_lemmas,
            'male': row['male']
        })

    return pd.DataFrame(data_clean) 


twitter_data = pd.read_csv("Tweet_data_for_gender_guessing/merged_data.csv")
output_file_path = "Tweet_data_for_gender_guessing/twitter_train_data_clean.csv"

twitter_data_clean = Preprocess(twitter_data)
twitter_data_clean.to_csv(output_file_path, index=False)


twitter_data = pd.read_csv("Tweet_data_for_gender_guessing/twitgen_test_201906011956.csv")
output_file_path = "Tweet_data_for_gender_guessing/twitter_test_data_clean.csv"
twitter_data_clean = Preprocess(twitter_data)
twitter_data_clean.to_csv(output_file_path, index=False)