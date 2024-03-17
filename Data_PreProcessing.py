import pandas as pd
import spacy
import emoji
import re
import sys
from sklearn.model_selection import train_test_split 

def Preprocess(twitter_data):
    eng_corpus = spacy.load("en_core_web_sm")
    data_clean = [] 

    for _, row in twitter_data.iterrows():
        sentence = eng_corpus(row['text'])
        lemmas = [emoji.demojize(str(token.text)) if emoji.demojize(token.text) != token.text else token.lemma_ for token in sentence if not token.is_stop and not token.is_punct]
        lemmas = [lem.replace(':', '').replace('_', ' ') for lem in lemmas]
        clean_lemmas = re.sub(r'[\r\n\t]', ' ', " ".join(lemmas))
        clean_lemmas = re.sub(r'[ ]+', ' ', clean_lemmas)
        clean_lemmas=clean_lemmas.strip()
        data_clean.append({
            'id': row['id'],
            'time': row['time'],
            'text': clean_lemmas,
            'male': row['male']
        })

    return pd.DataFrame(data_clean) 


twitter_data = pd.read_csv("Tweet_data_for_gender_guessing/merged_data.csv")
output_file_path = "Tweet_data_for_gender_guessing/twitter_train_data_clean.csv"
output_file_path_custome_train_set = "Tweet_data_for_gender_guessing/twitter_train_data_clean_Training_set.csv"
if len(sys.argv) <2:
    print("required argument not provided")
    print("Exiting........")
    sys.exit(0)
Training_size=-1
try:
    Training_size=int(sys.argv[1])
except:
    print("Non-integer value ")
    print("defaulting value to 80!!!!")
    Training_size=80
if Training_size<20 or Training_size>80 :
    print("Training size out of range!!!!!!")
    print("Defaulting to 80%!!!!!!")
    Training_size=80

# twitter_data_clean = Preprocess(twitter_data)
# twitter_data_clean.to_csv(output_file_path, index=False)
if Training_size==80:
    twitter_data_train=twitter_data
else:
    twitter_data_train,_=train_test_split(twitter_data,train_size=(Training_size+20)/100)
print(len(twitter_data_train),len(twitter_data))
twitter_data_train_clean = Preprocess(twitter_data_train)
twitter_data_train_clean.to_csv(output_file_path_custome_train_set, index=False)
twitter_data = pd.read_csv("Tweet_data_for_gender_guessing/twitgen_test_201906011956.csv")
output_file_path = "Tweet_data_for_gender_guessing/twitter_test_data_clean.csv"
twitter_data_clean = Preprocess(twitter_data)
twitter_data_clean.to_csv(output_file_path, index=False)