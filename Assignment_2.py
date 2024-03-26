import pandas as pd
import spacy
import emoji
import re
import sys
import numpy as np


def Preprocess(twitter_data):
    eng_corpus = spacy.load("en_core_web_sm")
    data_clean = []

    for _, row in twitter_data.iterrows():
        row['text'] = row['text'].replace('#', ' ')
        sentence = eng_corpus(row['text'])
        lemmas = [emoji.demojize(str(token.text)) if emoji.demojize(token.text) != token.text else token.lemma_ for token in sentence if not token.is_stop and not token.is_punct]
        lemmas = [lem.replace(':', '').replace('_', ' ') for lem in lemmas]
        clean_lemmas = re.sub(r'[\r\n\t]', ' ', " ".join(lemmas))
        clean_lemmas = re.sub(r'[ ]+', ' ', clean_lemmas)
        clean_lemmas = clean_lemmas.strip()
        data_clean.append({
            'id': row['id'],
            'time': row['time'],
            'text': clean_lemmas,
            'male': row['male']
        })

    return pd.DataFrame(data_clean)

print("Waradkar Piyush A20513460")
print("Chougule Aniket A20552758")
twitter_data = pd.read_csv("Tweet_data_for_gender_guessing/merged_data.csv")
output_file_path = "Tweet_data_for_gender_guessing/twitter_train_data_clean.csv"
output_file_path_custome_train_set = "Tweet_data_for_gender_guessing/twitter_train_data_clean_Training_set.csv"
if len(sys.argv) == 2 :
    training_size = sys.argv[1]
    if not training_size.isdigit() or not 20 <= int(training_size) <= 80:
        training_size = 80
    training_size = int(training_size)
else:
    training_size = 80

# twitter_data_clean = Preprocess(twitter_data)
# twitter_data_clean.to_csv(output_file_path, index=False)
print(f"Training set size: {training_size} %")
training_fraction = training_size / 100
num_training_samples = int(len(twitter_data) * training_fraction)
twitter_data_train = twitter_data.iloc[:num_training_samples]
test_start_index = len(twitter_data) - int(len(twitter_data) * 0.2)
twitter_data_test = twitter_data.iloc[test_start_index:]
# print(len(twitter_data_train), len(twitter_data_test), len(twitter_data))
# print(training_size, test_size)
twitter_data_train = Preprocess(twitter_data_train)
twitter_data_test = Preprocess(twitter_data_test)

eng_corpus = spacy.load("en_core_web_sm")


male = {}
not_male = {}
for id, row in twitter_data_train.iterrows():
    if row['male']:
        for text in list(set(str(row['text']).split(" "))):
            if text in male:
                male[text] = male[text]+1
            else:
                male[text] = 1
    else:
        for text in list(set(str(row['text']).split(" "))):
            if text in not_male:
                not_male[text] = not_male[text]+1
            else:
                not_male[text] = 1

for text in not_male.keys():
    if text not in male:
        male[text] = 0

for text in male.keys():
    if text not in not_male:
        not_male[text] = 0

male_df = pd.DataFrame(list(male.items()), columns=['word', 'present'])
not_male_df = pd.DataFrame(list(not_male.items()), columns=['word', 'present'])

male_df.drop(index=0, inplace=True)
not_male_df.drop(index=0, inplace=True)

# print(twitter_data.shape[0])
# print(male_df.shape[0], not_male_df.shape[0])
# print(male_df.head())
# print(not_male_df.head())

merge_df = pd.merge(male_df, not_male_df, on="word",
                    suffixes=("_male", "_not_male"))
# print(merge_df.head())
# print(merge_df.shape[0])

total_male_count = male_df['present'].sum()
total_not_male_count = not_male_df['present'].sum()

merge_df['male_probability'] = (
    merge_df['present_male'] + 1) / (total_male_count + merge_df.shape[0])
merge_df['not_male_probability'] = (
    merge_df['present_not_male'] + 1) / (total_not_male_count + merge_df.shape[0])

df_train_probabilty = merge_df
# print(df_train_probabilty.sort_values(by="male_probability", ascending=False).head())

male_prop = twitter_data['male'].mean()
not_male_prop = 1 - male_prop

# print(male_prop)
# print(not_male_prop)


pred_dict = {}

for id, row in twitter_data_test.iterrows():

    pred_male = np.log(male_prop)
    pred_not_male = np.log(not_male_prop)
    for text in list(set(str(row['text']).split(" "))):
        matching_rows = df_train_probabilty[df_train_probabilty['word'] == text]
        if not matching_rows.empty:
            pred_male += np.log(matching_rows['male_probability'].iloc[0])
            pred_not_male += np.log(
                matching_rows['not_male_probability'].iloc[0])

    pred_dict[id] = pred_male > pred_not_male

pred_df = pd.DataFrame(list(pred_dict.items()), columns=['id', 'Prediction'])

# print(pred_df.head())
# print(twitter_data_test.head())

twitter_data_test['Prediction'] = pred_df['Prediction'].values

# print(twitter_data_test.head(20))

TP = ((twitter_data_test['Prediction'] == True) & (
    twitter_data_test['male'] == True)).sum()
TN = ((twitter_data_test['Prediction'] == False) & (
    twitter_data_test['male'] == False)).sum()
FP = ((twitter_data_test['Prediction'] == True) & (
    twitter_data_test['male'] == False)).sum()
FN = ((twitter_data_test['Prediction'] == False) & (
    twitter_data_test['male'] == True)).sum()

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


def Preprocess(sentence):
    eng_corpus = spacy.load("en_core_web_sm")
    sentence = eng_corpus(sentence)
    lemmas = [emoji.demojize(str(token.text)) if emoji.demojize(
        token.text) != token.text else token.lemma_ for token in sentence if not token.is_stop and not token.is_punct]
    lemmas = [lem.replace(':', '').replace('_', ' ') for lem in lemmas]
    clean_lemmas = re.sub(r'[\r\n\t]', ' ', " ".join(lemmas))
    clean_lemmas = re.sub(r'[ ]+', ' ', clean_lemmas)
    clean_lemmas = clean_lemmas.strip()
    return predict(clean_lemmas)


def predict(sentence):
    pred_male = np.log(male_prop)
    pred_not_male = np.log(not_male_prop)
    for text in list(set(str(sentence).split(" "))):
        matching_rows = df_train_probabilty[df_train_probabilty['word'] == text]
        if not matching_rows.empty:
            pred_male += np.log(matching_rows['male_probability'].iloc[0])
            pred_not_male += np.log(
                matching_rows['not_male_probability'].iloc[0])

    return np.exp(pred_male), np.exp(pred_not_male),sentence


while True:
    sentence = input("Enter your sentence: ")
    pred_male, pred_not_male,sentence = Preprocess(sentence)
    class_label = "male" if pred_male > pred_not_male else "not_male"
    print("Sentence S:")
    print(f"\n{sentence}\n")
    print(f"was classified as {class_label}.")
    print(f"P(male | S) = {pred_male}")
    print(f"P(not_male | S) = {pred_not_male}\n")

    choice = input("Do you want to enter another sentence [Y/N]? ")
    if choice.lower() not in ['y', 'yes']:
        break

