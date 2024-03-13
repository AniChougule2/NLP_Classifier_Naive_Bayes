train_file_path = "Tweet_data_for_gender_guessing/twitgen_train_201906011956.csv"
valid_file_path = "Tweet_data_for_gender_guessing/twitgen_valid_201906011956.csv"
merged_file_path = "Tweet_data_for_gender_guessing/merged_data.csv"

with open(train_file_path, 'r', encoding='utf-8') as train_file, open(merged_file_path, 'w', encoding='utf-8') as merged_file:
    for line in train_file:
        merged_file.write(line)

with open(valid_file_path, 'r', encoding='utf-8') as valid_file:
    next(valid_file)
    with open(merged_file_path, 'a', encoding='utf-8') as merged_file:
        for line in valid_file:
            merged_file.write(line)
