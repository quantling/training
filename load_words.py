import math
import pickle
from collections import Counter
import os
import pandas as pd
import gc
import psutil
# Load the word counts
words = pickle.load(open("words.pkl", "rb"))


skip_index = 0

data_path = "../../../../mnt/Restricted/Corpora/CommonVoiceVTL/corpus_as_df_mp_folder_en"

# Initialize counters for test, validation, and training splits
test_words = Counter()
validation_words = Counter()
training_words = Counter()

# Distribute word counts into test, validation, and training
for word in words:
    word_count = words[word]
    ten_percent = math.ceil(word_count / 10)
    word_amount_for_test = ten_percent if ten_percent > 1 else 1
    test_words[word] = word_amount_for_test
    word_count -= word_amount_for_test
    word_amount_for_validation = ten_percent if ten_percent < math.ceil(word_count / 2) else math.ceil(word_count / 2)
    validation_words[word] = word_amount_for_validation
    word_count -= word_amount_for_validation
    training_words[word] = word_count
    assert word_count + word_amount_for_validation + word_amount_for_test == words[word]

print(test_words["the"], validation_words["the"], training_words["the"], words["the"])

print(test_words.total(), validation_words.total(), training_words.total(), words.total())
print(test_words.total() + validation_words.total() + training_words.total(), words.total())
print(test_words.total() / words.total(), validation_words.total() / words.total(), training_words.total() / words.total())

# Save the counters
pickle.dump(test_words, open("test_words.pkl", "wb"))
pickle.dump(validation_words, open("validation_words.pkl", "wb"))
pickle.dump(training_words, open("training_words.pkl", "wb"))

# Initialize dataframes for test, validation, and training splits


# Process each file in the directory
sorted_files = sorted(os.listdir(data_path))  # Sorting files for better debugging
for file in sorted_files:
  
    if file.endswith(".pkl"):
        print(f"Processing {file}")
        data = pd.read_pickle(os.path.join(data_path, file))

        # Split data based on the counters
        test_rows = []
        validation_rows = []
        training_rows = []
        unique_identifier = file.split("df")[1]
        print(unique_identifier)
        for _, row in data.iterrows():
            word = row["label"]  
            if test_words[word] > 0:
                test_words[word] -= 1
                test_rows.append(row)
            elif validation_words[word] > 0:
                validation_words[word] -= 1
                validation_rows.append(row)
            else:
                training_rows.append(row)

        # Append rows to respective dataframes
        pd.DataFrame(test_rows).to_pickle(os.path.join(data_path, f"test_data{file.split('df')[1]}"))
        pd.DataFrame(validation_rows).to_pickle(os.path.join(data_path, f"validation_data{file.split('df')[1]}"))
        pd.DataFrame(training_rows).to_pickle(os.path.join(data_path, f"training_data{file.split('df')[1]}"))
        # Free up memory
        del data
        del test_rows, validation_rows, training_rows
        print(f"Memory usage: {psutil.virtual_memory().percent}%")

        gc.collect()




print("Data splitting completed successfully.")
