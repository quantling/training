import pandas as pd
import os
import argparse
import pickle

def open_pickle(file_path):
    with open(file_path, 'rb') as f:
        return pickle.load(f)


def check_if_complete(data_path, language):
    word_counter = open_pickle(f"words_{language}.pkl")
    test_words = open_pickle(f"test_words_{language}.pkl")
    validation_words = open_pickle(f"validation_words_{language}.pkl")
    training_words = open_pickle(f"training_words_{language}.pkl")

    files = os.listdir(data_path)
    filtered_files = sorted([file for file in files if (file.endswith(".pkl") and  "_data_"  in file  )])
    test_files = [file for file in filtered_files if "test" in file]
    validation_files = [file for file in filtered_files if  "validation" in file]
    training_files = [file for file in filtered_files if   "training" in    file]

    print(filtered_files)

    assert len(test_files)== len(validation_files) == len(training_files)

    for file in test_files:
        print(file)
        test_data = open_pickle(os.path.join(data_path, file))
        for _, row in test_data.iterrows():
            if row["label"] in test_words:
                if test_words[row["label"]] == 0:
                    raise Exception(f"Test data is too large for word {row['label']}")
                test_words[row["label"]] -= 1

                if row["label"] in word_counter:
                    if word_counter[row["label"]] == 0:
                        raise Exception(f"Test data is too large for word {row['label']}")
                    word_counter[row["label"]] -= 1
    test_words_sum = sum(test_words.values())
    assert test_words_sum == 0, "Not all counts are zero in test_words"   
    for file in validation_files:
        print(file)
        validation_data = open_pickle(os.path.join(data_path, file))
        for _, row in validation_data.iterrows():
            if row["label"] in validation_words:
                if validation_words[row["label"]] == 0:
                    raise Exception(f"Validation data is too large for word {row['label']}")
                validation_words[row["label"]] -= 1

                if row["label"] in word_counter:
                    if word_counter[row["label"]] == 0:
                        raise Exception(f"Validation data is too large for word {row['label']}")
                    word_counter[row["label"]] -= 1
    validation_words_sum = sum(validation_words.values())
    assert validation_words_sum == 0, "Not all counts are zero in validation_words"
    for file in training_files:
        print(file)
        training_data = open_pickle(os.path.join(data_path, file))
        for _, row in training_data.iterrows():
            if row["label"] in training_words:
                if training_words[row["label"]] == 0:
                    raise Exception(f"Training data is too large for word {row['label']}")
                training_words[row["label"]] -= 1

                if row["label"] in word_counter:
                    if word_counter[row["label"]] == 0:
                        raise Exception(f"Training data is too large for word {row['label']}")
                    word_counter[row["label"]] -= 1
    training_words_sum = sum(training_words.values())
    assert training_words_sum == 0, "Not all counts are zero in training_words"
    word_counter_sum = sum(word_counter.values())
  
   
  
    assert word_counter_sum == 0, "Not all counts are zero in word_counter"
    
    print(test_words_sum, validation_words_sum, training_words_sum, word_counter_sum)
    print(test_words)
    print(validation_words)
    print(training_words)
    print(word_counter)
  
   
    print("All data is complete")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Check if the data is complete')
    parser.add_argument("--data_path", type=str, default="../../../../mnt/Restricted/Corpora/CommonVoiceVTL/corpus_as_df_mp_folder", help="Path to the data")
    parser.add_argument("--language", type=str, default='de', help="Language of the data")
    args = parser.parse_args()

    language = args.language
    data_path = args.data_path + f"_{language}"
    check_if_complete(data_path,language)