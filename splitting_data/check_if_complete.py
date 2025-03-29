import pandas as pd
import os
import argparse
import pickle
import psutil
from tqdm import tqdm
import logging
def open_pickle(file_path):
    with open(file_path, 'rb') as f:
        return pickle.load(f)


def check_if_complete(data_path, language):
    word_counter = open_pickle(f"word_counter_{language}.pkl")
    test_words = open_pickle(f"test_words_{language}.pkl")
    validation_words = open_pickle(f"validation_words_{language}.pkl")
    training_words = open_pickle(f"training_words_{language}.pkl")

    files = os.listdir(data_path)
    filtered_files = sorted([file for file in files if (file.endswith(".pkl") and  "_data_"  in file  )])
    test_files = [file for file in filtered_files if "test" in file]
    validation_files = [file for file in filtered_files if  "validation" in file]
    training_files = [file for file in filtered_files if   "training" in    file]

    print(filtered_files)

    print(len(test_files), len(validation_files), len(training_files))

    errors_in_test = 0
    errors_in_validation = 0
    errors_in_training = 0
    for file in tqdm(test_files):
        
        print(file)
        test_data = open_pickle(os.path.join(data_path, file))
        for _, row in test_data.iterrows():
            if row["lexical_word"] in test_words:
                if test_words[row["lexical_word"]] == 0:
                    logging.warning(f"Test data is too large for word {row['lexical_word'] } in the test_words")
                    errors_in_test += 1
                test_words[row["lexical_word"]] -= 1

                if row["lexical_word"] in word_counter:
                    if word_counter[row["lexical_word"]] == 0:
                        logging.warning(f"Test data is too large for word {row['lexical_word']} in the word_counter")
                        errors_in_test += 1
                    word_counter[row["lexical_word"]] -= 1
        del test_data
        print(f"Memory usage: {psutil.virtual_memory().percent}%")
    test_words_sum = sum(test_words.values())
    if test_words_sum == 0 : logging.warning("Not all counts are zero in test_words")   
    for file in tqdm(validation_files):
        print(file)
        validation_data = open_pickle(os.path.join(data_path, file))
        for _, row in validation_data.iterrows():
            if row["lexical_word"] in validation_words:
                if validation_words[row["lexical_word"]] == 0:
                    logging.warning(f"Validation data is too large for word {row['lexical_word']} in the validation_words")
                    errors_in_validation += 1
                validation_words[row["lexical_word"]] -= 1

                if row["lexical_word"] in word_counter:
                    if word_counter[row["lexical_word"]] == 0:
                        logging.warning(f"Validation data is too large for word {row['lexical_word']} in the word_counter")
                        errors_in_validation += 1
                    word_counter[row["lexical_word"]] -= 1
        del validation_data
        print(f"Memory usage: {psutil.virtual_memory().percent}%")
    validation_words_sum = sum(validation_words.values())
    if validation_words_sum == 0: logging.warning("Not all counts are zero in validation_words")
    for file in tqdm(training_files):
        print(file)
        training_data = open_pickle(os.path.join(data_path, file))
        for _, row in training_data.iterrows():
            if row["lexical_word"] in training_words:
                if training_words[row["lexical_word"]] == 0:
                    logging.warning(f"Training data is too large for word {row['lexical_word']} in the training_words")
                    errors_in_training += 1
                training_words[row["lexical_word"]] -= 1

                if row["lexical_word"] in word_counter:
                    if word_counter[row["lexical_word"]] == 0:
                        logging.warning(f"Training data is too large for word {row['lexical_word']} in the word_counter")
                        errors_in_training += 1

                    word_counter[row["lexical_word"]] -= 1
        del training_data
        print(f"Memory usage: {psutil.virtual_memory().percent}%")
    training_words_sum = sum(training_words.values())
   # assert training_words_sum == 0, "Not all counts are zero in training_words"
    word_counter_sum = sum(word_counter.values())
  
   
  
    if  word_counter_sum != 0: logging.warning( "Not all counts are zero in word_counter")
    
    print(test_words_sum, validation_words_sum, training_words_sum, word_counter_sum)
   
  
   
    print(errors_in_test, errors_in_validation, errors_in_training)
    print("Done")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Check if the data is complete')
    parser.add_argument("--data_path", type=str, default="../../../../../mnt/Restricted/Corpora/CommonVoiceVTL/corpus_as_df_mp_folder", help="Path to the data")
    parser.add_argument("--language", type=str, default='de', help="Language of the data")
    args = parser.parse_args()

    language = args.language
    data_path = args.data_path + f"_{language}"
    check_if_complete(data_path,language)
