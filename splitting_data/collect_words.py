

import logging
import os
logging.basicConfig(level=logging.INFO)
from collections import Counter
import os
import pandas as pd
import pickle
from tqdm import tqdm
from split_utils import replace_special_chars
import psutil

import argparse

def collect_words(data_path, collumn_name, language):

    #sorting the files for better debugging
   


    sorted_files = sorted(os.listdir(data_path))
    filtered_files = [
        file
        for file in sorted_files
        if file.startswith("corpus_as_df_mp_epoch_") and file.endswith(".pkl")
    ]

    for files in filtered_files:
        logging.info(f"Processing {files}")
    words = Counter()
    i = 0
    word_list = []
    for files in tqdm(filtered_files):
            data = None
            if i % 10 == 0:
                pickle.dump(words,open(f"word_counter_{i}.pkl","wb"))
                logging.info(f"Saved word_counter_{i}.pkl")
            if files.endswith(".pkl"):
                logging.info(f"Loading {files}")
                data = pd.read_pickle(os.path.join(data_path,files))
                if isinstance(data, pd.DataFrame):
                    
                    for i,row in data.iterrows():
                        word = row[collumn_name]
                        cleaned_word = replace_special_chars(word).lower()
                        mfa_word = row["mfa_word"]
                        cleaned_mfa_word = replace_special_chars(mfa_word).lower()
                        if cleaned_word != cleaned_mfa_word:
                            print(f"Word: {word} MFA: {mfa_word}, no match since cleaned_word: {cleaned_word} cleaned_mfa_word: {cleaned_mfa_word}")
                            data.drop(i, inplace=True)
                            continue
                        words[word] += 1
                        word_list.append(word)
                else:
                    print(f"Data is not a DataFrame: {files}")
            []
            i += 1
            temp_path = os.path.join(data_path,files) + "_temp"
            logging.info(f"Saving {files} to {temp_path}")
            data.to_pickle(temp_path)
            logging.info(f"Renaming {temp_path} to {files}")
            os.rename(temp_path, os.path.join(data_path,files))
            print(f"Processed {i} files")
            print(files)
            print(f"Memory usage: {psutil.virtual_memory().percent}%")

            total, used, free, percent = psutil.disk_usage(data_path)
            logging.info(f"Total: {total} Used: {used} Free: {free}")
            if free < 30 *10**9:
                logging.info("Less than 30GB free, stopping")
                break
            del data
    print("Done")
    pickle.dump(words,open(f"word_counter_{language}.pkl","wb"))
    pickle.dump(word_list,open("word_list.pkl","wb")) # I guess an ordered list of words is also useful


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Collect words from a folder of pickled dataframes")
    parser.add_argument("--data_path", help="Path to the folder containing the pickled dataframes", default = "../../../../../mnt/Restricted/Corpora/CommonVoiceVTL/corpus_as_df_mp_folder_")
    parser.add_argument("--collumn_name", help="The name of the collumn containing the words", default = "lexical_word")
    parser.add_argument("--language", help="The language of the data", default = "de")
    args = parser.parse_args()
    data_path = args.data_path + args.language
    collect_words(data_path, args.collumn_name, args.language)