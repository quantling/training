

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

def collect_words(data_path, collumn_name):
    #sorting the files for better debugging
   


    sorted_sentences = sorted(os.listdir(data_path))

    for files in sorted_sentences:
        logging.info(f"Processing {files}")
    words = Counter()
    i = 0
    word_list = []
    for files in tqdm(sorted_sentences):
            data = None
            if i % 10 == 0:
                pickle.dump(words,open(f"words_{i}.pkl","wb"))
            if files.endswith(".pkl"):
                data = pd.read_pickle(os.path.join(data_path,files))
                if isinstance(data, pd.DataFrame):
                    
                    for i,row in data.iterrows():
                        word = row[collumn_name]
                        cleaned_word = replace_special_chars(word).lower()
                        mfa_word = row["mfa_word"]
                        cleaned_mfa_word = replace_special_chars(mfa_word).lower()
                        if cleaned_word != cleaned_mfa_word:
                            print(f"Word: {word} MFA: {mfa_word}, no match since cleaned_word: {cleaned_word} cleaned_mfa_word: {cleaned_mfa_word}")
                            continue
                        words[word] += 1
                        word_list.append(word)
                else:
                    print(f"Data is not a DataFrame: {files}")
            []
            i += 1
            temp_path = os.path.join(data_path,files) + "_temp"
            data.to_pickle(temp_path)
            os.rename(temp_path, os.path.join(data_path,files))
            print(f"Processed {i} files")
            print(files)
            print(f"Memory usage: {psutil.virtual_memory().percent}%")
            del data
    print("Done")
    pickle.dump(words,open("words.pkl","wb"))
    pickle.dump(word_list,open("word_list.pkl","wb")) # I guess an ordered list of words is also useful


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Collect words from a folder of pickled dataframes")
    parser.add_argument("--data_path", help="Path to the folder containing the pickled dataframes", default = "../../../../mnt/Restricted/Corpora/CommonVoiceVTL/corpus_as_df_mp_folder_")
    parser.add_argument("--collumn_name", help="The name of the collumn containing the words", default = "lexical_word")
    parser.add_argument("--language", help="The language of the data", default = "de")
    args = parser.parse_args()
    data_path = args.data_path + args.language
    collect_words(data_path, args.collumn_name)