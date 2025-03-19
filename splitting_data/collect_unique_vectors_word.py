import pandas as pd
import argparse
import os
import logging
import pickle
from tqdm import tqdm
import time
logging.basicConfig(level=logging.INFO)
import psutil



def wait_for_file_stable(filepath, check_interval=0.1, stable_checks=3):
    """Wait until the file is no longer being modified."""
    last_sizes = []
    
    while True:
        try:
            current_size = os.path.getsize(filepath)
            last_sizes.append(current_size)

            # Keep only the last N size measurements
            if len(last_sizes) > stable_checks:
                last_sizes.pop(0)

            # If the last few size measurements are identical, assume writing is done
            if len(set(last_sizes)) == 1:
                break
        except FileNotFoundError:
            pass  # If the file disappears briefly, just retry

        time.sleep(check_interval)

def collect_unique_vectors(data_path, language, skip_index):
    data_path = data_path + language
    sorted_files = sorted(os.listdir(data_path))  
    filtered_files = [
        file
        for file in sorted_files
        if file.startswith("corpus_as_df_mp") and file.endswith(".pkl")
    ]
    unique_data = pd.DataFrame()
    unique_words = pd.DataFrame()
    i = 0
    for files in tqdm(filtered_files):  
        data = None
        if i < skip_index:
            i += 1
            logging.info(f"Skipping {files}")
            continue
        
        logging.info(f"Reading {files}")
        data = pd.read_pickle(os.path.join(data_path, files))
        if isinstance(data, pd.DataFrame):
            new_data = pd.DataFrame()
            new_words = pd.DataFrame()

            new_data['tuple_vector'] = data['vector'].apply(lambda x: tuple(x.tolist()))

            new_data['lexical_word'] = data['lexical_word']

            new_words['mfa_word'] = data['mfa_word']
            new_words['lexical_word'] = data['lexical_word']

            new_data = pd.concat((unique_data, new_data), axis=0)
            new_words = pd.concat((unique_words, new_words), axis=0)

            unique_data = new_data.drop_duplicates()
            unique_words = new_words.drop_duplicates()
            logging.info(f" {len(new_data) - len(unique_data)} duplicates removed")
            logging.info(f"Unique words: {len(unique_words)}")
            logging.info(f"Unique vectors: {len(unique_data)}")
            if i % 5 == 0:
                unique_data.to_pickle(f"unique_vectors_{i}_{language}.pkl")
                unique_words.to_pickle(f"unique_words_{i}_{language}.pkl")
                logging.info(f"Saved unique_words_{i}_{language}.pkl")
                logging.info(f"Saved unique_vectors_{i}_{language}.pkl")
            del new_data
            del new_words
            logging.info(f"Memory usage: {psutil.virtual_memory().percent}")
            i += 1
        del data
    
    unique_data.to_pickle(f"unique_vectors_{language}.pkl")
    unique_words.to_pickle(f"unique_words_{language}.pkl")

def collect_unique_words(data_path, language, skip_index):
    data_path = data_path + language
    sorted_files = sorted(os.listdir(data_path))
    filtered_files = [
        file
        for file in sorted_files
        if file.startswith("corpus_as_df_mp") and file.endswith(".pkl")
    ]
    unique_data = pd.DataFrame()
    i = 0
    for files in tqdm(filtered_files):
        data = None
        if i < skip_index:
            i += 1
            logging.info(f"Skipping {files}")
            continue
        logging.info(f"Reading {files}")
        wait_for_file_stable(os.path.join(data_path, files))
        data = pd.read_pickle(os.path.join(data_path, files))
        if isinstance(data, pd.DataFrame):
            new_data = pd.DataFrame()
            new_data['lexical_word'] = data['lexical_word']
            new_data['mfa_word'] = data['mfa_word']
            unique_data = pd.concat((unique_data, new_data), axis=0)
            unique_data = unique_data.drop_duplicates()
            logging.info(f"Unique words: {len(unique_data)}")
            if i % 5 == 0:
                unique_data.to_pickle(f"unique_words_{i}_{language}.pkl")
                logging.info(f"Saved unique_words_{i}_{language}.pkl")
            del new_data
            logging.info(f"Memory usage: {psutil.virtual_memory().percent}")
            i += 1

        del data
    unique_data.to_pickle(f"unique_words_{language}.pkl")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Collect unique vectors and words from the dataframes in the folder"
    )
    parser.add_argument(
        "--data_path",
        help="Path to the folder containing the pickled dataframes",
        default="../../../../../../mnt/Restricted/Corpora/CommonVoiceVTL/corpus_as_df_mp_folder_",
    )
    parser.add_argument(
        "--language",
        help="Language to process",
        default="de",
    )
    parser.add_argument(
        "--skip_index",
        help="Index to start from",
        default=0,
        type=int,
        required=False,
    )
    parser.add_argument("--no_vectors", help="Do not collect vectors", action="store_false")
    parser.add_argument("--no_words", help="Do not collect words", action="store_false")
    args = parser.parse_args()
    if args.no_vectors:
        collect_unique_vectors(args.data_path, args.language, args.skip_index)
        args.skip_index = 0
    if args.no_words:
        collect_unique_words(args.data_path, args.language, args.skip_index)
    logging.info("Done")
