import logging
import os

logging.basicConfig(level=logging.INFO)
import pandas as pd
import pickle
from tqdm import tqdm
import psutil

import argparse


def collect_fast_text_vectors(data_path, split_index, language):
    data_path = data_path + language
    sorted_files = sorted(os.listdir(data_path))
    filtered_files = [
        file
        for file in sorted_files
        if file.startswith("corpus_as_df_mpepoch_") and file.endswith(".pkl")
    ]
    for files in filtered_files:
        logging.info(f"Processing {files}")
    vector_dict = {}
    i = 0
    print(f"Starting from {split_index}")
    for files in tqdm(filtered_files):

        if i < split_index:
            i += 1
            continue
        data = None
        if i % 10 == 0:
            pickle.dump(vector_dict, open(f"vectors_{i}_{language}.pkl", "wb"))
        if files.endswith(".pkl"):
            data = pd.read_pickle(os.path.join(data_path, files))
            if isinstance(data, pd.DataFrame):

                for _, row in data.iterrows():
                    vector_dict[row["label"]] = [row["vector"], row["lexical_word"]]
            else:
                logging.info(f"Data is not a DataFrame: {files}")

        i += 1
        logging.info(f"Processed {i} files")
        logging.info(files)
        logging.info(f"Memory usage: {psutil.virtual_memory().percent}%")
        del data
    logging.info("Done")
    pickle.dump(vector_dict, open(f"vectors_{language}.pkl", "wb"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Collect words from a folder of pickled dataframes"
    )
    parser.add_argument(
        "--data_path",
        help="Path to the folder containing the pickled dataframes",
        default="../../../../../../mnt/Restricted/Corpora/CommonVoiceVTL/corpus_as_df_mp_folder_",
    )
    parser.add_argument("--split_index", help="Index to start from", default=0, type=int)
    parser.add_argument("--language", help="Language of the data", default="de")
    args = parser.parse_args()
    collect_fast_text_vectors(args.data_path,int( args.split_index), args.language)
