


import argparse
import os
import pickle
import logging
import pandas as pd
from tqdm import tqdm
import psutil

logging.basicConfig(level=logging.INFO)

def reset_index(data_path, language, skip_index):
    data_path = data_path + language
    sorted_files = sorted(os.listdir(data_path))
    filtered_files = [
        file
        for file in sorted_files
        if file.startswith("corpus_as_df_mp") and file.endswith(".pkl")
    ]
    for files in filtered_files:
        logging.info(f"Processing {files}")
    i = 0
    for files in tqdm(filtered_files):
        if i < skip_index:
            i += 1
            logging.info(f"Skipping {files}")
            continue
        data = None
        if files.endswith(".pkl"):
            logging.info(f"Reading {files}")
            data = pd.read_pickle(os.path.join(data_path, files))
            if isinstance(data, pd.DataFrame):
                data.reset_index(inplace=True, drop=True)
                logging.info(data.head())
                temp_path = os.path.join(data_path, files + ".temp")
                logging.info(f"Saving {files}")
                data.to_pickle(temp_path)
                logging.info(f"Saved {files} as {temp_path}")
                os.replace(temp_path, os.path.join(data_path, files))
                logging.info(f"Replaced {temp_path} with {os.path.join(data_path, files)}")
        
        
        del data
        logging.info(f"Using {psutil.virtual_memory().percent} percent of memory")
        i += 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Just reset the index of the dataframes in the folder"
    )
    parser.add_argument(
        "--data_path",
        help="Path to the folder containing the pickled dataframes",
        default="../../../../../../mnt/Restricted/Corpora/CommonVoiceVTL/corpus_as_df_mp_folder_",
    )
    parser.add_argument(
        "--skip_index",
        help="Index to start from",
        default=0,
        type=int,
        required=False,
    )
    parser.add_argument("--language", help="Language of the data", default="de")
    args = parser.parse_args()
    reset_index(args.data_path, args.language, args.skip_index)