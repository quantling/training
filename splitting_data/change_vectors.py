import logging
import os

logging.basicConfig(level=logging.INFO)
import pandas as pd
import pickle
from tqdm import tqdm
from collections import Counter
import math

import psutil
import fasttext
import fasttext.util
import argparse


def collect_fast_text_vectors(data_path, split_index, language, skip_index):
    fasttext.util.download_model(language, if_exists="ignore")
    model = fasttext.load_model(f"cc.{language}.300.bin")

    data_path = data_path + language
    sorted_files = sorted(os.listdir(data_path))
    filtered_files = [
        file
        for file in sorted_files
        if file.startswith("corpus_as_df_mp") and file.endswith(".pkl")
    ]
    for files in filtered_files:
        logging.info(f"Processing {files}")
    if split_index == 0:
        vector_dict = {}
        word_counter = Counter()
    else:
        vector_dict = pickle.load(open(f"vectors_{split_index}_{language}.pkl", "rb"))
        word_counter = pickle.load(open(f"word_counter_{split_index}_{language}.pkl", "rb"))
        logging.info(f"Loaded vectors_{split_index}_{language}.pkl")
        logging.info(f"Loaded word_counter_{split_index}_{language}.pkl")
    
    i = 0
   
    for files in tqdm(filtered_files):

        if i < split_index:
            i += 1
            logging.info(f"Skipping {files}")
            continue

        if i in skip_index:
            i += 1
            logging.info(f"Skipping {files}")
            continue
        data = None
        if i % 5 == 0:
            pickle.dump(vector_dict, open(f"vectors_{i}_{language}.pkl", "wb"))
            pickle.dump(word_counter, open(f"word_counter_{i}_{language}.pkl", "wb"))
            logging.info(f"Saved vectors_{i}_{language}.pkl")
            logging.info(f"Saved word_counter_{i}_{language}.pkl")
        if files.endswith(".pkl"):
            data = pd.read_pickle(os.path.join(data_path, files))
            if isinstance(data, pd.DataFrame):
                data.rename(columns={"label": "mfa_word"}, inplace=True)
                data["vector"] = data["lexical_word"].apply(
                    lambda word: model.get_word_vector(word)
                )

                for _, row in data.iterrows():
                    word_counter[row["lexical_word"]] += 1
                    vector_dict[row["lexical_word"]] = [row["vector"], row["mfa_word"]]

                logging.info(f"Processed {files}")
                logging.info(f"Saving {files}. Quitting now might damage the data.")
                temporary_file_name = files + ".temp"
                pickle.dump(
                    data, open(os.path.join(data_path, temporary_file_name), "wb")
                )

                os.replace(os.path.join(data_path, temporary_file_name), os.path.join(data_path, files))


                logging.info(f" Saved {files}")
            else:
                logging.info(f"Data is not a DataFrame: {files}")
        del data
        i += 1
        logging.info(f"Processed {i} files")
        logging.info(f"Memory usage: {psutil.virtual_memory().percent}%")

    logging.info("Done")

    pickle.dump(vector_dict, open(f"vectors_{language}.pkl", "wb"))
    pickle.dump(word_counter, open(f"word_counter_{language}.pkl", "wb"))


def split_words(data_path, skip_index, language):
    # Load the word counts
    words = pickle.load(open(f"word_counter_{language}.pkl", "rb"))

    # Initialize counters for test, validation, and training splits
    test_words = Counter()
    validation_words = Counter()
    training_words = Counter()

    # Distribute word counts into test, validation, and training
    print(words.items())
    for word in words:

        word_count = words[word]
        ten_percent = math.ceil(word_count / 10)
        word_amount_for_test = ten_percent if ten_percent > 1 else 1
        test_words[word] = word_amount_for_test
        word_count -= word_amount_for_test
        word_amount_for_validation = (
            ten_percent
            if ten_percent < math.ceil(word_count / 2)
            else math.ceil(word_count / 2)
        )
        validation_words[word] = word_amount_for_validation
        word_count -= word_amount_for_validation
        training_words[word] = word_count
        assert (
            word_count + word_amount_for_validation + word_amount_for_test
            == words[word]
        )

    print(
        test_words["the"], validation_words["the"], training_words["the"], words["the"]
    )

    print(
        test_words.total(),
        validation_words.total(),
        training_words.total(),
        words.total(),
    )
    print(
        test_words.total() + validation_words.total() + training_words.total(),
        words.total(),
    )
    print(
        test_words.total() / words.total(),
        validation_words.total() / words.total(),
        training_words.total() / words.total(),
    )

    # Save the counters
    pickle.dump(test_words, open(f"test_words_{language}.pkl", "wb"))
    pickle.dump(validation_words, open(f"validation_words_{language}.pkl", "wb"))
    pickle.dump(training_words, open(f"training_words_{language}.pkl", "wb"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Collect words from a folder of pickled dataframes"
    )
    parser.add_argument(
        "--data_path",
        help="Path to the folder containing the pickled dataframes",
        default="../../../../../../mnt/Restricted/Corpora/CommonVoiceVTL/corpus_as_df_mp_folder_",
    )
    parser.add_argument(
        "--split_index",
        help="Index to start from",
        default=0,
        type=int,
        required=False,
    )
    parser.add_argument("--language", help="Language of the data", default="en")
    parser.add_argument("--skip_index", help="Index to skip", default=[], type=list)
    args = parser.parse_args()
    collect_fast_text_vectors(
        args.data_path, args.split_index, args.language, args.skip_index
    )
    split_words(args.data_path, args.split_index, args.language)
