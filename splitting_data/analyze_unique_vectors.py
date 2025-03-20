import pandas
import os
import pickle
import argparse
from split_utils import replace_special_chars, open_pickle
data_path = "../../../../../mnt/Restricted/Corpora/CommonVoiceVTL/corpus_as_df_mp_folder"


    


def analyze_unique_vectors(data_path, language):
    words_file_name = "unique_words_" + language + ".pkl"
    words_file = os.path.join(data_path, words_file_name)
    words = open_pickle(words_file)
    print(f"Unique words: {len(words)}")
    print(f" {words.head()}")
    print(f" {words.columns}")
    print("-----------------------")
    vectors_file_name = "unique_vectors_" + language + ".pkl"
    vectors_file = os.path.join(data_path, vectors_file_name)
    vectors = open_pickle(vectors_file)
    print(f"Unique vectors: {len(vectors)}")
    print(f" {vectors.head()}")
    print(f" {vectors.columns}")
    print("-----------------------")

    dublicate_words_mask = words["lexical_word"].duplicated(keep=False)
    dublicate_words = words[dublicate_words_mask]
    print(f"Dublicate words: {len(dublicate_words)}")
    print(f"{dublicate_words.head()}")
    print(f"{dublicate_words.tail()}")
    print("-----------------------")
    dublicate_vectors_mask = vectors["tuple_vector"].duplicated(keep=False,)
    dublicate_vectors = vectors[dublicate_vectors_mask]
    print(f"Dublicate vectors: {len(dublicate_vectors)}")
    print(f"{dublicate_vectors.head()}")
    print(f"{dublicate_vectors.tail()}")
    dublicate_vectors["lexical_word"].to_csv(f"dublicate_vectors_{language}.csv")
    print("-----------------------")
    dublicate_words_in_vectors = dublicate_vectors["lexical_word"].duplicated(keep=False)
    dublicate_words_in_vectors = dublicate_vectors[dublicate_words_in_vectors]
    print(f"Dublicate words in vectors: {len(dublicate_words_in_vectors)}")
    print(f"{dublicate_words_in_vectors.head()}")
    print(f"{dublicate_words_in_vectors.tail()}")
    print("-----------------------")
    ### get incongruent words
    cleaned_words = words.copy()
    cleaned_words["lexical_word"] = words["lexical_word"].apply(lambda x: replace_special_chars(x).lower())
    """"
    
    print(f"lenght of cleaned words: {len(cleaned_words)}")
    cleaned_words.drop_duplicates(keep=False, inplace=True)
    print(f"Unique cleaned words: {len(cleaned_words)}")
    print(f"{cleaned_words.head()}")

    cleaned_words_mask = cleaned_words["lexical_word"].duplicated(keep=False)
    dublicate_cleaned_words = cleaned_words[cleaned_words_mask]
    print(f"Dublicate cleaned words: {len(dublicate_cleaned_words)}")
    print(f"{dublicate_cleaned_words.head()}")
    print(f"{dublicate_cleaned_words.tail()}")
    print("-----------------------")
    """
    cleaned_words["mfa_word"] = cleaned_words["mfa_word"].apply(lambda x: replace_special_chars(x).lower())
    incongruent_words = cleaned_words[cleaned_words["lexical_word"] != cleaned_words["mfa_word"]]
    print(f"Incongruent words: {len(incongruent_words)}")
    print(f"This is the percentage of incongruent words: {len(incongruent_words)/len(cleaned_words)}")
    print(f"{incongruent_words.head()}")
    print(f"{incongruent_words.tail()}")
    incongruent_words.to_csv(f"incongruent_words_{language}.csv")
    print("-----------------------")
    
    















if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Analyze unique vectors')
    parser.add_argument("--language", type=str, default="en", help="Language of the data")
    args = parser.parse_args()
    language = args.language
    data_path = data_path + "_" + language
    print(data_path)
    analyze_unique_vectors(data_path, language)