import argparse
import os
import pickle
from scipy.spatial import KDTree
import numpy as np
import fasttext
from split_utils import replace_special_chars
import pandas as pd

def min_euclidean_kdtree(points):
    tree = KDTree(points)
    distances, _ = tree.query(points, k=2)  # k=2 because the first result is the point itself
    return np.min(distances[:, 1])  # Ignore the zero distance to itself

def check_incongruence(language, vectors_dict):
    print(f"Checking incongruence for {language}")
    vectors_and_words = list(vectors_dict.values())
    incongruent_words = []
    num_incongruent = 0
    lexical_words = vectors_dict.keys()
    mfa_words = [s[1] for s in vectors_and_words]
    words = zip(lexical_words, mfa_words)
    for lexical_word, mfa_word in words:
        corrected_mfa_word = mfa_word.lower().replace("'", "")
        corrected_lexical_word = replace_special_chars(
                    lexical_word.lower()
                    ) 
        if corrected_mfa_word != corrected_lexical_word:
            incongruent_words = incongruent_words.append([lexical_word, mfa_word])
            num_incongruent += 1
    incongruent_words_df = pd.DataFrame(incongruent_words, columns=["lexical_word", "mfa_word"])
    incongruent_words_df.to_csv(f"incongruent_words_{language}.csv", index=False)
    print(f"Found {num_incongruent} incongruent words")
    return num_incongruent
            

def get_smallest_vector(language,vectors_dict):
   
    print(f"Loaded {len(vectors_dict)} vectors")
  
    vectors_and_words = list(vectors_dict.values())
    vectors = [s[0] for s in vectors_and_words]
    vectors = np.array(vectors)
    print(f"Loaded {vectors.shape} vectors")
    assert vectors.shape[1] == 300
    min_distance = min_euclidean_kdtree(vectors)
    print(f"The smallest distance between vectors in {language} is {min_distance}")
    pickle.dump(min_distance, open(f"min_distance_{language}.pkl", "wb"))
    return min_distance

def get_smallest_mfa_vector(language, vectors_dict):
    model = fasttext.load_model(f"cc.{language}.300.bin")
    vectors_and_words = list(vectors_dict.values())
    mfa_words= [s[1] for s in vectors_and_words]
    set_mfa_words = set(mfa_words)
    vectors = [model.get_word_vector(word) for word in set_mfa_words]
    vectors = np.array(vectors)
    print(f"Loaded {vectors.shape} vectors")
    assert vectors.shape[1] == 300

    min_distance = min_euclidean_kdtree(vectors)
    print(f"The smallest distance between vectors for mfa words in {language} is {min_distance}")
    pickle.dump(min_distance, open(f"mfa_min_distance_{language}.pkl", "wb"))
    return min_distance

if __name__ == "__main__":
    parser =argparse.ArgumentParser(
        description="get the shortest fasttext vector distance"
    )
    parser.add_argument(
        "--language",
        type=str,
        help="The language of the word vectors",
        default="en",
    )
    parser.add_argument("--mfa", action="store_true", help="Use the MFA words")
    parser.add_argument("--check_incongruence", action="store_true", help="Check if capitalization matters")
    args = parser.parse_args()
    vectors_dict = pickle.load(open(f"vectors_{args.language}.pkl", "rb"))
    capitalization_matters = False
    if args.language == "en":
      capitalization_matters =  not np.array_equal(vectors_dict["the"][0], vectors_dict["The"][0])
    if args.language == "de":
        capitalization_matters =   not np.array_equal(vectors_dict["der"][0], vectors_dict["Der"][0])
    if capitalization_matters:
        print(f"Capitalization matters for {args.language}")
    else:
        print(f"Capitalization does not matter for {args.language} or this check is not implemented")
    if args.mfa:
        get_smallest_mfa_vector(args.language, vectors_dict)
    else:
        get_smallest_vector(args.language, vectors_dict)

    if args.check_incongruence:
        check_incongruence(args.language, vectors_dict)