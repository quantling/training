import argparse
import os
import matplotlib.pyplot as plt
from collections import Counter
import numpy as np
import pickle

def plot_zipf(language):
    # Example word frequency counter
    word_counts = pickle.load(open(f"word_counter_{language}.pkl", "rb"))
    # Sort words by frequency
    sorted_counts = sorted(word_counts.values(), reverse=True)

    # Rank (1-based)
    ranks = np.arange(1, len(sorted_counts) + 1)

    # Plot
    plt.figure(figsize=(8, 6))
    plt.loglog(ranks, sorted_counts, marker="o", linestyle="None", label="Word Frequencies")

    plt.xlabel("Rank")
    plt.ylabel("Frequency")
    plt.title("Zipf's Law Plot")
    plt.legend()
    plt.grid(True, which="both", linestyle="--")

    plt.savefig(f"zipf_{language}.png")
    

def get_info(language):
    word_counter = pickle.load(open(f"word_counter_{language}.pkl", "rb"))
    print(f"Number of unique words: {len(word_counter)}")
    print(f"Most common words: {word_counter.most_common(10)}")
    print(f"Least common words: {word_counter.most_common()[:-10:-1]}")
    print(f"Total number of words: {sum(word_counter.values())}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot Zipf's Law")
    parser.add_argument("--language", type=str, help="language of the counter", default="en")
    parser.add_argument("--get_info", action="store_true", help="get info about the word counter")
    args = parser.parse_args()

    if args.get_info:
        get_info(args.language)
    
    plot_zipf(args.language)