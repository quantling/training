
data_path = "../../../../mnt/Restricted/Corpora/CommonVoiceVTL/corpus_as_df_mp_folder_en"
import logging
import os
logging.basicConfig(level=logging.INFO)


sorted = sorted(os.listdir(data_path)) #sorting the files for better debugging
for files in sorted:
    logging.info(f"Processing {files}")
import psutil

from collections import Counter
import os
import pandas as pd
import pickle
words = Counter()
i = 0
word_list = []
for files in sorted:
        data = None
        if i % 10 == 0:
            pickle.dump(words,open(f"lexical_words_{i}.pkl","wb"))
        if files.endswith(".pkl"):
            data = pd.read_pickle(os.path.join(data_path,files))
            if isinstance(data, pd.DataFrame):
                print(data.columns)
                for word in list(data["lexical_word"]):
                    words[word] += 1
                    word_list.append(word)
            else:
                print(f"Data is not a DataFrame: {files}")
        []
        i += 1
        print(f"Processed {i} files")
        print(files)
        print(f"Memory usage: {psutil.virtual_memory().percent}%")
        del data
print("Done")

words
pickle.dump(words,open("lexical_words.pkl","wb"))
pickle.dump(word_list,open("lexical_word_list.pkl","wb")) # I guess an ordered list of words is also useful