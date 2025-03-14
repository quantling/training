import pandas as pd


new_data = pd.DataFrame()

new_data['tuple_vector'] = data['vector'].apply(lambda x: tuple(x.tolist()))

new_data['lexical_word'] = data['lexical_word']
new_data['mfa_word'] = data['mfa_word']

unique_data = pd.DataFrame()

new_data = pd.concat((unique_data, new_data), axis=0)

unique_data = new_data.drop_duplicates()