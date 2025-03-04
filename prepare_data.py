import pandas as pd
from omegaconf import OmegaConf
import pickle
from nltk.stem.porter import PorterStemmer

def reformat(row):
    import ast
    new_row = []
    for item in ast.literal_eval(row):
        new_row.append(item['name'])

    return new_row

def reformat_cast(row):
    import ast
    new_row = []
    counter = 0
    for item in ast.literal_eval(row):
        if counter != 3:
            new_row.append(item['name'])
            counter+=1
        else:
            break

    return new_row

def get_director(row):
    import ast
    new_row = []
    for item in ast.literal_eval(row):
        if item['job'] == 'Director':
            new_row.append(item['name'])
            break
    
    return new_row

def stem(text):
    new_text = []
    ps = PorterStemmer()

    for word in text.split():
        new_text.append(ps.stem(word))
    
    return " ".join(new_text)
        

def prepare_data(config):
    print("Preparing data...")

    movies = pd.read_csv(config.data.movies_file_path)
    credits = pd.read_csv(config.data.credits_file_path)
    data = movies.merge(credits, on='title')

    df = data[['id', 'title', 'genres', 'keywords', 'overview', 'cast', 'crew']]

    df.dropna(inplace=True)

    df['genres'] = df['genres'].apply(reformat)
    df['keywords'] =  df['keywords'].apply(reformat)
    df['cast'] = df['cast'].apply(reformat_cast)
    df['crew'] = df['crew'].apply(get_director)

    df['overview'] = df['overview'].apply(lambda x:x.split())

    df['genres'] = df['genres'].apply(lambda x: [i.replace(" ", "") for i in x ])
    df['keywords'] = df['keywords'].apply(lambda x: [i.replace(" ", "") for i in x ])

    df['cast'] = df['cast'].apply(lambda x: [i.replace(" ", "") for i in x ])
    df['crew'] = df['crew'].apply(lambda x: [i.replace(" ", "") for i in x ])

    df['tags'] = df['genres'] + df['overview'] + df['keywords'] + df['cast'] + df['crew']
    df['tags'] = df['tags'].apply(lambda x: " ".join(x))
    df['tags'] = df['tags'].apply(lambda x: x.lower())

    prepped_df = df[['id', 'title', 'tags']]

    prepped_df['tags'] = prepped_df['tags'].apply(stem)

    print(prepped_df.head())
    pickle.dump(prepped_df.to_dict(), open(config.data.pr_data_save_path, 'wb'))


if __name__ == '__main__':
    config = OmegaConf.load("./params.yaml")
    prepare_data(config)

