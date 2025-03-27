import streamlit as st
import  pickle
import  pandas as pd
import  requests
from omegaconf import OmegaConf
from dotenv import load_dotenv
import os
import hydra


def fetch_poster(movie_id):
    load_dotenv()
    api_key = os.getenv("TMDB_API_KEY")
    response = requests.get(f"https://api.themoviedb.org/3/movie/{movie_id}?api_key={api_key}")
    data = response.json()
    # print(data)
    # if not data['success']:
    #     return -1
    # else:
    poster_path = data['poster_path']
    full_path = "https://image.tmdb.org/t/p/w500" + poster_path
    return full_path



def recommend(config, option, n, df, db=None):
    
    if config.app.method == 'use_llm_embeddings':

        ix = df[df['title'] == option].index[0]

        j = db.similarity_search_with_score(query=df.iloc[ix, -1], k=n+1)
        k = pd.DataFrame(j)
        k.iloc[:, 0] = k.iloc[:, 0].apply(lambda x: int(str(x).split("=")[1].split()[0].strip("'")))
        list_of_recommendations = k.set_index(keys=0).T.iloc[:, 1:].to_dict(orient='records')[0]

        counter = 1
        top_n = []
        top_n_posters = []
        for movie_idx in list_of_recommendations.keys():
            if counter < n + 1:
                counter += 1
                top_n.append(df[df['id'] == movie_idx].title.values[0])
                top_n_posters.append(fetch_poster(df[df['id'] == movie_idx].id.values[0]))
            else:
                break
    
    if config.app.method == 'create_vectors':
        ix = df[df['title'] == option].index[0]
        similarity = pickle.load(open(config.features.similarity_save_path, 'rb'))
        distances = similarity[ix]
        list_of_recommendations = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:]

        counter = 1
        top_n = []
        top_n_posters = []
        for movie_idx in list_of_recommendations:
            if counter < n + 1:
                counter += 1
                top_n.append(df.iloc[movie_idx[0]].title)
                top_n_posters.append(fetch_poster(df.iloc[movie_idx[0]].id))
            else:
                break
    
    
        
    st.write(f'Here are {n} movies that are most similar to {option}')
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.text(top_n[0])
        st.image(top_n_posters[0])
    with col2:
        st.text(top_n[1])
        st.image(top_n_posters[1])
    with col3:
        st.text(top_n[2])
        st.image(top_n_posters[2])
    with col4:
        st.text(top_n[3])
        st.image(top_n_posters[3])
    with col5:
        st.text(top_n[4])
        st.image(top_n_posters[4])

def main(config):

    if config.app.method == 'use_llm_embeddings':

        from langchain_chroma  import Chroma
        from langchain_huggingface import HuggingFaceEmbeddings

        persist_dir = config.features.documentstore 
        if os.path.exists(persist_dir) and os.listdir(persist_dir): 
            embedding_model = HuggingFaceEmbeddings(model_name=config.features.model_name)
            print("Loading existing Chroma database...")
            db_movies = Chroma(
                collection_name='TMDB5000',
                persist_directory=persist_dir,
                embedding_function=embedding_model,
            )

    movies = pickle.load(open(config.data.pr_data_save_path, 'rb'))
    movies = pd.DataFrame(movies)
    list_of_movies = movies.title.values

    st.title('Movie Recommender System')

    option = st.selectbox(
        'List of movies present in our database...',
        list_of_movies)

    n=5
    if st.button('Recommend'):
        recommend(config, option, n, movies, db_movies)

if __name__ == '__main__':
    config = OmegaConf.load("./params.yaml")
    main(config)