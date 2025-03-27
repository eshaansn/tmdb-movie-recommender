import streamlit as st
import  pickle
import  pandas as pd
import  requests
from omegaconf import OmegaConf
from dotenv import load_dotenv
import os

load_dotenv()
api_key = os.getenv("TMDB_API_KEY")


def fetch_poster(movie_id):
    api_key = '<API_KEY>'
    response = requests.get(f"https://api.themoviedb.org/3/movie/{movie_id}?api_key={api_key}")
    data = response.json()
    print(data)
    # if not data['success']:
    #     return -1
    # else:
    poster_path = data['poster_path']
    full_path = "https://image.tmdb.org/t/p/w500" + poster_path
    return full_path



def recommend(option, n, df, similarity):
    ix = df[df['title'] == option].index[0]
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


movies = pickle.load(open('data/movies.pkl', 'rb'))
movies = pd.DataFrame(movies)

similarity = pickle.load(open('data/similarity.pkl', 'rb'))

list_of_movies = movies.title.values

if __name__ == '__main__':
    st.title('Movie Recommender System')

    option = st.selectbox(
        'List of movies present in our database...',
        list_of_movies)

    n=5
    if st.button('Recommend'):
        recommend(option, n, movies, similarity)