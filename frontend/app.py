import streamlit as st
import pickle
import requests
import os
from sklearn.metrics.pairwise import cosine_similarity


# ---------------- PATH FIX ---------------- #
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

movies_path = os.path.join(BASE_DIR, "artifacts", "movies.pkl")
vectors_path = os.path.join(BASE_DIR, "artifacts", "vectors.pkl")


# ---------------- LOAD FILES ---------------- #
movies = pickle.load(open(movies_path, "rb"))
vectors = pickle.load(open(vectors_path, "rb"))


# ---------------- SAFE FETCH POSTER ---------------- #
def fetch_poster(movie_id):
    url = f"https://api.themoviedb.org/3/movie/{movie_id}?api_key=0662fb540ec5f1f046bb61cb6420c11d&language=en-US"

    try:
        response = requests.get(url, timeout=5)

        if response.status_code != 200:
            return "https://via.placeholder.com/500x750?text=No+Image"

        data = response.json()
        poster_path = data.get("poster_path")

        if poster_path:
            return "https://image.tmdb.org/t/p/w500/" + poster_path
        else:
            return "https://via.placeholder.com/500x750?text=No+Image"

    except requests.exceptions.RequestException:
        return "https://via.placeholder.com/500x750?text=No+Image"


# ---------------- RECOMMEND FUNCTION ---------------- #
def recommend(movie):
    movie_row = movies[movies['title'] == movie]

    if movie_row.empty:
        return [], []

    index = movie_row.index[0]

    distances = cosine_similarity(vectors[index], vectors).flatten()

    movie_list = sorted(
        list(enumerate(distances)),
        reverse=True,
        key=lambda x: x[1]
    )[1:6]

    recommended_names = []
    recommended_posters = []

    for i in movie_list:
        movie_id = movies.iloc[i[0]].movie_id
        recommended_names.append(movies.iloc[i[0]].title)
        recommended_posters.append(fetch_poster(movie_id))

    return recommended_names, recommended_posters


# ---------------- STREAMLIT UI ---------------- #
st.title("🎬 Movie Recommender System")

movie_list = movies["title"].values

selected_movie = st.selectbox(
    "Select a movie",
    movie_list
)

if st.button("Show Recommendation"):

    names, posters = recommend(selected_movie)

    if not names:
        st.warning("Movie not found")
    else:
        cols = st.columns(5)

        for i in range(5):
            with cols[i]:
                st.text(names[i])
                st.image(posters[i])