import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st

# Load the dataset
data = pd.read_csv('final.csv')

# Preprocess vitals data: Normalize or standardize
vitals_columns = data.iloc[:,3:].columns.tolist()
scaler = StandardScaler()
data[vitals_columns] = scaler.fit_transform(data[vitals_columns])

# Aggregate the vitals data and ratings for each movie
movie_profiles = data.groupby(['title', 'genres']).mean().reset_index()

# Function to take user vitals input and recommend movies
def recommend_movies_based_on_vitals(hr, resp, spo2, temp, top_n=10):
    # Normalize the user input vitals
    user_vitals = scaler.transform([[hr, resp, spo2, temp]])
    
    # Compute the cosine similarity between the user profile and movie profiles
    movie_vitals = movie_profiles[vitals_columns].values
    similarity_scores = cosine_similarity(movie_vitals, user_vitals).flatten()

    # Add similarity scores to the movie profiles
    movie_profiles['similarity_score'] = similarity_scores

    # Recommend top movies based on similarity scores
    recommended_movies = movie_profiles.sort_values(by='similarity_score', ascending=False).head(top_n)
    return recommended_movies[['title', 'genres', 'similarity_score']]

# Function to recommend movies based on an input movie
def recommend_movies_based_on_movie(input_movie, top_n=10):
    # Find the movie profile for the input movie
    movie_profile = movie_profiles[movie_profiles['title'] == input_movie]
    if movie_profile.empty:
        return pd.DataFrame()  # Return empty DataFrame if movie not found

    # Compute the cosine similarity between the input movie and all other movies
    movie_vitals = movie_profiles[vitals_columns].values
    input_movie_vitals = movie_profile[vitals_columns].values
    similarity_scores = cosine_similarity(movie_vitals, input_movie_vitals).flatten()

    # Add similarity scores to the movie profiles
    movie_profiles['similarity_score'] = similarity_scores

    # Recommend top movies based on similarity scores
    recommended_movies = movie_profiles.sort_values(by='similarity_score', ascending=False).head(top_n + 1)
    recommended_movies = recommended_movies[recommended_movies['title'] != input_movie]  # Exclude the input movie
    return recommended_movies[['title', 'genres', 'similarity_score']]

# Streamlit application
st.title("Vitals-Based and Movie-Based Movie Recommendation System")

# Vitals-based recommendation
st.header("Enter your vitals to get movie recommendations:")
hr = st.number_input("Heart Rate (BPM):", min_value=0, max_value=200, value=70)
resp = st.number_input("Respiratory Rate (BPM):", min_value=0, max_value=50, value=20)
spo2 = st.number_input("SpO2 (%):", min_value=0, max_value=100, value=95)
temp = st.number_input("Temperature (*C):", min_value=30.0, max_value=45.0, value=36.5)

if st.button("Get Recommendations Based on Vitals"):
    recommended_movies = recommend_movies_based_on_vitals(hr, resp, spo2, temp)
    st.subheader("Recommended Movies Based on Vitals:")
    for index, row in recommended_movies.iterrows():
        st.write(f"{row['title']} ({row['genres']}) - Similarity Score: {row['similarity_score']:.4f}")

# Movie-based recommendation
st.header("Enter a movie name to get next movie recommendations:")
input_movie = st.text_input("Movie Name:")

if st.button("Get Recommendations Based on Movie"):
    recommended_movies = recommend_movies_based_on_movie(input_movie)
    if recommended_movies.empty:
        st.write("Movie not found. Please check the movie name and try again.")
    else:
        st.subheader("Recommended Movies Based on Movie:")
        for index, row in recommended_movies.iterrows():
            st.write(f"{row['title']} ({row['genres']}) - Similarity Score: {row['similarity_score']:.4f}")
