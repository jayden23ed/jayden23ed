import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

# Sample anime dataset (you should replace this with your own data)
data = {
    'Title': ['Anime A', 'Anime B', 'Anime C', 'Anime D'],
    'Genres': ['Action, Adventure', 'Comedy, Romance', 'Action, Fantasy', 'Comedy, Slice of Life'],
    'Description': ['Description A', 'Description B', 'Description C', 'Description D'],
}

df = pd.DataFrame(data)

# TF-IDF vectorization on anime descriptions
tfidf_vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf_vectorizer.fit_transform(df['Description'])

# Calculate cosine similarity
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

# Function to recommend anime based on user preferences
def recommend_anime(title, cosine_sim=cosine_sim):
    idx = df[df['Title'] == title].index[0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:11]  # Get the top 10 most similar anime (excluding itself)
    anime_indices = [i[0] for i in sim_scores]
    return df['Title'].iloc[anime_indices]

# Example: Recommend anime similar to 'Anime A'
recommended_anime = recommend_anime('Anime A')
print(recommended_anime)

