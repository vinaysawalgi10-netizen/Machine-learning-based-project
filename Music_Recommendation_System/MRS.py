import pandas as pd
import nltk
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st

# -------------------------------------------------
# ✅ MUST BE THE FIRST Streamlit COMMAND
# -------------------------------------------------
st.set_page_config(page_title="🎵 Music Recommendation System", layout="wide")

# -------------------------------------------------
# Download NLTK resources
# -------------------------------------------------
nltk.download('punkt', quiet=True)

# -------------------------------------------------
# Load dataset
# -------------------------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv(r"C:\Users\vinay\OneDrive\Desktop\PROJECTS__\songdata.csv")
    df = df.sample(n=5000).drop("link", axis=1).reset_index(drop=True)
    df["text"] = (
        df["text"]
        .str.lower()
        .replace(r"[^\w\s]", " ", regex=True)
        .replace(r"\n", " ", regex=True)
    )
    return df

df = load_data()

# -------------------------------------------------
# Tokenization + stemming
# -------------------------------------------------
ps = PorterStemmer()
def tokenization(txt):
    tokens = word_tokenize(txt)
    stemming = [ps.stem(w) for w in tokens]
    return " ".join(stemming)

df["text"] = df["text"].apply(tokenization)

# -------------------------------------------------
# TF-IDF Vectorization + Similarity Matrix
# -------------------------------------------------
tfid = TfidfVectorizer(stop_words="english")
matrix = tfid.fit_transform(df["text"])
similarity = cosine_similarity(matrix)

# -------------------------------------------------
# Recommendation Function
# -------------------------------------------------
def recommendation(song):
    matches = df[df["song"].str.lower() == song.lower()]
    if matches.empty:
        return []
    idx = matches.index[0]
    distances = sorted(
        list(enumerate(similarity[idx])),
        key=lambda x: x[1],
        reverse=True
    )
    return [df.iloc[i[0]].song for i in distances[1:21]]

# -------------------------------------------------
# Streamlit UI
# -------------------------------------------------
st.title("🎵 Music Recommendation System")

# Song selection dropdown
song_choice = st.selectbox("Select a song:", df["song"].head(50))

if st.button("Recommend"):
    recs = recommendation(song_choice)
    if recs:
        st.subheader(f"Top Recommendations for: {song_choice}")
        cols = st.columns(3)
        for i, rec in enumerate(recs):
            with cols[i % 3]:
                st.image("img.jpg", caption=rec, use_container_width=True)
    else:
        st.error(f"⚠️ '{song_choice}' not found in dataset.")
