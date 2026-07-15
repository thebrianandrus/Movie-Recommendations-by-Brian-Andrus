import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer, util

st.title("Movie Reccommendations, By Brian Andrus")


@st.cache_resource
def load_model():
    # Small, fast, runs locally - good balance of speed and quality
    return SentenceTransformer("all-MiniLM-L6-v2")


@st.cache_data
def embed_descriptions(descriptions):
    model = load_model()
    return model.encode(descriptions, convert_to_tensor=True)


try:
    df = pd.read_csv("movies.csv")
    df.columns = df.columns.str.strip()

    model = load_model()
    description_embeddings = embed_descriptions(df["description"].tolist())

    query = st.text_input("What kind of movie are you looking for?")

    if st.button("Find Movies"):
        if query:
            query_embedding = model.encode(query, convert_to_tensor=True)
            similarities = util.cos_sim(query_embedding, description_embeddings)[0].tolist()

            df["similarity"] = similarities
            top = df.sort_values("similarity", ascending=False).head(3)

            for _, row in top.iterrows():
                st.subheader(row["title"])
                st.write(row["description"])
                st.markdown(f"[View on Letterboxd]({row['letterboxd_url']})")
                st.write(f"Score: {row['similarity']:.2f}")

except Exception as e:
    st.error(f"Error: {e}")
