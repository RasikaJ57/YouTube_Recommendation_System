import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(
    page_title="YouTube Recommendation System",
    page_icon="▶️"
)

st.title("🎥 YouTube Recommendation System")
st.write("Select a video to get similar video recommendations.")

# Load dataset
@st.cache_data
def load_data():
    df = pd.read_csv("cleaned_dataset.csv")
    df["clean_text"] = df["clean_text"].fillna("")
    return df

try:
    df = load_data()

    # TF-IDF
    tfidf = TfidfVectorizer()
    tfidf_matrix = tfidf.fit_transform(df["clean_text"])

    # Similarity
    similarity = cosine_similarity(tfidf_matrix)

    # Video selection
    video_list = df["title"].tolist()

    selected_video = st.selectbox(
        "Select Video",
        video_list
    )

    if st.button("Recommend Videos"):

        index = video_list.index(selected_video)

        scores = list(enumerate(similarity[index]))

        scores = sorted(
            scores,
            key=lambda x: x[1],
            reverse=True
        )

        st.subheader("Recommended Videos")

        count = 0

        for i, score in scores:
            if i != index:
                st.write("▶️", df.iloc[i]["title"])
                count += 1

            if count == 5:
                break

except Exception as e:
    st.error("Error loading dataset")
    st.write(e)
