import streamlit as st
import pickle
import re
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

# Function to clean text for embedding and search
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# Streamlit page config
st.set_page_config(page_title="Laptop Recommender + Chatbot", layout="wide")

# Load preprocessed dataset
@st.cache_data(show_spinner=True)
def load_data():
    with open("cleaned_laptops.pkl", "rb") as f:
        df_loaded = pickle.load(f)
    # Expand 'details' column if present
    if 'details' in df_loaded.columns:
        details_df = df_loaded['details'].apply(lambda x: pd.Series(x) if isinstance(x, dict) else pd.Series())
        df_loaded = pd.concat([df_loaded.drop(columns=['details']), details_df], axis=1)
    return df_loaded

df = load_data()

# Load embedding model
@st.cache_resource(show_spinner=True)
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

model = load_model()

# Columns to display (filter only those present in df)
display_columns = [
    'link', 'name', 'user rating', 'Price', 'Color', 'Type', 'Suitable For', 'Power Supply',
    'Processor Brand', 'Processor Name', 'Processor Generation', 'SSD Capacity', 'RAM',
    'Expandable Memory', 'Graphic Processor', 'Operating System', 'HDMI Port', 'Touchscreen',
    'Screen Size', 'Weight', 'Backlit Keyboard', 'Warranty Summary', 'Battery Backup'
]

available_display_cols = [col for col in display_columns if col in df.columns]

# Title
st.title(" Laptop Recommender + Chatbot Assistant")

# Session state for top 5 results and chat messages
if "top_5_laptops" not in st.session_state:
    st.session_state.top_5_laptops = None
if "messages" not in st.session_state:
    st.session_state.messages = []

# Search function using cosine similarity on embeddings
def hybrid_search(query, n=5):
    query_vec = model.encode(clean_text(query))
    embeddings = np.vstack(df['embedding'].values)
    similarities = cosine_similarity([query_vec], embeddings).flatten()
    top_indices = similarities.argsort()[-n:][::-1]
    return df.iloc[top_indices]

# Input from user
query = st.text_input("Enter your laptop requirement query:")

if query:
    top_5 = hybrid_search(query)
    st.session_state.top_5_laptops = top_5

    if available_display_cols:
        st.subheader(" Top 5 Recommended Laptops:")
        st.dataframe(top_5[available_display_cols], use_container_width=True)
    else:
        st.warning(" None of the desired display columns are available to display.")
        st.dataframe(top_5, use_container_width=True)

# Chatbot section (optional)
if st.session_state.top_5_laptops is not None and len(st.session_state.top_5_laptops) > 0:

    st.markdown("---")
    st.markdown("### Chat with the Assistant (refine your query based on top 5 laptops)")

    chat_query = st.chat_input("Ask about specific features (e.g., best one with SSD and 16GB RAM)?")

    if chat_query:
        st.session_state.messages.append({"role": "user", "content": chat_query})
        with st.chat_message("user"):
            st.markdown(chat_query)

        # Simple scoring: count how many keywords appear in 'clean_text' column
        keywords = clean_text(chat_query).split()
        filtered = st.session_state.top_5_laptops.copy()

        if "clean_text" not in filtered.columns:
            # fallback: combine columns to text if clean_text missing
            filtered["clean_text"] = filtered.astype(str).agg(" ".join, axis=1).apply(clean_text)

        filtered["match_score"] = filtered["clean_text"].apply(lambda x: sum(k in x for k in keywords))
        best_match = filtered.sort_values("match_score", ascending=False).iloc[0]

        name_feature = best_match.get('name') or best_match.get('name_and_feature') or "Laptop"

        response = (
            f" Based on your query, the best match is:\n\n**{name_feature}**\n\n"
            f" Rating: {best_match.get('user rating', 'N/A')}\n\n"
            f" Price: {best_match.get('Price', 'N/A')}"
        )

        st.session_state.messages.append({"role": "assistant", "content": response})

        with st.chat_message("assistant"):
            st.markdown(response)
