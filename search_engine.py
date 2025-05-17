#--------------final update------------
import pickle
import numpy as np
import pandas as pd
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

model = SentenceTransformer("all-MiniLM-L6-v2")

def load_data(path="cleaned_laptops.pkl"):
    with open(path, "rb") as f:
        df = pickle.load(f)

    # Extract details dict column into columns if present (just in case)
    if 'details' in df.columns:
        details_df = df['details'].apply(lambda x: pd.Series(x) if isinstance(x, dict) else pd.Series())
        df = pd.concat([df.drop(columns=['details']), details_df], axis=1)
    return df

df = load_data()

def bm25_search(query):
    corpus = df['clean_text'].apply(lambda x: x.split()).tolist()
    bm25 = BM25Okapi(corpus)
    query_tokens = query.lower().split()
    return bm25.get_scores(query_tokens)

def cosine_search(query, embedding_matrix):
    query_embedding = model.encode(query, show_progress_bar=False).reshape(1, -1)
    return cosine_similarity(query_embedding, embedding_matrix)[0]

def hybrid_search(query, top_k=5, bm25_weight=0.5, cosine_weight=0.5):
    bm25_scores = bm25_search(query)
    embedding_matrix = np.vstack(df['embedding'].values).astype('float32')
    cosine_scores = cosine_search(query, embedding_matrix)

    bm25_scores = np.array(bm25_scores) / np.linalg.norm(bm25_scores)
    cosine_scores = np.array(cosine_scores) / np.linalg.norm(cosine_scores)

    combined_scores = bm25_weight * bm25_scores + cosine_weight * cosine_scores
    top_k_indices = combined_scores.argsort()[-top_k:][::-1]

    return df.iloc[top_k_indices][['name_and_feature', 'average_rating', 'clean_text']].assign(
        score=combined_scores[top_k_indices]
    )
