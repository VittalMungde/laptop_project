# # # # streamlit_app.py
# # #
# # # import streamlit as st
# # # import pickle
# # # import nltk
# # # import re
# # # import numpy as np
# # # import pandas as pd
# # # from sklearn.metrics.pairwise import cosine_similarity
# # # from sentence_transformers import SentenceTransformer
# # #
# # # from preprocessing import clean_text  # Make sure this exists and works properly
# # #
# # # # MUST be first Streamlit command
# # # st.set_page_config(page_title="Laptop Recommender + Chatbot", layout="wide")
# # #
# # # nltk.download("punkt")
# # #
# # # # Load preprocessed data
# # # with open("laptop_data_2000.pkl", "rb") as f:
# # #     df = pickle.load(f)
# # #
# # # # Load embedding model
# # # try:
# # #     model = SentenceTransformer("all-MiniLM-L6-v2")
# # # except Exception as e:
# # #     st.error(" Error loading embedding model.")
# # #     st.stop()
# # #
# # # # 🛠 Extract missing spec columns from 'details' column if needed
# # # required_columns = ["Processor", "RAM", "Storage", "GPU", "Display"]
# # # if not all(col in df.columns for col in required_columns):
# # #     if "details" in df.columns:
# # #         details_df = df["details"].apply(lambda d: pd.Series(d) if isinstance(d, dict) else pd.Series({}))
# # #         for col in required_columns:
# # #             if col not in df.columns and col in details_df.columns:
# # #                 df[col] = details_df[col]
# # #
# # # # Check again if any required columns are still missing
# # # still_missing = [col for col in required_columns if col not in df.columns]
# # #
# # # # Title
# # # st.title(" Laptop Recommender + Chatbot Assistant")
# # #
# # # # Session state
# # # if "top_5_laptops" not in st.session_state:
# # #     st.session_state.top_5_laptops = None
# # # if "messages" not in st.session_state:
# # #     st.session_state.messages = []
# # #
# # # #  Hybrid Search Function
# # # def hybrid_search(query, n=5):
# # #     query_vec = model.encode(clean_text(query))
# # #     similarities = cosine_similarity([query_vec], list(df['embedding'].values)).flatten()
# # #     top_indices = similarities.argsort()[-n:][::-1]
# # #     return df.iloc[top_indices]
# # #
# # # # User query input
# # # query = st.text_input("Enter your laptop requirement query:")
# # #
# # # if query:
# # #     top_5 = hybrid_search(query)
# # #     st.session_state.top_5_laptops = top_5
# # #
# # #     st.write(" Top 5 Recommended Laptops:")
# # #
# # #     if still_missing:
# # #         st.error(f"Missing columns in data: {still_missing}")
# # #     else:
# # #         st.dataframe(top_5[["title", "Processor", "RAM", "Storage", "GPU", "Display", "average_rating"]].rename(columns={
# # #             "title": "Laptop Name",
# # #             "average_rating": "Rating"
# # #         }), use_container_width=True)
# # #
# # # #  Chatbot Section
# # # if st.session_state.top_5_laptops is not None and not still_missing:
# # #     st.markdown("### Chat with the Assistant (refine within top 5)")
# # #
# # #     chat_query = st.chat_input("Ask about specific features (e.g., best one with SSD and 16GB RAM)?")
# # #
# # #     if chat_query:
# # #         st.session_state.messages.append({"role": "user", "content": chat_query})
# # #         with st.chat_message("user"):
# # #             st.markdown(chat_query)
# # #
# # #         def score_laptop(text, keywords):
# # #             return sum(1 for k in keywords if k in text)
# # #
# # #         keywords = clean_text(chat_query).split()
# # #         filtered = st.session_state.top_5_laptops.copy()
# # #         filtered["match_score"] = filtered["clean_text"].apply(lambda x: score_laptop(x, keywords))
# # #
# # #         best_match = filtered.sort_values("match_score", ascending=False).iloc[0]
# # #
# # #         response = f"Based on your query, the best match is:\n\n**{best_match['name_and_feature']}**\n\n Rating: {best_match['average_rating']}"
# # #
# # #         st.session_state.messages.append({"role": "assistant", "content": response})
# # #         with st.chat_message("assistant"):
# # #             st.markdown(response)
# #
# #
# #
# # #------------final code------------
# # import streamlit as st
# # import pickle
# # import nltk
# # import numpy as np
# # import pandas as pd
# # from sklearn.metrics.pairwise import cosine_similarity
# # from sentence_transformers import SentenceTransformer
# # from preprocessing import clean_text
# #
# # st.set_page_config(page_title="Laptop Recommender + Chatbot", layout="wide")
# # nltk.download("punkt")
# #
# # with open("cleaned_laptops.pkl", "rb") as f:
# #     df = pickle.load(f)
# #
# # try:
# #     model = SentenceTransformer("all-MiniLM-L6-v2")
# # except Exception as e:
# #     st.error(" Error loading embedding model.")
# #     st.stop()
# #
# # display_columns = ['title', 'average_rating', 'Processor', 'RAM', 'Storage', 'GPU', 'Display', 'features', 'description']
# #
# # st.title(" Laptop Recommender + Chatbot Assistant")
# #
# # if "top_5_laptops" not in st.session_state:
# #     st.session_state.top_5_laptops = None
# # if "messages" not in st.session_state:
# #     st.session_state.messages = []
# #
# # def hybrid_search(query, n=5):
# #     query_vec = model.encode(clean_text(query))
# #     similarities = cosine_similarity([query_vec], list(df['embedding'].values)).flatten()
# #     top_indices = similarities.argsort()[-n:][::-1]
# #     return df.iloc[top_indices]
# #
# # query = st.text_input("Enter your laptop requirement query:")
# #
# # if query:
# #     top_5 = hybrid_search(query)
# #     st.session_state.top_5_laptops = top_5
# #     st.write(" Top 5 Recommended Laptops:")
# #     available = [col for col in display_columns if col in top_5.columns]
# #     if not available:
# #         st.warning(" None of the desired display columns are available.")
# #         st.dataframe(top_5, use_container_width=True)
# #     else:
# #         st.dataframe(top_5[available], use_container_width=True)
# #
# # if st.session_state.top_5_laptops is not None:
# #     st.markdown("###  Chat with the Assistant (refine within top 5)")
# #     chat_query = st.chat_input("Ask about specific features (e.g., best one with SSD and 16GB RAM)?")
# #
# #     if chat_query:
# #         st.session_state.messages.append({"role": "user", "content": chat_query})
# #         with st.chat_message("user"):
# #             st.markdown(chat_query)
# #
# #         def score_laptop(text, keywords):
# #             return sum(1 for k in keywords if k in text)
# #
# #         keywords = clean_text(chat_query).split()
# #         filtered = st.session_state.top_5_laptops.copy()
# #
# #         if "clean_text" not in filtered.columns:
# #             filtered["clean_text"] = filtered.astype(str).agg(" ".join, axis=1).apply(clean_text)
# #
# #         filtered["match_score"] = filtered["clean_text"].apply(lambda x: score_laptop(x, keywords))
# #         best_match = filtered.sort_values("match_score", ascending=False).iloc[0]
# #
# #         name_feature = best_match.get('title') or "Laptop"
# #         response = f" Based on your query, the best match is:\n\n**{name_feature}**\n\n Rating: {best_match.get('average_rating', 'N/A')}"
# #         st.session_state.messages.append({"role": "assistant", "content": response})
# #
# #         with st.chat_message("assistant"):
# #             st.markdown(response)
#
#
# #-----------------update final ------------
# import streamlit as st
# import pickle
# import nltk
# import re
# import numpy as np
# import pandas as pd
# from sklearn.metrics.pairwise import cosine_similarity
# from sentence_transformers import SentenceTransformer
#
# # Import clean_text from preprocessing or redefine here
# def clean_text(text):
#     text = str(text).lower()
#     text = re.sub(r'[^\w\s]', '', text)
#     text = re.sub(r'\s+', ' ', text).strip()
#     return text
#
# # Streamlit configuration
# st.set_page_config(page_title="Laptop Recommender + Chatbot", layout="wide")
#
# nltk.download("punkt")
#
# # Load preprocessed dataset
# with open("cleaned_laptops.pkl", "rb") as f:
#     df = pickle.load(f)
#
# # Extract details dict column into separate columns if present (just in case)
# if 'details' in df.columns:
#     details_df = df['details'].apply(lambda x: pd.Series(x) if isinstance(x, dict) else pd.Series())
#     df = pd.concat([df.drop(columns=['details']), details_df], axis=1)
#
# st.write("Available columns in the dataset:")
# st.write(df.columns.tolist())
#
# # Load embedding model
# try:
#     model = SentenceTransformer("all-MiniLM-L6-v2")
# except Exception as e:
#     st.error(" Error loading embedding model.")
#     st.stop()
#
# display_columns = [
#     'link', 'name', 'user rating', 'Price', 'Color', 'Type', 'Suitable For', 'Power Supply',
#     'Processor Brand', 'Processor Name', 'Processor Generation', 'SSD Capacity', 'RAM',
#     'Expandable Memory', 'Graphic Processor', 'Operating System', 'HDMI Port', 'Touchscreen',
#     'Screen Size', 'Weight', 'Backlit Keyboard', 'Warranty Summary', 'Battery Backup'
# ]
#
# # Filter to only available display columns
# available_display_cols = [col for col in display_columns if col in df.columns]
#
# st.title(" Laptop Recommender + Chatbot Assistant")
#
# # Session state
# if "top_5_laptops" not in st.session_state:
#     st.session_state.top_5_laptops = None
# if "messages" not in st.session_state:
#     st.session_state.messages = []
#
# def hybrid_search(query, n=5):
#     query_vec = model.encode(clean_text(query))
#     similarities = cosine_similarity([query_vec], list(df['embedding'].values)).flatten()
#     top_indices = similarities.argsort()[-n:][::-1]
#     return df.iloc[top_indices]
#
# query = st.text_input("Enter your laptop requirement query:")
#
# if query:
#     top_5 = hybrid_search(query)
#     st.session_state.top_5_laptops = top_5
#     st.write(" Top 5 Recommended Laptops:")
#
#     missing_columns = [col for col in display_columns if col not in df.columns]
#     if missing_columns:
#         st.warning(f"These columns are missing and will not be shown: {missing_columns}")
#
#     if not available_display_cols:
#         st.warning(" None of the desired display columns are available.")
#         st.dataframe(top_5, use_container_width=True)
#     else:
#         st.dataframe(top_5[available_display_cols], use_container_width=True)
#
# if st.session_state.top_5_laptops is not None and len(missing_columns) < len(display_columns):
#     st.markdown("### Chat with the Assistant (refine within top 5)")
#     chat_query = st.chat_input("Ask about specific features (e.g., best one with SSD and 16GB RAM)?")
#
#     if chat_query:
#         st.session_state.messages.append({"role": "user", "content": chat_query})
#         with st.chat_message("user"):
#             st.markdown(chat_query)
#
#         def score_laptop(text, keywords):
#             return sum(1 for k in keywords if k in text)
#
#         keywords = clean_text(chat_query).split()
#         filtered = st.session_state.top_5_laptops.copy()
#
#         if "clean_text" not in filtered.columns:
#             filtered["clean_text"] = filtered.astype(str).agg(" ".join, axis=1).apply(clean_text)
#
#         filtered["match_score"] = filtered["clean_text"].apply(lambda x: score_laptop(x, keywords))
#         best_match = filtered.sort_values("match_score", ascending=False).iloc[0]
#
#         name_feature = best_match.get('name_and_feature') or best_match.get('name') or "Laptop"
#
#         response = f" Based on your query, the best match is:\n\n**{name_feature}**\n\n Rating: {best_match.get('user rating', 'N/A')}"
#         st.session_state.messages.append({"role": "assistant", "content": response})
#
#         with st.chat_message("assistant"):
#             st.markdown(response)



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
