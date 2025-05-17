# ---------------updated final code --------------------
import pandas as pd
import re
import pickle
from sentence_transformers import SentenceTransformer

# Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Clean individual value
def clean_value(x):
    if isinstance(x, list):
        return ', '.join(map(str, x))
    elif isinstance(x, dict):
        return ', '.join(map(str, x.values()))
    else:
        return str(x)

# Clean text for embedding
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^\w\s]', '', text)  # remove punctuation
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# Remove ₹ or ? and commas from price and convert to int
def clean_price(price_str):
    if pd.isna(price_str):
        return None
    price_str = re.sub(r'[^\d]', '', str(price_str))  # keep only digits
    return int(price_str) if price_str.isdigit() else None

def preprocess_and_save(file_path, output_path="cleaned_laptops.pkl"):
    print("Reading dataset...")
    df = pd.read_csv(file_path)

    print("Cleaning values...")
    for col in df.columns:
        df[col] = df[col].apply(clean_value)

    # Clean price
    print("Cleaning price...")
    if 'Price' in df.columns:
        df['Cleaned_Price'] = df['Price'].apply(clean_price)

    # Extract details dict column into separate columns (IMPORTANT!)
    if 'details' in df.columns:
        print("Extracting 'details' column into separate columns...")
        details_df = df['details'].apply(lambda x: pd.Series(eval(x)) if isinstance(x, str) else pd.Series(x) if isinstance(x, dict) else pd.Series())
        df = pd.concat([df.drop(columns=['details']), details_df], axis=1)

    # Combine text fields for embedding
    print("Combining relevant fields for embedding...")
    text_columns = ['name', 'Processor Brand', 'Processor Name', 'Processor Generation', 'RAM',
                    'SSD Capacity', 'Graphic Processor', 'Operating System', 'Screen Size',
                    'Weight', 'Type', 'Suitable For', 'Color', 'Warranty Summary', 'Battery Backup']

    # Filter columns actually present
    available_text_cols = [col for col in text_columns if col in df.columns]

    df['combined_text'] = df[available_text_cols].astype(str).apply(lambda row: ' '.join(row), axis=1)
    df["clean_text"] = df["combined_text"].apply(clean_text)

    # Generate embeddings
    print("Generating embeddings...")
    df["embedding"] = df["clean_text"].apply(lambda x: model.encode(x, show_progress_bar=False))

    print("Saving cleaned data to pickle...")
    with open(output_path, "wb") as f:
        pickle.dump(df, f)

    print(f"Done. Saved cleaned data to '{output_path}'")

# Run only if script is called directly
if __name__ == "__main__":
    preprocess_and_save(file_path="laptop_data.csv")

