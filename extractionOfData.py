import pandas as pd
import numpy as np
import pickle
from sklearn.decomposition import NMF
from sklearn.metrics.pairwise import cosine_similarity

# ----- Step 1. Load the Data -----
# Adjust these paths to where you saved the Goodbooks-10K CSV files.
ratings_path = 'ratings.csv'
books_path = 'books.csv'

# Load ratings; assume columns: user_id, book_id, rating, timestamp
ratings_df = pd.read_csv(ratings_path)
# Load books; assume columns include: book_id, title, authors, etc.
books_df = pd.read_csv(books_path)

# Create mapping dictionaries for books:
#   book_id_to_title: maps book_id to its title.
book_id_to_title = pd.Series(books_df.title.values, index=books_df.book_id).to_dict()

ratings_df = ratings_df.groupby(['user_id', 'book_id'])['rating'].mean().reset_index()
# ----- Step 2. Pivot the Ratings -----
# Create a user-book rating matrix.
R = ratings_df.pivot(index='user_id', columns='book_id', values='rating')
# Fill missing ratings with zeros (for simplicity)
R_filled = R.fillna(0)

# Create dictionaries to map raw ids to matrix indices and vice versa.
user_ids = R_filled.index.tolist()
book_ids = R_filled.columns.tolist()

user_to_index = {uid: idx for idx, uid in enumerate(user_ids)}
book_to_index = {bid: idx for idx, bid in enumerate(book_ids)}
index_to_book = {idx: bid for bid, idx in book_to_index.items()}

# Convert the pivoted DataFrame to a NumPy array.
R_matrix = R_filled.values  # shape: (n_users, n_books)

# ----- Step 3. Apply Matrix Factorization with NMF -----
n_components = 30  # number of latent factors; adjust as needed
nmf_model = NMF(n_components=n_components, init='random', random_state=42, max_iter=200)
W = nmf_model.fit_transform(R_matrix)  # shape: (n_users, n_components)
H = nmf_model.components_                # shape: (n_components, n_books)
# Note: The approximated ratings matrix is W @ H.

# ----- Save the model and mappings -----
model_data = {
    'W': W,                    # User latent factors
    'H': H,                    # Item latent factors (each column corresponds to a book)
    'book_to_index': book_to_index,
    'index_to_book': index_to_book,
    'book_id_to_title': book_id_to_title
}

with open('nmf_model.pkl', 'wb') as f:
    pickle.dump(model_data, f)

print("Model training complete and saved as 'nmf_model.pkl'.")
