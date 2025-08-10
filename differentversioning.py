from flask import Flask, request, jsonify
import numpy as np
import pickle
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

from flask_cors import CORS
CORS(app)

# Load the NMF model and mappings
with open('nmf_model.pkl', 'rb') as f:
    model_data = pickle.load(f)

# Extract components from the model data
H = model_data['H']  # Item latent factors (shape: [n_components, n_books])
book_to_index = model_data['book_to_index']
index_to_book = model_data['index_to_book']
book_id_to_title = model_data['book_id_to_title']

# Transpose H to get book embeddings (shape: [n_books, n_components])
book_embeddings = H.T


def get_similar_books(book_id, top_n=10):
    """Get similar books using cosine similarity on NMF components"""
    if book_id not in book_to_index:
        return None

    # Get the index and embedding for the target book
    book_idx = book_to_index[book_id]
    target_embedding = book_embeddings[book_idx].reshape(1, -1)

    # Calculate cosine similarities
    similarities = cosine_similarity(target_embedding, book_embeddings)[0]

    # Exclude the original book and get top matches
    similar_indices = np.argsort(similarities)[::-1][1:top_n + 1]

    recommendations = []
    for idx in similar_indices:
        similar_book_id = index_to_book[idx]
        recommendations.append({
            "book_id": similar_book_id,
            "title": book_id_to_title.get(similar_book_id, "Unknown Title"),
            "similarity": float(similarities[idx])
        })

    return recommendations


@app.route('/recommend', methods=['POST'])
def recommend():
    data = request.get_json()
    book_id = data.get('book_id')

    if not book_id:
        return jsonify({"error": "book_id is required in the JSON payload."}), 400

    try:
        book_id = int(book_id)  # Ensure book_id is integer
    except ValueError:
        return jsonify({"error": "book_id must be an integer."}), 400

    recommendations = get_similar_books(book_id)

    if recommendations is None:
        return jsonify({"error": "Book ID not found."}), 404

    return jsonify({
        "requested_book": {
            "book_id": book_id,
            "title": book_id_to_title.get(book_id, "Unknown Title")
        },
        "recommendations": recommendations
    })


if __name__ == '__main__':
    app.run(debug=True)