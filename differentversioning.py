from flask import Flask, request, jsonify
import numpy as np
import pickle
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

from flask_cors import CORS
CORS(app)

with open('nmf_model.pkl', 'rb') as f:
    model_data = pickle.load(f)

W = model_data['W']
H = model_data['H']
book_to_index = model_data['book_to_index']
index_to_book = model_data['index_to_book']
book_id_to_title = model_data['book_id_to_title']

book_embeddings = H.T


def get_similar_books_using_cosine_similarity(book_id, top_n=10):
    if book_id not in book_to_index:
        return None

    book_idx = book_to_index[book_id]
    target_embedding = book_embeddings[book_idx].reshape(1, -1)

    similarities = cosine_similarity(target_embedding, book_embeddings)[0]

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

def get_similar_books(book_id, top_n=10):
    if book_id not in book_to_index:
        return None

    book_idx = book_to_index[book_id]

    # Predviđene ocjene svih korisnika za tu knjigu
    predicted_ratings = W @ H[:, book_idx]

    # Prosječna predviđena ocjena po knjizi
    avg_predicted_ratings = H.T @ np.mean(W, axis=0)

    # Sortiramo po najvećoj predviđenoj ocjeni
    target_vec = H[:, book_idx]
    distances = np.linalg.norm(H.T - target_vec, axis=1)
    # recommended_indices = np.argsort(distances)
    recommended_indices = np.argsort(avg_predicted_ratings)[::-1]

    recommendations = []
    count = 0
    for idx in recommended_indices:
        similar_book_id = index_to_book[idx]
        if similar_book_id == book_id:
            continue
        recommendations.append({
            "book_id": similar_book_id,
            "title": book_id_to_title.get(similar_book_id, "Unknown Title"),
            "predicted_score": float(avg_predicted_ratings[idx])
        })
        count += 1
        if count >= top_n:
            break

    return recommendations



@app.route('/recommend', methods=['POST'])
def recommend():
    data = request.get_json()
    book_id = data.get('book_id')

    if not book_id:
        return jsonify({"error": "book_id is required in the JSON payload."}), 400

    try:
        book_id = int(book_id)
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