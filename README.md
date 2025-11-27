# Bookify Recommendations

A book recommendation system built with Python that uses Non-negative Matrix Factorization (NMF) for collaborative filtering. The system analyzes user-book ratings to generate personalized book recommendations.

## Features

- **Collaborative Filtering**: Uses NMF to decompose the user-item rating matrix into latent factors
- **REST API**: Flask-based API endpoint for getting book recommendations
- **Cosine Similarity**: Alternative recommendation method using cosine similarity between book embeddings

## Dataset

This project uses the [Goodbooks-10K dataset](https://github.com/zygmuntz/goodbooks-10k), which contains:
- `ratings.csv`: User ratings with columns `user_id`, `book_id`, `rating`
- `books.csv`: Book metadata including `book_id`, `title`, `authors`, etc.

## Project Structure

```
├── extractionOfData.py    # Data processing and NMF model training
├── differentversioning.py # Flask API server for recommendations
├── nmf_model.pkl          # Pre-trained NMF model (generated after training)
└── README.md
```

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/bozmund/BookifyRecommendations.git
   cd BookifyRecommendations
   ```

2. Install dependencies:
   ```bash
   pip install pandas numpy scikit-learn flask flask-cors
   ```

3. Download the Goodbooks-10K dataset and place `ratings.csv` and `books.csv` in the project root.

## Usage

### Training the Model

Run the data extraction script to train the NMF model:

```bash
python extractionOfData.py
```

This will:
- Load and process the ratings data
- Create a user-book rating matrix
- Train an NMF model with 30 latent factors
- Save the model to `nmf_model.pkl`

### Starting the API Server

Run the Flask server:

```bash
python differentversioning.py
```

The server will start on `http://localhost:5000`.

### Getting Recommendations

Send a POST request to the `/recommend` endpoint:

```bash
curl -X POST http://localhost:5000/recommend \
  -H "Content-Type: application/json" \
  -d '{"book_id": 1}'
```

**Response:**
```json
{
  "requested_book": {
    "book_id": 1,
    "title": "The Hunger Games"
  },
  "recommendations": [
    {
      "book_id": 2,
      "title": "Harry Potter and the Sorcerer's Stone",
      "predicted_score": 4.5
    }
  ]
}
```

## How It Works

1. **Matrix Factorization**: The user-item rating matrix R is decomposed into two lower-rank matrices W (user factors) and H (item factors) such that R ≈ W × H

2. **Recommendation Generation**: For a given book, the system finds books with similar latent factor representations using the learned embeddings

3. **API**: The Flask server loads the pre-trained model and serves recommendations via a REST endpoint

## Configuration

The following parameters can be adjusted in the source files:

**In `extractionOfData.py`:**
- `n_components`: Number of latent factors (default: 30)
- `max_iter`: Maximum NMF iterations (default: 200)

**In `differentversioning.py`:**
- `top_n`: Number of recommendations to return (default: 10) - passed to recommendation functions

## License

This project is open source and available under the MIT License.
