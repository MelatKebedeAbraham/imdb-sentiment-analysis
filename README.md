# IMDb Sentiment Analysis (Mini-Project)

This project implements a sentiment analysis pipeline using a **Logistic Regression** classifier on a 5,000-sample subset of the IMDb movie reviews dataset. Built entirely with **scikit-learn**, it includes both a **command-line interface** and a **Flask API** for making predictions.

---

## Features

- Binary sentiment classification: **positive** or **negative**
- **TF-IDF vectorization** with English stop word removal
- **Logistic Regression** for fast, interpretable predictions
- **CLI tool** for quick sentiment checks
- **Flask API** for programmatic or web-based predictions
- Optional: `test_api.py` to test API endpoints programmatically

---

##  Installation

### 1. Clone the repository

```bash
git clone https://github.com/your-username/imdb-sentiment.git
cd imdb-sentiment
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

##  Train the Model

To train the sentiment model on the IMDb review subset:

```bash
python train.py
```

## This will:

- Load the dataset from `Dataset/imdb_balanced_5000`.csv
- Preprocess and vectorize text with TF-IDF
- Train a Logistic Regression classifier
- Evaluate on a test set and print classification report
- Save:
    - `model.pkl`: the trained model
    - `vectorizer.pkl`: the TF-IDF vectorizer


##  Predict via Command Line

To analyze a review from your terminal:

```bash
python predict.py "I loved this movie!"
```

**Output example:**

```bash
positive (0.89 confidence)
```

##  Predict via Flask API

Start the server:

```bash
python app.py
```

## API will be accessible at:

- Local: `http://127.0.0.1:5000`
- LAN: `http://192.168.x.x:5000` (if accessible)

Ensure `model.pkl` and `vectorizer.pkl` are in the same directory.

##  Use the API

Endpoint

```bash
POST /predict
```

JSON Body

```bash
{
  "review": "This movie was fantastic and thrilling!"
}

```

##  Example Usage

1. `curl`

```bash
curl -X POST -H "Content-Type: application/json" \
     -d '{"review":"I loved this movie!"}' \
     http://127.0.0.1:5000/predict

```

2. `test_api.py`

Run your API testing script:

```bash
python test_api.py
```

Make sure `test_api.py` contains your POST request to the local server and handles response output.

3. `Postman Setup`

- Method: `POST`
- URL: `http://127.0.0.1:5000/predict`
- Header: `Content-Type: application/json`
- Body:

```bash
{
  "review": "I hated this boring film."
}
```

##  Notes

- Root `(/)` returns 404 - only `/predict` endpoint is defined.
- Empty input returns:

```bash
{"error": "No review text provided"}
```
- Flask dev server is not for production, use Gunicorn or similar if needed.
- Make sure port 5000 is allowed through firewalls for LAN access.

## Project Structure

```text
.
├── app.py               # Flask API for sentiment predictions
├── train.py             # Script to preprocess data, train, and save model
├── predict.py           # Command-line prediction script
├── test_api.py          # Script to test the Flask API endpoint
├── model.pkl            # Trained Logistic Regression model
├── vectorizer.pkl       # Fitted TF-IDF vectorizer
├── requirements.txt     # Python dependencies
├── README.md            # Project documentation
└── Dataset/
    └── imdb_balanced_5000.csv  # Subset of IMDb dataset used for training

```

## Model Details

- **Algorithm**: Logistic Regression (from `scikit-learn`)
- **Vectorizer**: TF-IDF
  - English stop words removed
- **Dataset**: IMDb movie reviews  
  - 5,000 balanced samples (positive and negative)
  - Source: [IMDb Dataset on Kaggle](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews)