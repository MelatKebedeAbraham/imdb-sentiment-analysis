from flask import Flask, request, jsonify
import joblib

app = Flask(__name__)

# Load model and vectorizer
model = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "Welcome to the IMDb Sentiment Analysis API. Use POST /predict with a JSON body containing 'review'."})

@app.route("/favicon.ico")
def favicon():
    return "", 204

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    review_text = data.get("review", "")
    if not review_text:
        return jsonify({"error": "No review text provided"}), 400

    # Transform input text
    text_tfidf = vectorizer.transform([review_text])

    # Predict sentiment and probability
    prediction = model.predict(text_tfidf)[0]
    prob = model.predict_proba(text_tfidf)[0]
    sentiment = "positive" if prediction == 1 else "negative"
    confidence = prob[prediction]

    return jsonify({"sentiment": sentiment, "confidence": float(confidence)})

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)