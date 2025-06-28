import sys
import joblib

def predict_sentiment(text, model, vectorizer):
    vec = vectorizer.transform([text])
    pred = model.predict(vec)[0]
    prob = model.predict_proba(vec)[0][pred]
    sentiment = "positive" if pred == 1 else "negative"
    print(f"{sentiment} ({prob:.2f} confidence)")

def main():
    if len(sys.argv) != 2:
        print("Usage: python predict.py \"Your review text here\"")
        sys.exit(1)

    review_text = sys.argv[1]

    # Load model and vectorizer
    model = joblib.load('model.pkl')
    vectorizer = joblib.load('vectorizer.pkl')

    # Predict
    predict_sentiment(review_text, model, vectorizer)

if __name__ == "__main__":
    main()
 