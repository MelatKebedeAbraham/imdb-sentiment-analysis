import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, accuracy_score


# Preprocess the dataset
def preprocess_data(csv_path):
    data = pd.read_csv(csv_path)

    data['label'] = data['sentiment'].map({'positive': 1, 'negative': 0})
    data = data.drop(columns=['sentiment'])

    X = data['review']
    y = data['label']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test


# Vectorize the text using TF-IDF
def vectorize_text(X_train, X_test):
    vectorizer = TfidfVectorizer(stop_words='english')
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    return X_train_vec, X_test_vec, vectorizer


# Train the model and evaluate
def train_and_evaluate(X_train_vec, X_test_vec, y_train, y_test):
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_vec, y_train)
    pred = model.predict(X_test_vec)

    accuracy = accuracy_score(y_test, pred)
    print("Test Accuracy:", accuracy)
    print(classification_report(y_test, pred))

    return model, accuracy

# Save model and vectorizer
def save_model(clf, vectorizer, model_path='model.pkl', vec_path='vectorizer.pkl'):
    joblib.dump(clf, model_path)
    joblib.dump(vectorizer, vec_path)
    print(f"Model saved to '{model_path}' and vectorizer to '{vec_path}'")


# Main function
def main():
    csv_path = 'Dataset/imdb_balanced_5000.csv'

    X_train, X_test, y_train, y_test = preprocess_data(csv_path)
    X_train_vec, X_test_vec, vectorizer = vectorize_text(X_train, X_test)
    model, accuracy = train_and_evaluate(X_train_vec, X_test_vec, y_train, y_test)
    save_model(model, vectorizer)

if __name__ == '__main__':
    main()
