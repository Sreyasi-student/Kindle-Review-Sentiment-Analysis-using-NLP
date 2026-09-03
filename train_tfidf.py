"""
TF-IDF sentiment model.

Fix applied vs. original notebook:
- MultinomialNB instead of GaussianNB (same reasoning as BoW - TF-IDF
  vectors are sparse and non-negative, not Gaussian-distributed).
"""
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score

from preprocessing import load_and_clean
from split import get_split


def run(raw_csv_path: str):
    df = load_and_clean(raw_csv_path)
    X_train, X_test, y_train, y_test = get_split(df)

    tfidf = TfidfVectorizer()
    X_train_tfidf = tfidf.fit_transform(X_train)   # fit ONLY on train
    X_test_tfidf = tfidf.transform(X_test)         # transform test, no refit

    model = MultinomialNB()
    model.fit(X_train_tfidf, y_train)

    y_pred = model.predict(X_test_tfidf)
    y_proba = model.predict_proba(X_test_tfidf)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)
    report = classification_report(y_test, y_pred)

    print("=== TF-IDF + MultinomialNB ===")
    print(f"Accuracy: {acc:.4f}")
    print(f"ROC-AUC : {auc:.4f}")
    print(report)
    return {"model": "TF-IDF + MultinomialNB", "accuracy": acc, "roc_auc": auc}


if __name__ == "__main__":
    run("/mnt/user-data/uploads/all_kindle_review.csv")
