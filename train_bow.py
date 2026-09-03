"""
Bag-of-Words sentiment model.

Fix applied vs. original notebook:
- MultinomialNB instead of GaussianNB. BoW vectors are sparse,
  non-negative integer counts - GaussianNB assumes continuous,
  roughly-normal features, which counts are not.
(Split-before-fit was already correct in the original notebook - kept as-is.)
"""
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score

from preprocessing import load_and_clean
from split import get_split


def run(raw_csv_path: str):
    df = load_and_clean(raw_csv_path)
    X_train, X_test, y_train, y_test = get_split(df)

    bow = CountVectorizer()
    X_train_bow = bow.fit_transform(X_train)   # fit ONLY on train
    X_test_bow = bow.transform(X_test)         # transform test, no refit

    model = MultinomialNB()
    model.fit(X_train_bow, y_train)

    y_pred = model.predict(X_test_bow)
    y_proba = model.predict_proba(X_test_bow)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)
    report = classification_report(y_test, y_pred)

    print("=== Bag-of-Words + MultinomialNB ===")
    print(f"Accuracy: {acc:.4f}")
    print(f"ROC-AUC : {auc:.4f}")
    print(report)
    return {"model": "BoW + MultinomialNB", "accuracy": acc, "roc_auc": auc}


if __name__ == "__main__":
    run("/mnt/user-data/uploads/all_kindle_review.csv")
