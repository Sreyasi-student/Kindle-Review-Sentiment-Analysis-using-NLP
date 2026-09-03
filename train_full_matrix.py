"""
Full comparison matrix: every classifier x every feature technique,
using the SAME train/test split throughout (no leakage).

Models: GaussianNB, MultinomialNB, AdaBoostClassifier, RandomForestClassifier
Features: BoW, TF-IDF, Averaged Word2Vec

Notes on cells that are skipped / adapted:
- MultinomialNB requires non-negative features. Averaged Word2Vec vectors
  contain negative values, so MultinomialNB x Word2Vec is mathematically
  invalid and marked N/A rather than silently producing a meaningless
  number.
- GaussianNB needs a DENSE matrix. Full BoW/TF-IDF vocab is ~35k
  features x 9,600 train rows -> a dense float64 matrix would be
  several GB, too large for this environment's memory. For the
  GaussianNB rows only, BoW/TF-IDF are refit with max_features=3000
  (still fit on train only, transform on test only) so the comparison
  is tractable. This is called out explicitly in the results rather
  than hidden.
"""
import warnings
warnings.filterwarnings("ignore")

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score

from preprocessing import load_and_clean
from split import get_split, RANDOM_STATE
from train_word2vec import tokenize, vectorize
import gensim

GNB_MAX_FEATURES = 3000  # dense-matrix cap, GaussianNB rows only


def fit_eval(clf, X_train, y_train, X_test, y_test, needs_dense=False):
    if needs_dense and hasattr(X_train, "toarray"):
        X_train, X_test = X_train.toarray(), X_test.toarray()
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    try:
        y_proba = clf.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, y_proba)
    except Exception:
        auc = None
    return acc, auc


def main():
    df = load_and_clean("/mnt/user-data/uploads/all_kindle_review.csv")
    X_train_text, X_test_text, y_train, y_test = get_split(df)
    y_train = y_train.reset_index(drop=True)
    y_test = y_test.reset_index(drop=True)

    results = []  # (feature, model, accuracy, roc_auc)

    # ---------- BoW / TF-IDF, full vocab (sparse-friendly models) ----------
    bow = CountVectorizer().fit(X_train_text)
    Xtr_bow, Xte_bow = bow.transform(X_train_text), bow.transform(X_test_text)

    tfidf = TfidfVectorizer().fit(X_train_text)
    Xtr_tfidf, Xte_tfidf = tfidf.transform(X_train_text), tfidf.transform(X_test_text)

    # ---------- BoW / TF-IDF, reduced vocab for GaussianNB (dense) ----------
    bow_small = CountVectorizer(max_features=GNB_MAX_FEATURES).fit(X_train_text)
    Xtr_bow_s, Xte_bow_s = (bow_small.transform(X_train_text),
                             bow_small.transform(X_test_text))

    tfidf_small = TfidfVectorizer(max_features=GNB_MAX_FEATURES).fit(X_train_text)
    Xtr_tfidf_s, Xte_tfidf_s = (tfidf_small.transform(X_train_text),
                                 tfidf_small.transform(X_test_text))

    # ---------- Word2Vec (dense, 100-dim, train-only embeddings) ----------
    train_tokens = tokenize(X_train_text)
    w2v_model = gensim.models.Word2Vec(
        train_tokens, vector_size=100, min_count=5, seed=RANDOM_STATE
    )
    Xtr_w2v = vectorize(X_train_text, w2v_model)
    Xte_w2v = vectorize(X_test_text, w2v_model)

    feature_sets = {
        "BoW": (Xtr_bow, Xte_bow, Xtr_bow_s, Xte_bow_s),
        "TF-IDF": (Xtr_tfidf, Xte_tfidf, Xtr_tfidf_s, Xte_tfidf_s),
        "Word2Vec": (Xtr_w2v, Xte_w2v, Xtr_w2v, Xte_w2v),  # already dense/small
    }

    for feat_name, (Xtr_full, Xte_full, Xtr_gnb, Xte_gnb) in feature_sets.items():
        # GaussianNB -> always dense
        acc, auc = fit_eval(GaussianNB(), Xtr_gnb, y_train, Xte_gnb, y_test,
                             needs_dense=True)
        note = "" if feat_name == "Word2Vec" else f"(max_features={GNB_MAX_FEATURES})"
        results.append((feat_name, f"GaussianNB {note}".strip(), acc, auc))

        # MultinomialNB -> only valid for non-negative features
        if feat_name == "Word2Vec":
            results.append((feat_name, "MultinomialNB", None, None))  # N/A
        else:
            acc, auc = fit_eval(MultinomialNB(), Xtr_full, y_train, Xte_full, y_test)
            results.append((feat_name, "MultinomialNB", acc, auc))

        # AdaBoost -> works on sparse or dense
        acc, auc = fit_eval(
            AdaBoostClassifier(random_state=RANDOM_STATE),
            Xtr_full, y_train, Xte_full, y_test,
        )
        results.append((feat_name, "AdaBoost", acc, auc))

        # RandomForest -> works on sparse or dense
        acc, auc = fit_eval(
            RandomForestClassifier(random_state=RANDOM_STATE),
            Xtr_full, y_train, Xte_full, y_test,
        )
        results.append((feat_name, "RandomForest", acc, auc))

    print(f"{'Feature':10s} {'Model':30s} {'Accuracy':>10s} {'ROC-AUC':>10s}")
    print("-" * 64)
    for feat, model, acc, auc in results:
        acc_s = f"{acc:.4f}" if acc is not None else "N/A"
        auc_s = f"{auc:.4f}" if auc is not None else "N/A"
        print(f"{feat:10s} {model:30s} {acc_s:>10s} {auc_s:>10s}")

    return results


if __name__ == "__main__":
    main()
