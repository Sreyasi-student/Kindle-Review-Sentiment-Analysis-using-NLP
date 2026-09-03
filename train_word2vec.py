"""
Averaged Word2Vec sentiment model.

Fixes applied vs. original notebook (this was the notebook with real
data leakage):
1. CRITICAL: train/test split now happens BEFORE Word2Vec is trained.
   The embedding model is fit ONLY on X_train tokens. In the original
   notebook, Word2Vec was trained on the full corpus (train+test)
   before the split - meaning the model had already "seen" test-set
   vocabulary/context, which leaked information into the features
   and inflated test accuracy.
2. Out-of-vocabulary documents (all words filtered by min_count) now
   get a zero vector instead of being silently dropped from the
   feature matrix - previously this could desync X and y if it ever
   triggered (didn't trigger on this dataset, but was a latent bug).
3. max_features='auto' removed from the RandomForest search grid -
   this value was removed in scikit-learn >=1.3 and raises a
   ValueError on current versions.
4. The best hyperparameters found by RandomizedSearchCV are actually
   used for the final model (previously they were found but ignored -
   the final model used different, hardcoded values).
5. random_state added to the split for reproducibility.
"""
import re
import numpy as np
import pandas as pd
import gensim
from nltk import sent_tokenize
from gensim.utils import simple_preprocess
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score

from preprocessing import load_and_clean
from split import get_split, RANDOM_STATE

VECTOR_SIZE = 100
MIN_COUNT = 5


def tokenize(text_series):
    """Sentence/word tokenize a series of cleaned review strings."""
    tokenized = []
    for text in text_series:
        for sent in sent_tokenize(text):
            tokenized.append(simple_preprocess(sent))
    return tokenized


def avg_word2vec(doc, model):
    """Average the embeddings of known words; zero-vector fallback for OOV docs."""
    vectors = [model.wv[w] for w in doc if w in model.wv.index_to_key]
    if len(vectors) == 0:
        return np.zeros(model.vector_size)
    return np.mean(vectors, axis=0)


def vectorize(text_series, model):
    tokenized = tokenize(text_series)
    return np.array([avg_word2vec(doc, model) for doc in tokenized])


def run(raw_csv_path: str):
    df = load_and_clean(raw_csv_path)

    # Split FIRST - before anything is fit on the corpus.
    X_train_text, X_test_text, y_train, y_test = get_split(df)
    y_train = y_train.reset_index(drop=True)
    y_test = y_test.reset_index(drop=True)

    # Word2Vec trained ONLY on training text.
    train_tokens = tokenize(X_train_text)
    w2v_model = gensim.models.Word2Vec(
        train_tokens, vector_size=VECTOR_SIZE, min_count=MIN_COUNT, seed=RANDOM_STATE
    )

    X_train = vectorize(X_train_text, w2v_model)
    X_test = vectorize(X_test_text, w2v_model)  # same model, transform only

    results = {}

    # --- Quick comparison across model families (as in the original notebook) ---
    candidates = {
        "Gaussian Naive Bayes": GaussianNB(),
        "AdaBoost Classifier": AdaBoostClassifier(random_state=RANDOM_STATE),
        "Random Forest": RandomForestClassifier(random_state=RANDOM_STATE),
    }
    print("=== Averaged Word2Vec: model comparison ===")
    for name, clf in candidates.items():
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        print(f"{name:22s} test accuracy: {acc:.4f}")
        results[name] = acc

    # --- Hyperparameter search for Random Forest (train-only, cv on train) ---
    rf_params = {
        "max_depth": [8, 15, None, 10],
        "max_features": [5, 7, "sqrt", "log2"],   # 'auto' removed (invalid now)
        "min_samples_split": [2, 8, 15],
        "n_estimators": [100, 200, 300],
    }
    search = RandomizedSearchCV(
        estimator=RandomForestClassifier(random_state=RANDOM_STATE),
        param_distributions=rf_params,
        n_iter=8,
        cv=3,
        n_jobs=-1,
        random_state=RANDOM_STATE,
    )
    search.fit(X_train, y_train)
    print("\nBest RF params found on train set:", search.best_params_)

    # Use the SEARCHED params (previously ignored in the original notebook).
    best_rf = RandomForestClassifier(random_state=RANDOM_STATE, **search.best_params_)
    best_rf.fit(X_train, y_train)
    y_pred = best_rf.predict(X_test)
    y_proba = best_rf.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)
    print("\n=== Averaged Word2Vec + Tuned Random Forest (final) ===")
    print(f"Accuracy: {acc:.4f}")
    print(f"ROC-AUC : {auc:.4f}")
    print(classification_report(y_test, y_pred))

    return {"model": "Word2Vec + Tuned RandomForest", "accuracy": acc, "roc_auc": auc}


if __name__ == "__main__":
    run("/mnt/user-data/uploads/all_kindle_review.csv")
