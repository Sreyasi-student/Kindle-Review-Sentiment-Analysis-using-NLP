"""
Single shared train/test split, reused by every feature approach.

Fix applied vs. the original notebooks:
- Splitting happens ONCE, on raw cleaned text, before any vectorizer
  or Word2Vec model is fit. Every downstream feature method (BoW,
  TF-IDF, Word2Vec) fits only on X_train and transforms X_test -
  this is what removes the data leakage in the Word2Vec notebook,
  where the embedding model used to be trained on train+test combined.
"""
from sklearn.model_selection import train_test_split

RANDOM_STATE = 42
TEST_SIZE = 0.20


def get_split(df):
    X_train, X_test, y_train, y_test = train_test_split(
        df["reviewText"],
        df["rating"],
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=df["rating"],
    )
    return X_train, X_test, y_train, y_test
