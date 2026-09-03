"""
Text preprocessing for the Kindle review sentiment pipeline.

Fixes applied vs. the original notebooks:
- stopwords loaded ONCE as a set (was reloaded on every single word before)
- to_csv now uses index=False (was leaking a stray 'Unnamed: 0' column)
- one shared clean_text() used everywhere instead of duplicated logic
  across BoW / TF-IDF / Word2Vec notebooks
"""
import re
import pandas as pd
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

STOPWORDS = set(stopwords.words("english"))
LEMMATIZER = WordNetLemmatizer()

URL_PATTERN = re.compile(
    r"(http|https|ftp|ssh)://([\w_-]+(?:(?:\.[\w_-]+)+))"
    r"([\w.,@?^=%&:/~+#-]*[\w@?^=%&/~+#-])?"
)


def clean_text(text: str) -> str:
    """Lowercase, strip urls/html/special chars, remove stopwords, lemmatize."""
    text = str(text).lower()
    text = URL_PATTERN.sub("", text)
    text = BeautifulSoup(text, "lxml").get_text()
    text = re.sub("[^a-z A-Z 0-9-]+", "", text)
    words = [w for w in text.split() if w not in STOPWORDS]
    words = [LEMMATIZER.lemmatize(w) for w in words]
    return " ".join(words)


def load_and_clean(raw_csv_path: str) -> pd.DataFrame:
    """Load raw kindle review csv -> cleaned text + binary label dataframe."""
    data = pd.read_csv(raw_csv_path)
    df = data[["reviewText", "rating"]].copy()
    df["rating"] = (df["rating"] > 3).astype(int)  # 1 = positive, 0 = negative
    df["reviewText"] = df["reviewText"].apply(clean_text)
    df = df[df["reviewText"].str.len() > 0].reset_index(drop=True)  # drop empties
    return df


if __name__ == "__main__":
    df = load_and_clean("/mnt/user-data/uploads/all_kindle_review.csv")
    df.to_csv("/home/claude/kindle_pipeline/data_processed.csv", index=False)
    print(df.shape)
    print(df.head(3))
