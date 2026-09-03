# Kindle Review Sentiment Analysis — Project Report

## 1. Problem Statement

Classify Amazon Kindle book reviews as **positive** or **negative** sentiment
using classical NLP feature-engineering techniques and machine learning
classifiers, and identify which combination of text representation and
model performs best.

- **Dataset**: 12,000 Kindle reviews (`all_kindle_review.csv`), each with a
  1–5 star rating.
- **Label**: binarized as positive (`rating > 3`, ratings 4–5, 6,000 reviews)
  vs. negative (`rating <= 3`, ratings 1–3, 6,000 reviews) — a balanced
  binary classification problem.
- **Features compared**: Bag-of-Words (BoW), TF-IDF, Averaged Word2Vec.
- **Models compared**: Gaussian Naive Bayes, Multinomial Naive Bayes,
  AdaBoost, Random Forest.

## 2. Methodology

### 2.1 Preprocessing
Applied uniformly to all reviews before any feature extraction:
1. Lowercase the text.
2. Strip URLs, HTML tags, and non-alphanumeric characters.
3. Remove English stopwords (`nltk.corpus.stopwords`, loaded once as a set).
4. Lemmatize with `WordNetLemmatizer`.
5. Drop any review that becomes empty after cleaning (none did on this
   dataset, but the check is retained for robustness).

### 2.2 Train/test split
A **single 80/20 stratified split** (`random_state=42`) was created once,
on the cleaned text, and reused identically across every feature technique
and model. This guarantees every number in the comparison matrix is
computed on the exact same held-out test set, making the comparison
apples-to-apples.

### 2.3 Feature extraction
- **BoW**: `CountVectorizer`, fit on training text only.
- **TF-IDF**: `TfidfVectorizer`, fit on training text only.
- **Word2Vec**: `gensim.models.Word2Vec` (100-dim, `min_count=5`), trained
  **only on tokenized training text**, then used to embed both train and
  test reviews via unweighted average of word vectors. Reviews with no
  in-vocabulary words receive a zero vector (see §3.2).

In every case the vectorizer/embedding model is **fit only on the training
split** and applied (`.transform()`) to the test split — never fit on
combined data.

### 2.4 Models
- `GaussianNB`, `MultinomialNB`, `AdaBoostClassifier`, `RandomForestClassifier`
  (all `scikit-learn`, `random_state=42` where applicable).
- Random Forest was tuned via `RandomizedSearchCV` (train-only, 3-fold CV)
  over `max_depth`, `max_features`, `min_samples_split`, `n_estimators`.
  AdaBoost was intentionally left untuned (see §5).
- `MultinomialNB` was not run on Word2Vec features: it requires
  non-negative input (it models counts), and averaged Word2Vec vectors
  contain negative values. Marked N/A rather than computed incorrectly.
- `GaussianNB` on BoW/TF-IDF was run on a 3,000-feature-capped vocabulary
  (`max_features=3000`, still fit-train/transform-test) because GaussianNB
  requires a dense matrix, and the full ~35,400-word vocabulary would
  produce a multi-gigabyte dense array — impractical for this environment.
  This cap is called out explicitly wherever the corresponding numbers
  appear; it does not apply to any other model.

## 3. Issues Found in the Original Notebooks (and Fixes)

### 3.1 Data leakage — Word2Vec (critical)
**Original**: `Word2Vec` was trained on the *entire* corpus (all 12,000
reviews) *before* the train/test split was performed. This means the
embedding space — and therefore every downstream feature vector, including
those for the test set — was learned using words and contexts drawn from
reviews that were later placed in the test set. The model had effectively
already "seen" test-set vocabulary/context prior to evaluation.

**Impact**: this leakage was a real contributor to the original notebook's
claim that Word2Vec was "substantially better" than BoW/TF-IDF. After the
fix (embeddings trained on `X_train` only), Word2Vec's best result dropped
to 75.04% accuracy — below both BoW and TF-IDF.

**Fix**: split raw text into train/test first; fit `Word2Vec` only on
tokenized `X_train`; use that fixed model to embed both `X_train` and
`X_test`.

### 3.2 Silent row-dropping bug — Word2Vec
**Original**: reviews whose every word was filtered out by `min_count=5`
caused `avg_word2vec()` to return a bare NaN scalar (not an array). A
downstream `isinstance(x, np.ndarray)` filter silently dropped those rows
from the feature matrix, while the label vector (`messages['rating']`)
kept its original length — a latent bug that would desynchronize
features and labels, or raise a length-mismatch error outright, on any
dataset where it triggers. It happened not to trigger on this dataset.

**Fix**: out-of-vocabulary reviews now receive an explicit zero vector,
so row count is always preserved and features/labels never disalign.

### 3.3 Wrong Naive Bayes variant — BoW & TF-IDF
**Original**: `GaussianNB` applied to BoW/TF-IDF vectors. GaussianNB
assumes continuous, roughly normally-distributed features; BoW/TF-IDF
vectors are sparse, non-negative counts/weights — a poor statistical
match.

**Fix**: `MultinomialNB`, the standard choice for count/TF-IDF features.
**Effect**: BoW accuracy rose from an unsatisfactory baseline to 82.87%;
TF-IDF to 83.08% — the single largest accuracy improvement in this project.

### 3.4 Invalid / unused hyperparameters
- `max_features='auto'` in the `RandomForestClassifier` search grid raises
  a `ValueError` on scikit-learn ≥1.3 (the option was removed). Replaced
  with `'sqrt'` / `'log2'`.
- `RandomizedSearchCV.best_params_` was found but never applied — the
  final model in the original notebook used different, hardcoded values.
  Fixed so the tuned model actually uses the searched parameters.

### 3.5 Minor issues
- `df.to_csv(...)` without `index=False` produced a stray `Unnamed: 0`
  column on reload — fixed.
- `stopwords.words('english')` was reloaded from NLTK on every single
  word during cleaning — fixed by loading once as a `set`.
- No `random_state` on the Word2Vec notebook's train/test split — fixed,
  and centralized into one shared `split.py` so every pipeline uses an
  identical, reproducible split.

## 4. Results

### 4.1 Headline comparison (best config per feature technique)

| Model | Accuracy | ROC-AUC |
|---|---|---|
| **TF-IDF + MultinomialNB** | **83.08%** | **0.9068** |
| BoW + MultinomialNB | 82.87% | 0.8905 |
| Word2Vec + Tuned Random Forest | 75.04% | 0.8314 |

### 4.2 Full matrix — every model x every feature technique

| Feature | Model | Accuracy | ROC-AUC |
|---|---|---|---|
| BoW | GaussianNB *(3k-feature cap)* | 64.83% | 0.8024 |
| BoW | MultinomialNB | 82.87% | 0.8905 |
| BoW | AdaBoost (untuned) | 71.58% | 0.7911 |
| BoW | RandomForest (untuned) | 80.00% | 0.8815 |
| TF-IDF | GaussianNB *(3k-feature cap)* | 71.42% | 0.7919 |
| TF-IDF | MultinomialNB | 83.08% | 0.9068 |
| TF-IDF | AdaBoost (untuned) | 71.25% | 0.7927 |
| TF-IDF | RandomForest (untuned) | 80.00% | 0.8805 |
| Word2Vec | GaussianNB | 72.17% | 0.7944 |
| Word2Vec | MultinomialNB | N/A | N/A |
| Word2Vec | AdaBoost (untuned) | 74.29% | 0.8207 |
| Word2Vec | RandomForest (untuned) | 75.46% | 0.8319 |

## 5. Discussion

**TF-IDF + MultinomialNB is the best model** on this dataset, both by
accuracy and ROC-AUC. This is a case where a simple, well-matched
model/feature pairing beats more sophisticated representations: averaged
Word2Vec discards word order and sentence structure just like BoW/TF-IDF
do, but additionally compresses each review into only 100 dimensions,
losing the vocabulary-level detail that TF-IDF's ~35,000-dimensional
sparse space retains — detail that matters for a lexical task like
sentiment.

**Naive Bayes variant selection mattered more than model sophistication.**
The single largest accuracy swing in this project (GaussianNB → MultinomialNB
on BoW: 64.8% → 82.9%) came from matching the classifier's distributional
assumptions to the feature type, not from switching to a fundamentally
different algorithm.

**Random Forest was tuned; AdaBoost deliberately was not.** Untuned
AdaBoost trailed the best model by 9–12 points across all three feature
types, and its ceiling is structurally limited on this data: its default
weak learner (a depth-1 decision stump) is a poor fit for 100-dimensional
continuous Word2Vec embeddings, and it under-performed Random Forest even
before tuning on BoW/TF-IDF. Typical hyperparameter tuning gains
(1–3 points) were judged very unlikely to close a 9+ point gap, so tuning
was skipped in favor of reporting the honest, untuned baseline. This
decision — and the full untuned matrix — is documented here so the
comparison remains fair and transparent rather than hiding an unfavorable
number.

## 6. Conclusion & Recommendation

For production sentiment classification on this type of review data,
**TF-IDF features with Multinomial Naive Bayes** is the recommended
baseline: highest accuracy and AUC, fast to train, and simple to deploy.
Word2Vec-based approaches did not outperform it here once the data leakage
in the original implementation was corrected — a useful reminder that a
more "advanced" technique is not automatically better, and that evaluation
methodology (leakage-free splitting) can matter more than model choice.

## 7. Limitations & Future Work
- Only unweighted average Word2Vec was tested; TF-IDF-weighted averaging
  or pretrained embeddings (GloVe, FastText) might close some of the gap.
- No transformer-based model (e.g., BERT) was evaluated in this iteration.
- AdaBoost was not tuned; a quick pass with a deeper base estimator
  (`DecisionTreeClassifier(max_depth=3-5)` instead of the default stump)
  is the most promising direction if AdaBoost is revisited.
- GaussianNB's BoW/TF-IDF numbers use a reduced 3,000-feature vocabulary
  for memory reasons; results at full vocabulary size may differ.
