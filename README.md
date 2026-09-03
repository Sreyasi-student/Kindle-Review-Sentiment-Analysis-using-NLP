# Kindle Review Sentiment — Fixed Pipeline

Binary sentiment classification (rating > 3 = positive) on Amazon Kindle
reviews, comparing Bag-of-Words, TF-IDF, and Averaged Word2Vec features
across Gaussian Naive Bayes, Multinomial Naive Bayes, AdaBoost, and
Random Forest.

## What was fixed vs. the original notebooks

| Issue | Where | Fix |
|---|---|---|
| **Data leakage**: Word2Vec trained on train+test combined before the split | `AvgWord2Vec.ipynb` | Split happens first; Word2Vec is fit **only** on `X_train` tokens |
| Wrong Naive Bayes variant for sparse count/TF-IDF features | `BoW.ipynb`, `TF-IDF.ipynb` | `GaussianNB` → `MultinomialNB` |
| `max_features='auto'` (removed in sklearn ≥1.3) | `AvgWord2Vec.ipynb` | Replaced with `'sqrt'` / `'log2'` |
| `RandomizedSearchCV` best params found but never used | `AvgWord2Vec.ipynb` | Final model now uses `search.best_params_` |
| Docs with all out-of-vocab words silently dropped, desyncing X/y | `AvgWord2Vec.ipynb` | Zero-vector fallback instead of dropping |
| Stray `Unnamed: 0` column from `to_csv()` without `index=False` | `Preprocessing.ipynb` | `index=False` |
| `stopwords.words('english')` reloaded per word (slow) | `Preprocessing.ipynb` | Loaded once as a `set` |
| No fixed `random_state` on the Word2Vec split | `AvgWord2Vec.ipynb` | Added, shared `split.py` used everywhere |

## Structure
```
src/
  preprocessing.py     # shared text cleaning
  split.py              # single train/test split reused by every model
  train_bow.py
  train_tfidf.py
  train_word2vec.py     # leakage-free Word2Vec + tuned Random Forest
  train_full_matrix.py  # every model x every feature technique
  run_all.py             # runs BoW/TF-IDF/Word2Vec pipelines, prints comparison
requirements.txt
PROJECT_REPORT.md
```

## Run
```bash
pip install -r requirements.txt
python src/run_all.py            # BoW, TF-IDF, tuned Word2Vec pipeline
python src/train_full_matrix.py  # full model x feature comparison matrix
```

## Headline results (80/20 split, random_state=42, same split for all models)

| Model | Accuracy | ROC-AUC |
|---|---|---|
| **TF-IDF + MultinomialNB** | **0.8308** | **0.9068** |
| BoW + MultinomialNB | 0.8287 | 0.8905 |
| Word2Vec + Tuned Random Forest | 0.7504 | 0.8314 |

## Full comparison matrix

| Feature | Model | Accuracy | ROC-AUC |
|---|---|---|---|
| BoW | GaussianNB *(3k features, dense-capped)* | 0.6483 | 0.8024 |
| BoW | MultinomialNB | 0.8287 | 0.8905 |
| BoW | AdaBoost | 0.7158 | 0.7911 |
| BoW | RandomForest | 0.8000 | 0.8815 |
| TF-IDF | GaussianNB *(3k features, dense-capped)* | 0.7142 | 0.7919 |
| TF-IDF | MultinomialNB | 0.8308 | 0.9068 |
| TF-IDF | AdaBoost | 0.7125 | 0.7927 |
| TF-IDF | RandomForest | 0.8000 | 0.8805 |
| Word2Vec | GaussianNB | 0.7217 | 0.7944 |
| Word2Vec | MultinomialNB | N/A (needs non-negative features) | N/A |
| Word2Vec | AdaBoost | 0.7429 | 0.8207 |
| Word2Vec | RandomForest | 0.7546 | 0.8319 |

Only Random Forest was hyperparameter-tuned (via `RandomizedSearchCV`, train-only
CV). AdaBoost was left untuned — its untuned scores (71–74%) trail both
MultinomialNB and tuned Random Forest by enough margin (9+ points) that
tuning was judged unlikely to change the ranking; see `PROJECT_REPORT.md`
for the full reasoning.

## Key takeaway
Once the Word2Vec leakage is removed, it no longer outperforms BoW/TF-IDF
on this dataset. **TF-IDF + MultinomialNB is the best model overall.** The
original notebook's "substantially better" Word2Vec result was inflated by
training the embeddings on test-set text before the split.
