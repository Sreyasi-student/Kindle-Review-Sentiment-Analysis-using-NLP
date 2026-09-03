"""Run all three feature/model pipelines and print a comparison table."""
import train_bow
import train_tfidf
import train_word2vec

RAW_CSV = "/mnt/user-data/uploads/all_kindle_review.csv"

if __name__ == "__main__":
    results = []
    results.append(train_bow.run(RAW_CSV))
    print()
    results.append(train_tfidf.run(RAW_CSV))
    print()
    results.append(train_word2vec.run(RAW_CSV))

    print("\n" + "=" * 50)
    print("FINAL COMPARISON (leakage-free, same train/test split)")
    print("=" * 50)
    print(f"{'Model':35s} {'Accuracy':>10s} {'ROC-AUC':>10s}")
    for r in results:
        print(f"{r['model']:35s} {r['accuracy']:>10.4f} {r['roc_auc']:>10.4f}")
