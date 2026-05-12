from pathlib import Path
import joblib

from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

from data_loader import load_processed_data

# Define the project root and path to the trained model.
BASE_DIR = Path(__file__).parent
MODEL_PATH = BASE_DIR / "models" / "spam_classifier.pkl"


def train_model():
    """
    Train and save the best SMS spam classification model.

    Workflow:
    1. Load the claned dataset
    2. Split into train/test sets
    3. Use GridSearchCV with StratifiedKFold
    4. Compare MultinomialNB and LogisticRegression
    5. Evaluate the best model on the test set
    6. Save the best model
    """

    print("Loading the cleaned dataset...")
    df = load_processed_data()

    X = df["message"]
    y = df["label"]

    print(f"Dataset size: {df.shape[0]} messages")
    print("\nClass distribution:")
    print(df["label"].value_counts())

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print("\nTraining set size:", X_train.shape[0])
    print("Test set size:", X_test.shape[0])

    pipeline = Pipeline(
        [("vectorizer", CountVectorizer()), ("classifier", MultinomialNB())]
    )

    param_grid = [
        {
            "vectorizer": [CountVectorizer()],
            "vectorizer__lowercase": [True],
            "vectorizer__stop_words": [None, "english"],
            "vectorizer__max_features": [3000, 5000, 10000],
            "vectorizer__ngram_range": [(1, 1), (1, 2)],
            "classifier": [MultinomialNB()],
            "classifier__alpha": [0.01, 0.1, 0.5, 1.0],
        },
        {
            "vectorizer": [TfidfVectorizer()],
            "vectorizer__lowercase": [True],
            "vectorizer__stop_words": [None, "english"],
            "vectorizer__max_features": [3000, 5000, 10000],
            "vectorizer__ngram_range": [(1, 1), (1, 2)],
            "classifier": [MultinomialNB()],
            "classifier__alpha": [0.01, 0.1, 0.5, 1.0],
        },
        {
            "vectorizer": [CountVectorizer()],
            "vectorizer__lowercase": [True],
            "vectorizer__stop_words": [None, "english"],
            "vectorizer__max_features": [3000, 5000, 10000],
            "vectorizer__ngram_range": [(1, 1), (1, 2)],
            "classifier": [
                LogisticRegression(
                    max_iter=1000, class_weight="balanced", solver="liblinear"
                )
            ],
            "classifier__C": [0.01, 0.1, 1.0, 10.0],
        },
        {
            "vectorizer": [TfidfVectorizer()],
            "vectorizer__lowercase": [True],
            "vectorizer__stop_words": [None, "english"],
            "vectorizer__max_features": [3000, 5000, 10000],
            "vectorizer__ngram_range": [(1, 1), (1, 2)],
            "classifier": [
                LogisticRegression(
                    max_iter=1000, class_weight="balanced", solver="liblinear"
                )
            ],
            "classifier__C": [0.01, 0.1, 1.0, 10.0],
        },
    ]

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        scoring={"precision": "precision", "recall": "recall", "f1": "f1"},
        refit="f1",
        cv=cv,
        n_jobs=-1,
        verbose=2,
    )

    print("\nStarting grid search...")
    grid_search.fit(X_train, y_train)

    print("\nBest parameters:")
    print(grid_search.best_params_)

    print("\nBest cross-validation F1:")
    print(grid_search.best_score_)

    best_model = grid_search.best_estimator_

    y_pred = best_model.predict(X_test)

    print("\nFinal test set evaluation:")
    print(
        classification_report(
            y_test, y_pred, labels=[0, 1], target_names=["NotSpam", "Spam"]
        )
    )

    print("Confusion matrix:")
    print(confusion_matrix(y_test, y_pred, labels=[0, 1]))

    joblib.dump(best_model, MODEL_PATH)

    print(f"\nBest model saved to: {MODEL_PATH}")


if __name__ == "__main__":
    train_model()
