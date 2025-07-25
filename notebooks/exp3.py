import os
import re
import string
import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature
import dagshub
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import scipy.sparse
import warnings
from dotenv import load_dotenv
# ========== SUPPRESS WARNINGS ==========
warnings.simplefilter("ignore", UserWarning)
warnings.filterwarnings("ignore")


load_dotenv()

# Set MLflow Tracking URI & DAGsHub integration
MLFLOW_TRACKING_URI = os.getenv.MLFLOW_TRACKING_URI
dagshub.init(repo_owner=os.getenv.DAGSHUB_USER_NAME, repo_name=os.getenv.DAGSHUB_REPO_NAME, mlflow=True)
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment("LoR Hyperparameter Tuning")


# ==========================
# Text Preprocessing Functions
# ==========================
def preprocess_text(text):
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words("english"))
    
    text = text.lower()
    text = re.sub(r'\d+', '', text)
    text = re.sub(f"[{re.escape(string.punctuation)}]", " ", text)
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = " ".join([lemmatizer.lemmatize(word) for word in text.split() if word not in stop_words])
    return text.strip()

# ========== LOAD & PREPARE DATA ==========
def load_and_prepare_data(filepath):
    df = pd.read_csv(filepath)
    df["review"] = df["review"].astype(str).apply(preprocess_text)
    df = df[df["sentiment"].isin(["positive", "negative"])]
    df["sentiment"] = df["sentiment"].map({"negative": 0, "positive": 1})
    
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(df["review"])
    y = df["sentiment"]
    
    return train_test_split(X, y, test_size=0.2, random_state=42), vectorizer

# ========== TRAIN & LOG MODEL ==========
def train_and_log_model(X_train, X_test, y_train, y_test, vectorizer):
    param_grid = {
        "C": [0.1, 1, 10],
        "penalty": ["l1", "l2"],
        "solver": ["liblinear"]
    }

    with mlflow.start_run():
        grid_search = GridSearchCV(LogisticRegression(), param_grid, cv=5, scoring="f1", n_jobs=-1)
        grid_search.fit(X_train, y_train)

        # Log all hyperparameter tuning runs
        for params, mean_score, std_score in zip(grid_search.cv_results_["params"], 
                                                 grid_search.cv_results_["mean_test_score"], 
                                                 grid_search.cv_results_["std_test_score"]):
            with mlflow.start_run(run_name=f"LR with params: {params}", nested=True):
                model = LogisticRegression(**params)
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)

                metrics = {
                    "accuracy": accuracy_score(y_test, y_pred),
                    "precision": precision_score(y_test, y_pred),
                    "recall": recall_score(y_test, y_pred),
                    "f1_score": f1_score(y_test, y_pred),
                    "mean_cv_score": mean_score,
                    "std_cv_score": std_score
                }

                mlflow.log_params(params)
                mlflow.log_metrics(metrics)

                print(f"Params: {params} | Accuracy: {metrics['accuracy']:.4f} | F1: {metrics['f1_score']:.4f}")

        # Log best model
        best_params = grid_search.best_params_
        best_model = grid_search.best_estimator_
        best_f1 = grid_search.best_score_

        mlflow.log_params(best_params)
        mlflow.log_metric("best_f1_score", best_f1)
        mlflow.sklearn.log_model(best_model, "model")

        print(f"\n Best Params: {best_params} | Best F1 Score: {best_f1:.4f}")

# ========== MAIN ==========
if __name__ == "__main__":
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_PATH = os.path.join(BASE_DIR, "..", "data", "raw", "imdb", "data.csv")
    (X_train, X_test, y_train, y_test), vectorizer = load_and_prepare_data(DATA_PATH)
    train_and_log_model(X_train, X_test, y_train, y_test, vectorizer)