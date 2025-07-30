# 🎬 Sentiment Analysis of IMDB Reviews – MLOps Project

This project demonstrates an end-to-end MLOps pipeline for performing sentiment analysis on the IMDB movie review dataset. It covers every stage of the ML lifecycle: from data ingestion to model deployment and monitoring using modern MLOps tools and AWS services.

> 🔒 Note: The service is currently deactivated to avoid AWS billing. However, the complete pipeline and configurations are preserved in the repository.

---

## 🚀 Architecture Overview
![architecture](/images/architecture.png)
---


## 📦 Pipeline Stages (from `dvc.yaml`)

Each stage is modular and reproducible:

1. **Data Ingestion** – Load and split raw data.
2. **Data Preprocessing** – Clean and tokenize data.
3. **Feature Engineering** – Convert text to vectorized format using `TfidfVectorizer`.
4. **Model Building** – Train a classification model.
5. **Model Evaluation** – Evaluate metrics (accuracy, precision, recall) and log with MLflow.
6. **Model Registration** – Register the best model version with MLflow.

---

## 🛠 Tech Stack

| Area | Tool/Service |
|------|--------------|
| Programming | Python, Flask |
| Pipeline Orchestration | DVC |
| Model Tracking | MLflow |
| CI/CD | GitHub Actions |
| Data & Model Storage | AWS S3 |
| Containerization | Docker + AWS ECR |
| Deployment | AWS EKS |
| Monitoring | Prometheus + Grafana on EC2 |

---

## Snapshots of Web app
![flask app 1](/images/flaskApp1.png)
![flash app 2](/images/flaskApp2.png)

---
## Snapshots of Grafana monitoring
![grafana monitoring](/images/metrics.png)