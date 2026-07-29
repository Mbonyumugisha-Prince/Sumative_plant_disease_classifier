# Plant Disease Classifier — ML Ops Dashboard

A end-to-end machine learning pipeline that classifies tomato leaf diseases from a
photo, with a full MLOps loop: prediction, bulk data upload, and one-click retraining.

- **Video Demo:** https://www.youtube.com/watch?v=mCk6PUxcik8
- **Deployed App (frontend):** https://sumative-plant-disease-classifier-1.onrender.com/
- **Deployed API (backend):** https://sumative-plant-disease-classifier.onrender.com/

> **Note on the free-tier deployment:** both services are hosted on Render's
> free tier, which spins a service down after a period of inactivity. The
> backend does **not** wake itself up automatically when you open the
> frontend — if it's been idle, the first request from the dashboard will
> time out. Before using the live app, open the backend's health endpoint
> directly first and wait for it to respond:
> https://sumative-plant-disease-classifier.onrender.com/health
> Once that returns `{"status": "up", ...}`, the backend is awake and the
> frontend app will work normally. This is a limitation of the free hosting
> tier, not the application itself.

## Project Description

Given a photo of a tomato leaf, the model predicts one of 10 disease classes
(or "Healthy"), plus an 11th `Not_Tomato_Leaf` class used to reject clearly
out-of-distribution uploads (e.g. random photos that aren't tomato leaves at all)
instead of confidently mislabeling them.

- **Model:** custom CNN (3×[Conv2D + MaxPooling], Dropout, L2 regularization,
  built-in Keras data augmentation layers), trained with Adam + EarlyStopping.
- **Dataset:** ~18,300 training images / ~4,600 test images across the 10
  original tomato disease/health classes (PlantVillage-derived), plus a
  bootstrap `Not_Tomato_Leaf` negative class added for production robustness.
- **Latest notebook evaluation (10-class, held-out test set):** 90.47% accuracy,
  91.17% macro precision, 90.32% macro recall, 90.49% macro F1.
- **API:** FastAPI serving `/predict`, `/upload`, `/train`, `/health`.
- **UI:** Streamlit dashboard — model uptime, dataset visualizations, single-image
  prediction, and bulk upload + retrain triggering.

## Project Structure

```
plant_disease_classifier/
├── README.md
├── notebook/
│   └── plant_disease_classifier.ipynb   # data exploration, training, evaluation
├── src/
│   ├── preprocessing.py                 # image loading/preprocessing, dataset loader
│   ├── model.py                         # CNN architecture + training callbacks
│   └── prediction.py                    # model loading, single-image prediction
├── api/
│   └── main.py                          # FastAPI app
├── ui/
│   └── app.py                           # Streamlit dashboard
├── data/
│   ├── train/                           # training images, one folder per class
│   └── test/                            # held-out test images, one folder per class
├── models/
│   ├── model_v1.h5
│   └── class_names.json
├── locust/
│   ├── locustfile.py                    # load test definition
│   └── results/                         # CSV output from load test runs
├── Dockerfile
├── docker-compose.yml                   # API + nginx load balancer, scalable
└── docker/nginx.conf
```

## Setup

**1. Clone the repo and create a virtual environment:**

```bash
git clone <repo-url>
cd plant_disease_classifier
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

**2. Start the API** (in one terminal):

```bash
uvicorn api.main:app --reload --port 8000
```

Serves `/health`, `/predict`, `/upload`, `/train` on `http://localhost:8000`.

**3. Start the Streamlit dashboard** (in a second terminal):

```bash
streamlit run ui/app.py
```

Opens `http://localhost:8501`. Both processes need to stay running — the
dashboard calls the API under the hood for every tab except "Data Visualizations".

**4. (Optional) Regenerate the notebook's evaluation outputs:**

```bash
jupyter nbconvert --to notebook --execute --inplace notebook/plant_disease_classifier.ipynb
```

## Retraining

- **Single prediction:** Predict tab — upload one leaf photo, click Predict.
- **Bulk upload + retrain:** Upload & Retrain tab — upload multiple images under
  a class name (existing or new), then click "Retrain Model Now" to retrain on
  everything currently in `data/train/` (original + uploaded).

## Load Testing (Locust + Docker)

The API is containerized and load-tested with [Locust](https://locust.io) to
show how prediction latency scales with the number of API containers behind
an nginx load balancer.

**Run it yourself:**

```bash
docker compose build
docker compose up -d --scale api=<N>   # N = number of API containers
venv/bin/locust -f locust/locustfile.py --headless -u 25 -r 5 -t 45s \
  --host http://localhost:8080 --csv locust/results/containers_<N>
```

`docker-compose.yml` runs an `api` service (FastAPI + model) behind an `nginx`
reverse proxy on port 8080. nginx re-resolves the `api` service name on every
request via Docker's embedded DNS, so requests get load-balanced across
however many `api` replicas are currently running.

### Results

25 simulated users, 45 seconds, hitting `/predict` (real image upload) and
`/health`. Zero request failures in every run.

| Containers | Avg latency | Median | p95 | p99 | Throughput |
|---|---|---|---|---|---|
| 1 | 341 ms | 350 ms | 480 ms | 540 ms | 28.2 req/s |
| 2 | 111 ms | 95 ms | 240 ms | 290 ms | 44.6 req/s |
| 4 | 63 ms | 49 ms | 130 ms | 210 ms | 49.7 req/s |

**Interpretation:** `/predict` calls the CNN synchronously inside an `async def`
route with no thread pool, so a single container has effectively zero internal
concurrency for predictions — one request fully blocks the event loop until it
finishes. That makes container count the primary lever for throughput here:
going from 1→2 containers cut average latency by roughly 3x and raised
throughput by ~58%. The gain from 2→4 containers is smaller (diminishing
returns), consistent with this machine's Docker VM having a limited number of
CPU cores shared across replicas rather than unlimited parallel capacity.

Raw per-run CSVs are in `locust/results/`.
