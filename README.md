# 🎬 Movie Review Sentiment – Flask ML System (MLOps Practicals)

Course: **2CEAI702 – MLOps (Sem 7)**
Repository: https://github.com/Mr-CodeGuru/mlops-movie-sentiment

## 1. Project Title and Description

**Movie Review Sentiment Classification: from notebook to a Dockerized, cloud-deployed Flask service.**

This project trains a lightweight text classifier on the *IMDB Dataset of 50K Movie Reviews* (Kaggle: `lakshmi25npathi/imdb-dataset-of-50k-movie-reviews`) that labels a review as **positive** or **negative**. The trained model pipeline is serialized to `model.pkl` and served through a Flask application with input validation, a web interface and a REST API. The service is then packaged with Docker and deployed on AWS EC2.

<p align="center">
  <img width="623" alt="image" src="https://github.com/user-attachments/assets/41dc36cc-1d1e-41ea-b304-ed3c1ff2fc0c" />
</p>

The system is **local-first and lightweight**: it runs fully offline, needs no GPU and no external database, and the model file is only a few MB.

**ML pipeline**

```
IMDB CSV → TF-IDF (1-2 grams) + Logistic Regression → model.pkl → Flask (routes + validation) → Web UI / REST API → Docker → AWS EC2
```

**Features**
- TF-IDF + Logistic Regression pipeline with evaluation metrics saved to `metrics.json`
- Web UI with sample reviews, character counter, confidence ring and positive/negative probability bar (light/dark theme, mobile friendly)
- REST API returning label, confidence and class probabilities
- Input validation (10 to 10,000 characters) and graceful handling of a missing model (`/health` returns 503)
- Automated tests with pytest (no dependency on the real model)
- Production serving with Gunicorn inside a non-root Docker container with a healthcheck

## 2. Technologies and Tools Used

| Area | Tools |
|---|---|
| Language | Python 3.11 |
| ML | scikit-learn (TfidfVectorizer, LogisticRegression, Pipeline), pandas, NumPy, joblib |
| Web / API | Flask, Gunicorn |
| Frontend | HTML, CSS, vanilla JavaScript (no external libraries or CDNs) |
| Testing | pytest |
| Dataset access | kagglehub (training only) |
| Containerization | Docker |
| Cloud | AWS EC2 (Ubuntu 24.04 LTS, `t3.micro`, region `eu-north-1`) |
| Dev environment | macOS, VS Code, Python `venv` |

## 3. Project Structure

```
mlops-movie-sentiment/
├── app.py                    # Flask app (routes, validation, prediction)
├── train_model.py            # Trains the pipeline and writes model.pkl + metrics.json
├── download_data.py          # Downloads the dataset from Kaggle into data/
├── model.pkl                 # Serialized trained pipeline
├── metrics.json              # Evaluation metrics from the last training run
├── requirements.txt          # Full development environment (pip freeze)
├── requirements-docker.txt   # Minimal pinned runtime dependencies for the image
├── Dockerfile
├── .dockerignore
├── .gitignore
├── templates/index.html      # Web UI
├── static/style.css          # UI styling
└── tests/test_app.py         # pytest suite
```

## 4. API Endpoints

| Route | Method | Description |
|---|---|---|
| `/` | GET | Web interface |
| `/health` | GET | Health check (`200` if model loaded, `503` otherwise) |
| `/predict` | POST | HTML form submission, renders the result page |
| `/api/predict` | POST | JSON API |

Example:

```bash
curl -X POST http://127.0.0.1:5001/api/predict \
  -H "Content-Type: application/json" \
  -d '{"review": "An absolute masterpiece, superb acting and direction."}'
```

Response:

```json
{
  "sentiment": "positive",
  "confidence": 0.97,
  "probabilities": {"negative": 0.03, "positive": 0.97}
}
```

`GET` requests to `/predict` redirect to the home page and `GET /api/predict` returns a usage hint, since both routes are POST-only for actual predictions.

## 5. Instructions to Run the Project

### 5.1 Run locally

```bash
# 1. Clone and enter the project
git clone https://github.com/Mr-CodeGuru/mlops-movie-sentiment.git
cd mlops-movie-sentiment

# 2. Create and activate a virtual environment
python3 -m venv venv
source venv/bin/activate            # Windows PowerShell: .\venv\Scripts\Activate.ps1

# 3. Install dependencies
pip install -r requirements.txt

# 4. Download the dataset and train the model (creates model.pkl and metrics.json)
python download_data.py
python train_model.py

# 5. Start the app
python app.py
```

Open **http://127.0.0.1:5001** (the default is 5001 because port 5000 is used by AirPlay Receiver on macOS). Override with the `PORT` environment variable, for example `PORT=5000 python app.py`.

If `download_data.py` fails, download the CSV manually from Kaggle and save it as `data/imdb.csv`.

### 5.2 Run the tests

```bash
pytest -v
```

### 5.3 Run with Docker

```bash
docker build -t movie-sentiment:1.0 .
docker run -d --name movie-sentiment-app --restart unless-stopped -p 5001:5000 movie-sentiment:1.0

docker ps
docker logs movie-sentiment-app
curl http://localhost:5001/health
```

Open **http://localhost:5001**. The container always listens on port 5000 internally; `5001:5000` maps it to port 5001 on the host.

Stop and clean up:

```bash
docker stop movie-sentiment-app
docker rm -f movie-sentiment-app
```

Note: `requirements-docker.txt` pins the same scikit-learn version that produced `model.pkl`. A model pickled with one scikit-learn version may fail to load on another, so keep them in sync when retraining.

### 5.4 Deploy on AWS EC2

1. Launch an Ubuntu 24.04 LTS `t3.micro` instance with a key pair, and a security group allowing **TCP 22** (SSH) and **TCP 5000** (application).
2. Connect: `ssh -i <key>.pem ubuntu@<EC2_PUBLIC_IP>`
3. Install Docker on the instance:
   ```bash
   sudo apt update && sudo apt install -y docker.io
   sudo systemctl enable --now docker
   ```
4. Copy the project files from your machine:
   ```bash
   ssh -i <key>.pem ubuntu@<EC2_PUBLIC_IP> "mkdir -p ~/movie-sentiment"
   scp -i <key>.pem -r Dockerfile .dockerignore requirements-docker.txt app.py model.pkl templates static \
     ubuntu@<EC2_PUBLIC_IP>:/home/ubuntu/movie-sentiment/
   ```
5. Build and run on the instance (the image is built on the server, which avoids ARM vs x86 architecture mismatches):
   ```bash
   cd ~/movie-sentiment
   sudo docker build -t movie-sentiment:1.0 .
   sudo docker run -d --name movie-sentiment-app --restart unless-stopped -p 5000:5000 movie-sentiment:1.0
   ```
6. Verify: open `http://<EC2_PUBLIC_IP>:5000` and `http://<EC2_PUBLIC_IP>:5000/health`.

## 6. Deployment URLs

| Platform | URL | Status |
|---|---|---|
| **AWS (EC2)** | http://13.49.44.172:5000<br>(health check: http://13.49.44.172:5000/health) | Deployed on EC2 (`eu-north-1`). The public IP changes if the instance is stopped and started, and the instance may be terminated after evaluation to avoid charges, so the link can be offline. |
| **GCP (Compute Engine)** | Not available | Not deployed. Google Cloud requires a billing account with a payment method before the Compute Engine API can be enabled, and none was available for this lab. The same Docker image and the same steps used for AWS apply (create an Ubuntu VM, add a firewall rule for TCP 5000, install Docker, build and run the container). |

## 7. Notes

- Evaluation metrics (accuracy and F1 on a stratified 20% hold-out split) are written to `metrics.json` after each training run.
- The dataset (`data/`) is not committed to the repository because of its size. `model.pkl` is committed so that the Docker image can be built without retraining.
- Security: the lab deployment exposes port 5000 to the internet without authentication. Do not use it for sensitive data. Terminate the instance when finished.
