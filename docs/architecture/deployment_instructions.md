# Deployment & Infrastructure Guide

This guide details how to run the MedicalVQA application locally, execute the remote GPU worker on free Google Colab/Kaggle instances, and deploy via Docker Compose.

---

## 1. Local Development Mode (Laptop CPU)

```
[React Frontend] (localhost:5173) ──► [FastAPI Backend] (localhost:8000) ──► [Mock / Remote VLM Client]
```

### 1.1 Backend Startup
```bash
# Activate Python 3.11 environment
.\.venv\Scripts\activate

# Launch FastAPI development server with hot-reload
uvicorn api.app:app --host 127.0.0.1 --port 8000 --reload
```
Interactive OpenAPI documentation will be accessible at: `http://127.0.0.1:8000/docs`.

### 1.2 Frontend Startup
```bash
cd web
npm install
npm run dev
```
Interactive web interface will be accessible at: `http://localhost:5173`.

---

## 2. Free Cloud GPU Inference Setup (Google Colab / Kaggle)

To run **Qwen2.5-VL-3B** and **MedGemma-4B** without local GPU hardware:

1. Open a new Google Colab notebook with a free **T4 GPU** runtime.
2. Clone this repository or copy `api/remote_worker.py` and `notebooks/colab_inference_server.ipynb`.
3. Run the notebook cells:
   ```python
   # Inside Colab Notebook
   !pip install -q transformers accelerate bitsandbytes fastapi uvicorn pyngrok
   
   # Start the remote worker exposing both models in 4-bit precision
   from api.remote_worker import start_worker
   start_worker(auth_token="YOUR_SHARED_SECRET_KEY")
   ```
4. Copy the generated public tunnel URL (e.g., `https://random-subdomain.trycloudflare.com` or ngrok URL).
5. In your local `.env` file, set:
   ```env
   VLM_BACKEND_MODE=remote
   REMOTE_VLM_URL=https://random-subdomain.trycloudflare.com
   REMOTE_VLM_AUTH_TOKEN=YOUR_SHARED_SECRET_KEY
   ```
6. The local backend and web interface will now stream live VLM predictions seamlessly through your local machine at $0.00 cost.

---

## 3. Docker Compose Orchestration

For reproducible containerized deployment on any machine with Docker installed:

```yaml
# docker-compose.yml summary
services:
  backend:
    build:
      context: .
      dockerfile: Dockerfile.backend
    ports:
      - "8000:8000"
    environment:
      - VLM_BACKEND_MODE=mock # or remote
    volumes:
      - ./data:/app/data

  frontend:
    build:
      context: ./web
      dockerfile: Dockerfile.frontend
    ports:
      - "3000:80"
    depends_on:
      - backend
```

Run with:
```bash
docker compose up --build
```
Web app will be available at `http://localhost:3000`.
