# Mood-to-Music Recommender

A Python agent that maps free-text moods to music recommendations, using:

- **OpenAI** (LLM) for mood parsing, tag generation, and re-ranking.
- A small local **seed catalog** of tracks with tags and metadata.

It ships with a CLI and a FastAPI HTTP API.

## Data
This Project uses the following [Kaggle dataset](https://www.kaggle.com/datasets/abdullahorzan/moodify-dataset)

## Quickstart

```bash
uvicorn src.server:app --reload --port 8000
```

Then open your browser to [http://localhost:8000/docs](http://localhost:8000/docs) to see the interactive API docs.

# 🎧 Mood → Music Recommender

A project that takes a free-text mood description and recommends Spotify tracks that fit the vibe.

It uses:

- **OpenAI API** for parsing mood text into structured preferences and for final ranking/rationales.
- **Numeric audio features** from your dataset (`tempo`, `energy`, `valence`, `danceability`, etc.).
- **FastAPI** backend with a `/recommend` endpoint.
- A small **web frontend** served at `/` with a search field and a dropdown (1–10 tracks).

---

## 📊 Dataset

This Project uses the following [Kaggle dataset](https://www.kaggle.com/datasets/abdullahorzan/moodify-dataset)
- `uri` = Spotify track URI
- Only **numeric features** are used (labels/genres ignored for now).

---

## ⚙️ Setup

### 1. Clone & create environment

```bash
git clone <your-repo-url>
cd Mood2Music

# Create a virtual environment with Python ≥3.12
uv venv --python 3.12
source .venv/bin/activate  # (Linux/Mac)
.venv\Scripts\activate     # (Windows)
```
### 2. Install dependencies

```bash
uv sync
```

### 3. Configure .env
Create a .env file in the project with the following content:
```env
OPENAI_API_KEY=your_openai_api_key
OPENAI_MODEL=gpt-4o-mini   # or another chat-capable model
SEED_CATALOG=./path/to/your/dataset.csv
```

## 🚀 Usage
CLI
```bash
python -m src.cli --mood "chill and relaxed" --n_tracks 5
```

API Server
```bash
uvicorn src.server:app --reload --port 8000
```
- Open your browser to [http://localhost:8000](http://localhost:8000) to see the frontend.
- Type your mood, select the number of tracks to be recommended in the dropdown (currently <=10 to reduce compute), and hit "Recommend".
- The results then show:
  - Spotify links
  - LLM-curated reasons
  - A debug panel with the parsed mood


