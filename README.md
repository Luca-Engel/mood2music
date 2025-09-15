# 🎵 Mood2Music 🎵
_Movie Recommender based on free-text mood input_

A Python agent that maps free-text moods to music recommendations, using:

- **OpenAI** (LLM) for mood parsing, tag generation, and re-ranking.
- **Numeric audio features** from a local dataset (tempo, energy, valence, danceability, acousticness,
  instrumentalness, liveness, speechiness, loudness).
- **Spotify Web API** (Client Credentials flow) to resolve **track title + artist** from Spotify URIs.
- A **FastAPI** HTTP API and a tiny web frontend.

It ships with a CLI and a FastAPI HTTP API.

## 📊 Dataset

This Project uses the following [Kaggle dataset](https://www.kaggle.com/datasets/abdullahorzan/moodify-dataset)

Key fields used:

- `uri` — Spotify track URI (e.g., `spotify:track:...`)
- Audio features (all 0..1 unless noted):
  - `tempo` (BPM), `energy`, `valence`, `danceability`, `acousticness`,
    `instrumentalness`, `liveness`, `speechiness`, `loudness` (dB, ~[-60, 0])

---

## ⚙️ Setup
### 1. Install uv
Follow the steps described [here](https://docs.astral.sh/uv/getting-started/installation/)

### 2. Install dependencies

```bash
uv sync
```

### 3. Configure .env
Create a .env file in the project with the following content:
```env
# OpenAI
OPENAI_API_KEY=your_openai_api_key
OPENAI_MODEL=gpt-4o-mini   # or another chat-capable model

# Dataset
SONG_METADATA=./path/to/your/dataset.csv

# Spotify (for track title/artist lookups)
SPOTIFY_CLIENT_ID=your_spotify_client_id
SPOTIFY_CLIENT_SECRET=your_spotify_client_secret
```

### 4. Download the dataset
Download the dataset from [Kaggle](https://www.kaggle.com/datasets/abdullahorzan/moodify-dataset) and place it in the path specified in `SONG_METADATA` in your `.env` file.


## 🚀 Usage

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


