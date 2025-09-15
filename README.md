# Mood-to-Music Recommender

A Python agent that maps free-text moods to music recommendations, using:

- **OpenAI** (LLM) for mood parsing, tag generation, and re-ranking.
- **TMDB** (optional) for *movie vibe* retrieval.
- A small local **seed catalog** of tracks with tags and metadata.

It ships with a CLI and a FastAPI HTTP API.

## Data
This Project uses the following [Kaggle dataset](https://www.kaggle.com/datasets/abdullahorzan/moodify-dataset)

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
