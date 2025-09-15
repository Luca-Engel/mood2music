from __future__ import annotations
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from .agent import Mood2MusicAgent

app = FastAPI(title="Mood-to-Music Recommender")
agent = Mood2MusicAgent()


class RecBody(BaseModel):
    mood: str
    k: int = 6


@app.post("/recommend")
async def recommend(body: RecBody):
    try:
        result = agent.recommend(mood=body.mood, k=body.k)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
