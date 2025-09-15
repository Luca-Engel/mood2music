from __future__ import annotations
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from .agent import Mood2MusicAgent

app = FastAPI(title="Mood-to-Music Recommender")
agent = Mood2MusicAgent()

# CORS so the browser can call your API (loosened for local dev)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static frontend from /public
app.mount("/static", StaticFiles(directory="public"), name="static")


class RecBody(BaseModel):
    mood: str
    k: int = Field(6, ge=1, le=10)  # backend validation (1..10)


@app.get("/")
async def index():
    return FileResponse("public/index.html")


@app.get("/healthz")
async def healthz():
    return {"ok": True}


@app.post("/recommend")
async def recommend(body: RecBody):
    try:
        # (Optional) safety clamp even if someone bypasses UI validation
        k = max(1, min(body.k, 10))
        print("mood:", body.mood)
        result = agent.recommend(mood=body.mood, k=k)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
