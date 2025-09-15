from __future__ import annotations
from typing import Any, Dict, List, Optional
from dataclasses import dataclass
import json
from openai import OpenAI
from .config import settings
from .music_suggester import SeedCatalog


@dataclass
class ParsedMood:
    valence: str
    energy: str
    tempo_bpm: List[int]
    tags: List[str]
    genres: List[str]
    instruments: List[str]
    setting: str
    time_of_day: str
    weather: str
    movie_vibes: List[str]


SYSTEM_INSTRUCT = """Extract compact JSON for the following fields from the mood text:
valence (low/neutral/high), energy (low/medium/high), tempo_bpm (list with [min,max]),
tags (<=8), genres (<=5), instruments (<=5), setting (short), time_of_day, weather, movie_vibes (<=5 short noun-phrases).
Return ONLY JSON.
"""


class Mood2MusicAgent:
    def __init__(self, client: Optional[OpenAI] = None):
        settings.validate()
        self.client = client or OpenAI(api_key=settings.openai_api_key)
        self.model = settings.openai_model
        self.catalog = SeedCatalog(settings.seed_catalog_path)

    def parse_mood(self, mood: str) -> ParsedMood:
        messages = [
            {"role": "system", "content": SYSTEM_INSTRUCT},
            {"role": "user", "content": mood},
        ]
        resp = self.client.chat.completions.create(model=self.model, messages=messages, temperature=0.2)
        answer_content = resp.choices[0].message.content
        print("answer parsed mood:")
        print(answer_content)
        obj = json.loads(answer_content)
        return ParsedMood(**obj)

    def recommend(self, mood: str, k: int = 6) -> Dict[str, Any]:
        parsed = self.parse_mood(mood)

        # Constraints from mood
        bpm_range = None
        if parsed.tempo_bpm and len(parsed.tempo_bpm) == 2:
            bpm_range = (int(parsed.tempo_bpm[0]), int(parsed.tempo_bpm[1]))
        energy_map = {"low": (0.0, 0.4), "medium": (0.35, 0.7), "high": (0.65, 1.0)}
        energy_range = energy_map.get(parsed.energy.lower())

        # Numeric-only retrieval (ignore tags for scoring, but we still pass them to LLM for reasoning)
        seeds = self.catalog.search(
            tags=[],  # no text retrieval
            k=max(k * 3, 18),
            bpm_range=bpm_range,
            energy_range=energy_range,
            valence_pref=parsed.valence,
        )

        # Prepare compact candidates for LLM re-rank / rationale
        seed_items = [
            {
                "uri": row["uri"],
                "tempo": float(row["tempo"]),
                "energy": float(row["energy"]),
                "valence": float(row["valence"]),
                "danceability": float(row["danceability"]),
                "score": float(row["score"]),
            }
            for _, row in seeds.iterrows()
        ]

        prompt = {
            "mood_text": mood,
            "parsed_mood": parsed.__dict__,
            "candidates": seed_items,
            "k": k
        }

        messages = [
            {
                "role": "system",
                "content": (
                    "You are a meticulous music curator. Pick k tracks (by uri) from 'candidates' that best fit the mood. "
                    "Use the numeric features (tempo, energy, valence, danceability) to maintain a cohesive flow. "
                    "Return STRICT JSON: {'recommendations': [{'uri': <uri>, 'reason': <reason>}...]}. Keep each reason <= 25 words."
                    "You MUST return ONLY a single valid JSON object."
                    "Do NOT include markdown, code fences, or any text outside the JSON."
                ),
            },
            {"role": "user", "content": json.dumps(prompt)},
        ]
        resp = self.client.chat.completions.create(model=self.model, messages=messages, temperature=0.35)
        print("answer:")
        answer_content = resp.choices[0].message.content
        print(answer_content)
        rec_json = json.loads(answer_content)

        # Attach links
        out = []
        for rec in rec_json.get("recommendations", []):
            uri = rec.get("uri")
            out.append({
                "uri": uri,
                "reason": rec.get("reason"),
                "links": self.catalog.mk_links_from_uri(uri),
            })

        return {
            "mood": mood,
            "parsed": parsed.__dict__,
            "recommendations": out,
        }
