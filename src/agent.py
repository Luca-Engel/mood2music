from __future__ import annotations
from typing import Any, Dict, List, Optional
from dataclasses import dataclass
import json
import re
from openai import OpenAI
from .config import settings
from .music_suggester import SeedCatalog

# Optional: include titles/artists if you added spotify_client.py
try:
    from .spotify_client import SpotifyClient  # requires SPOTIFY_CLIENT_ID/SECRET in .env
except Exception:  # pragma: no cover
    SpotifyClient = None  # type: ignore


@dataclass
class ParsedMood:
    # categorical core
    valence: str | None
    energy: str | None
    tempo_bpm: List[int] | None
    # numeric prefs 0..1 (plus loudness range)
    acousticness_pref: float | None
    danceability_pref: float | None
    energy_pref: float | None
    instrumentalness_pref: float | None
    liveness_pref: float | None
    speechiness_pref: float | None
    valence_pref: float | None
    loudness_db_range: List[float] | None  # e.g. [-18, -6]
    # dataset class label
    mood_class: str | None  # {"sad","happy","energetic","calm"}


SYSTEM_INSTRUCT = """
You MUST return ONLY a single valid JSON object. No prose, no markdown, no code fences.

From the mood text, produce EXACTLY these keys (note that some are categorical and others are continuous 0..1 or ranges):

- mood_class: one of {"sad","happy","energetic","calm"}
- valence: one of {"low","neutral","high"}  # categorical mood positiveness
- energy: one of {"low","medium","high"}    # categorical intensity
- tempo_bpm: [min,max] integers (reasonable defaults if unspecified)
- acousticness_pref: 0..1 (1.0 = very acoustic)
- danceability_pref: 0..1 (1.0 = highly danceable)
- energy_pref: 0..1 (1.0 = very energetic)   # perceptual measure of intensity and activity. Typically, energetic tracks feel fast, loud, and noisy. For example, death metal has high energy, while a Bach prelude scores low on the scale
- instrumentalness_pref: 0..1 (1.0 = likely no vocals)   # The closer to 1 the score, the higher the probability that the retrieved tracks will contain vocals
- liveness_pref: 0..1 (1.0 = likely live performance)
- speechiness_pref: 0..1 (1.0 = speech-like/podcast)   # The more exclusively speech-like the recording (e.g. talk show, audiobook, poetry), the closer to 1.0 the attribute value. Values above 0.66 describe tracks that are probably made entirely of spoken words. Values between 0.33 and 0.66 describe tracks that may contain both music and speech, either in sections or layered, including such cases as rap music. Values below 0.33 most likely represent music and other non-speech-like tracks
- valence_pref: 0..1 (1.0 = very positive/cheerful/euphoric, 0.0 = very sad/depressed/angry)
- loudness_db_range: [min_db,max_db] in dB, typically within [-60, 0] (e.g., [-18, -6])

Return ONLY the JSON object with those keys.
"""


def safe_json(text: str) -> Dict[str, Any]:
    """Tolerate accidental code fences; extract first JSON object."""
    text = (text or "").strip()
    text = re.sub(r"^\s*```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```\s*$", "", text).strip()
    if text.startswith("{") and text.endswith("}"):
        return json.loads(text)
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end != -1 and end > start:
        return json.loads(text[start:end + 1])
    return json.loads(text)


class Mood2MusicAgent:
    def __init__(self, client: Optional[OpenAI] = None):
        settings.validate()
        self.client = client or OpenAI(api_key=settings.openai_api_key)
        self.model = settings.openai_model
        self.catalog = SeedCatalog(settings.song_metadata_path)
        self.spotify = SpotifyClient() if SpotifyClient else None  # optional

    def parse_mood(self, mood: str) -> ParsedMood:
        messages = [
            {"role": "system", "content": SYSTEM_INSTRUCT},
            {"role": "user", "content": mood},
        ]
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0.2,
            response_format={"type": "json_object"},
        )

        mood_dict = safe_json(resp.choices[0].message.content or "{}")

        # Defaults (safe, reasonable)
        mood_dict.setdefault("valence", "neutral")
        mood_dict.setdefault("energy", "medium")
        mood_dict.setdefault("tempo_bpm", [90, 120])

        mood_dict.setdefault("acousticness_pref", 0.5)
        mood_dict.setdefault("danceability_pref", 0.5)
        mood_dict.setdefault("energy_pref", 0.6 if (mood_dict.get("energy") or "medium") == "high" else 0.5)
        mood_dict.setdefault("instrumentalness_pref", 0.5)
        mood_dict.setdefault("liveness_pref", 0.3)
        mood_dict.setdefault("speechiness_pref", 0.2)
        mood_dict.setdefault("valence_pref", 0.5)
        mood_dict.setdefault("loudness_db_range", [-18.0, -6.0])
        mood_dict.setdefault("mood_class", "calm")

        return ParsedMood(**mood_dict)

    def recommend(self, mood: str, k: int = 6) -> Dict[str, Any]:
        k = max(1, min(k, 10))
        parsed = self.parse_mood(mood)

        # Map categorical energy to numeric range for soft filtering
        bpm_range = None
        if parsed.tempo_bpm and len(parsed.tempo_bpm) == 2:
            try:
                bpm_range = (int(parsed.tempo_bpm[0]), int(parsed.tempo_bpm[1]))
            except Exception:
                bpm_range = None

        energy_map = {"low": (0.0, 0.4), "medium": (0.35, 0.7), "high": (0.65, 1.0)}
        energy_range = energy_map.get((parsed.energy or "medium").lower())

        seeds = self.catalog.search(
            k=max(k * 3, 21),
            bpm_range=bpm_range,
            energy_range=energy_range,  # from categorical energy
            valence_pref=parsed.valence or "neutral",  # categorical valence for fallback
            valence_num_pref=float(parsed.valence_pref or 0.5),  # explicit numeric targets
            energy_pref=float(parsed.energy_pref or 0.5),
            danceability_pref=float(parsed.danceability_pref or 0.5),
            acousticness_pref=float(parsed.acousticness_pref or 0.5),
            instrumentalness_pref=float(parsed.instrumentalness_pref or 0.5),
            liveness_pref=float(parsed.liveness_pref or 0.3),
            speechiness_pref=float(parsed.speechiness_pref or 0.2),
            loudness_db_range=tuple(parsed.loudness_db_range or [-18.0, -6.0]),
            mood_class=(parsed.mood_class or "calm"),
        )

        if seeds.empty:
            return {
                "mood": mood,
                "parsed": parsed.__dict__,
                "recommendations": [],
                "note": "No candidates found from the dataset given the constraints.",
            }

        # Compact candidates for re-rank reasons
        seed_items = [
            {
                "uri": row.get("uri"),
                "tempo": float(row.get("tempo", 0.0)),
                "energy": float(row.get("energy", 0.5)),
                "valence": float(row.get("valence", 0.5)),
                "danceability": float(row.get("danceability", 0.5)),
                "acousticness": float(row.get("acousticness", 0.0)),
                "instrumentalness": float(row.get("instrumentalness", 0.0)),
                "liveness": float(row.get("liveness", 0.0)),
                "speechiness": float(row.get("speechiness", 0.0)),
                "loudness": float(row.get("loudness", -10.0)),
                "score": float(row.get("score", 0.0)),
                "label_id": int(row.get("labels")) if "labels" in row else None,
            }
            for _, row in seeds.iterrows()
        ]

        # Ask LLM to pick k & provide short reasons
        prompt = {
            "mood_text": mood,
            "parsed_mood": parsed.__dict__,
            "candidates": seed_items,
            "k": k,
        }
        messages = [
            {
                "role": "system",
                "content": (
                    "You MUST return ONLY JSON. No prose, no code fences.\n"
                    "Pick k tracks by 'uri' that best fit the parsed mood, optimizing flow across tempo/energy/valence.\n"
                    "Also consider acousticness, danceability, instrumentalness, liveness, speechiness and loudness.\n"
                    'Return: {"recommendations":[{"uri":"...", "reason":"<=25 words"}]}'
                ),
            },
            {"role": "user", "content": json.dumps(prompt)},
        ]
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0.3,
            response_format={"type": "json_object"},
        )
        rec_json = safe_json(resp.choices[0].message.content or '{"recommendations": []}')

        # Build output, optionally enriching with title/artist
        out: List[Dict[str, Any]] = []
        for rec in rec_json.get("recommendations", [])[:k]:
            uri = rec.get("uri")
            title = artist = None
            if self.spotify:
                try:
                    info = self.spotify.get_track_info(uri)
                    if info:
                        title, artist = info.get("name"), info.get("artists")
                except Exception:
                    pass
            out.append({
                "uri": uri,
                "title": title,
                "artist": artist,
                "reason": rec.get("reason"),
                "links": self.catalog.mk_links_from_uri(uri),
            })

        return {
            "mood": mood,
            "parsed": parsed.__dict__,
            "recommendations": out,
        }
