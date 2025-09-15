from __future__ import annotations
import re
import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Tuple
from .config import settings

SPOTIFY_URI_RX = re.compile(r"spotify:track:([A-Za-z0-9]+)|https?://open\.spotify\.com/track/([A-Za-z0-9]+)")
LABEL_MAP = {"sad": 0, "happy": 1, "energetic": 2, "calm": 3}


def spotify_link_from_uri(uri: str) -> str | None:
    if not isinstance(uri, str) or not uri:
        return None
    m = SPOTIFY_URI_RX.search(uri)
    if not m:
        if isinstance(uri, str) and re.fullmatch(r"[A-Za-z0-9]{22}", uri):
            return f"https://open.spotify.com/track/{uri}"
        return None
    track_id = m.group(1) or m.group(2)
    return f"https://open.spotify.com/track/{track_id}"


class SeedCatalog:
    """
    Numeric recommender using Spotify-style features:
      acousticness, danceability, energy, instrumentalness, liveness,
      loudness (dB), speechiness, valence, tempo, labels (0..3), uri
    The 'labels' column (if present) is used as a soft class bias only.
    """

    def __init__(self, path: str | None = None):
        path = path or settings.song_metadata_path
        self.df = pd.read_csv(path)

        # Normalize names
        if "duration (ms)" in self.df.columns:
            self.df = self.df.rename(columns={"duration (ms)": "duration_ms"})

        # Ensure required columns; create neutral defaults if missing
        req_01 = ["danceability", "energy", "valence", "acousticness",
                  "instrumentalness", "liveness", "speechiness"]
        for c in req_01:
            if c not in self.df.columns:
                self.df[c] = 0.5

        if "tempo" not in self.df.columns:
            self.df["tempo"] = 120.0
        if "loudness" not in self.df.columns:
            self.df["loudness"] = -10.0
        if "uri" not in self.df.columns:
            raise ValueError("Dataset must include a 'uri' column")

        # Coerce numerics and clamp to sensible ranges
        for c in req_01 + ["tempo", "loudness"]:
            self.df[c] = pd.to_numeric(self.df[c], errors="coerce")

        self.df["tempo"] = self.df["tempo"].fillna(self.df["tempo"].median())
        for c in req_01:
            self.df[c] = self.df[c].fillna(0.5).clip(0.0, 1.0)
        self.df["loudness"] = self.df["loudness"].fillna(-10.0).clip(-60.0, 0.0)

        # Optional labels column (class bias)
        if "labels" in self.df.columns:
            self.df["labels"] = pd.to_numeric(self.df["labels"], errors="coerce").fillna(-1).astype(int)
        else:
            self.df["labels"] = -1

        # Normalized tempo and loudness
        t_lo, t_hi = float(self.df["tempo"].min()), float(self.df["tempo"].max())
        if t_lo == t_hi:
            t_hi = t_lo + 1e-6
        self._tempo_lo, self._tempo_hi = t_lo, t_hi
        self.df["tempo_norm"] = (self.df["tempo"] - t_lo) / (t_hi - t_lo)

        L_lo, L_hi = -60.0, 0.0
        self.df["loudness_norm"] = (self.df["loudness"] - L_lo) / (L_hi - L_lo)

    def _align(self, series: pd.Series, target: float, bandwidth: float) -> np.ndarray:
        s = series.to_numpy(dtype=float)
        d = np.clip(np.abs(s - target), 0.0, 1.0)
        return np.exp(- (d / bandwidth) ** 2)

    def _norm_tempo_target(self, bpm_mid: float) -> float:
        return (bpm_mid - self._tempo_lo) / (self._tempo_hi - self._tempo_lo + 1e-6)

    def _norm_loudness_target(self, loud_mid_db: float) -> float:
        # map [-60..0] -> [0..1]
        L_lo, L_hi = -60.0, 0.0
        return (loud_mid_db - L_lo) / (L_hi - L_lo)

    def search(
            self,
            k: int = 12,
            bpm_range: Optional[Tuple[int, int]] = None,
            energy_range: Optional[Tuple[float, float]] = None,  # from categorical energy
            valence_pref: Optional[str] = None,  # "low"/"neutral"/"high"
            valence_num_pref: Optional[float] = None,  # 0..1 override
            energy_pref: Optional[float] = None,  # 0..1 override
            danceability_pref: Optional[float] = None,  # 0..1
            acousticness_pref: Optional[float] = None,  # 0..1
            instrumentalness_pref: Optional[float] = None,  # 0..1
            liveness_pref: Optional[float] = None,  # 0..1
            speechiness_pref: Optional[float] = None,  # 0..1
            loudness_db_range: Optional[Tuple[float, float]] = None,  # dB range
            mood_class: Optional[str] = None,  # "sad"/"happy"/"energetic"/"calm"
    ) -> pd.DataFrame:

        # Targets
        tempo_mid = float(np.mean(bpm_range)) if bpm_range and len(bpm_range) == 2 else float(self.df["tempo"].median())
        tempo_t = self._norm_tempo_target(tempo_mid)

        # energy target: numeric pref overrides range-mid if provided
        if energy_pref is not None:
            energy_t = float(np.clip(energy_pref, 0.0, 1.0))
        elif energy_range and len(energy_range) == 2:
            energy_t = float(np.mean(energy_range))
        else:
            energy_t = float(self.df["energy"].median())

        # valence target: numeric pref overrides categorical mapping
        if valence_num_pref is not None:
            valence_t = float(np.clip(valence_num_pref, 0.0, 1.0))
        else:
            val_map = {"low": 0.25, "neutral": 0.5, "medium": 0.5, "high": 0.75}
            valence_t = val_map.get((valence_pref or "neutral").lower(), 0.5)

        dance_t = 0.5 if danceability_pref is None else float(np.clip(danceability_pref, 0.0, 1.0))
        ac_t = 0.5 if acousticness_pref is None else float(np.clip(acousticness_pref, 0.0, 1.0))
        instr_t = 0.5 if instrumentalness_pref is None else float(np.clip(instrumentalness_pref, 0.0, 1.0))
        live_t = 0.3 if liveness_pref is None else float(np.clip(liveness_pref, 0.0, 1.0))
        speech_t = 0.2 if speechiness_pref is None else float(np.clip(speechiness_pref, 0.0, 1.0))

        loud_mid = float(np.mean(loudness_db_range)) if loudness_db_range and len(loudness_db_range) == 2 else float(
            self.df["loudness"].median())
        loud_t = self._norm_loudness_target(loud_mid)

        # Alignments
        tempo_align = self._align(self.df["tempo_norm"], tempo_t, 0.20)
        energy_align = self._align(self.df["energy"], energy_t, 0.20)
        val_align = self._align(self.df["valence"], valence_t, 0.25)
        dance_align = self._align(self.df["danceability"], dance_t, 0.30)
        ac_align = self._align(self.df["acousticness"], ac_t, 0.30)
        instr_align = self._align(self.df["instrumentalness"], instr_t, 0.30)
        live_align = self._align(self.df["liveness"], live_t, 0.35)
        speech_align = self._align(self.df["speechiness"], speech_t, 0.35)
        loud_align = self._align(self.df["loudness_norm"], loud_t, 0.25)

        # Soft penalties for out-of-range
        soft = np.ones(len(self.df), dtype=float)
        if bpm_range:
            lo, hi = bpm_range
            soft *= np.where((self.df["tempo"] >= lo) & (self.df["tempo"] <= hi), 1.0, 0.85)
        if energy_range:
            lo, hi = energy_range
            soft *= np.where((self.df["energy"] >= lo) & (self.df["energy"] <= hi), 1.0, 0.85)

        # Class bias
        class_boost = np.zeros(len(self.df), dtype=float)
        if mood_class:
            target = LABEL_MAP.get(mood_class.lower(), None)
            if target is not None:
                class_boost = np.where(self.df["labels"].to_numpy() == target, 0.05, 0.0)

        # Final score (weights tunable)
        score = (
                0.20 * tempo_align +
                0.18 * energy_align +
                0.18 * val_align +
                0.10 * dance_align +
                0.08 * ac_align +
                0.08 * instr_align +
                0.06 * speech_align +
                0.04 * live_align +
                0.08 * loud_align
        )
        score = (score * soft) + class_boost

        out = self.df.copy()
        out["score"] = score
        out = out.sort_values("score", ascending=False).head(k).copy()
        out["spotify_url"] = out["uri"].apply(spotify_link_from_uri)
        return out

    @staticmethod
    def mk_links_from_uri(uri: str) -> Dict[str, str]:
        sp = spotify_link_from_uri(uri)
        return {"spotify": sp} if sp else {}
