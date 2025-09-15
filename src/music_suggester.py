from __future__ import annotations
import re
import pandas as pd
import numpy as np
from typing import List, Dict, Optional
from .config import settings

SPOTIFY_URI_RX = re.compile(r"spotify:track:([A-Za-z0-9]+)|https?://open\.spotify\.com/track/([A-Za-z0-9]+)")


def spotify_link_from_uri(uri: str) -> str | None:
    if not isinstance(uri, str) or not uri:
        return None
    m = SPOTIFY_URI_RX.search(uri)
    if not m:
        # maybe it's just the ID
        if re.fullmatch(r"[A-Za-z0-9]{22}", uri):
            return f"https://open.spotify.com/track/{uri}"
        return None
    track_id = m.group(1) or m.group(2)
    return f"https://open.spotify.com/track/{track_id}"


class SeedCatalog:
    """
    Numeric-only recommender for datasets with columns:
      duration (ms), danceability, energy, loudness, speechiness, acousticness,
      instrumentalness, liveness, valence, tempo, spec_rate, uri
    (The 'labels' column is ignored.)
    """

    def __init__(self, path: str | None = None):
        path = path or settings.seed_catalog_path
        self.df = pd.read_csv(path)

        # Normalize names
        if "duration (ms)" in self.df.columns:
            self.df = self.df.rename(columns={"duration (ms)": "duration_ms"})

        # Required numeric columns
        for col in ["uri", "tempo", "energy", "valence", "danceability"]:
            if col not in self.df.columns:
                raise ValueError(f"Missing required column: {col}")

        # Coerce numeric + fill sensible defaults
        for col in ["tempo", "energy", "valence", "danceability"]:
            self.df[col] = pd.to_numeric(self.df[col], errors="coerce")

        # Tempo fill: median or generic
        if self.df["tempo"].isna().all():
            self.df["tempo"] = 120.0
        else:
            self.df["tempo"] = self.df["tempo"].fillna(self.df["tempo"].median())

        # 0..1 bounded features
        for col in ["energy", "valence", "danceability"]:
            self.df[col] = self.df[col].fillna(0.5).clip(0.0, 1.0)

        # Normalize tempo for alignment math
        t_lo, t_hi = float(self.df["tempo"].min()), float(self.df["tempo"].max())
        if t_lo == t_hi:
            t_hi = t_lo + 1e-6
        self._tempo_lo, self._tempo_hi = t_lo, t_hi
        self.df["tempo_norm"] = (self.df["tempo"] - t_lo) / (t_hi - t_lo)

    def _align(self, series: pd.Series, target: float, bandwidth: float) -> np.ndarray:
        # Gaussian-ish preference centered at target (0..1)
        s = series.to_numpy(dtype=float)
        d = np.clip(np.abs(s - target), 0.0, 1.0)
        return np.exp(- (d / bandwidth) ** 2)

    def search(
            self,
            tags: List[str],  # kept for API compatibility, unused
            k: int = 12,
            bpm_range: Optional[tuple[int, int]] = None,
            energy_range: Optional[tuple[float, float]] = None,
            valence_pref: Optional[str] = None,
    ) -> pd.DataFrame:
        # Targets derived from mood constraints
        if bpm_range and len(bpm_range) == 2:
            tempo_mid = float(sum(bpm_range) / 2.0)
        else:
            tempo_mid = float(self.df["tempo"].median())
        tempo_t = (tempo_mid - self._tempo_lo) / (self._tempo_hi - self._tempo_lo + 1e-6)

        if energy_range and len(energy_range) == 2:
            energy_t = float(sum(energy_range) / 2.0)
        else:
            energy_t = float(self.df["energy"].median())

        valence_map = {"low": 0.25, "neutral": 0.5, "medium": 0.5, "high": 0.75}
        valence_t = valence_map.get((valence_pref or "").lower(), 0.5)

        # Alignment scores
        tempo_align = self._align(self.df["tempo_norm"], tempo_t, bandwidth=0.20)
        energy_align = self._align(self.df["energy"], energy_t, bandwidth=0.20)
        val_align = self._align(self.df["valence"], valence_t, bandwidth=0.25)
        dance_align = self._align(self.df["danceability"], 0.5, bandwidth=0.30)

        # Soft masks for ranges (gentle penalties out of range)
        soft = np.ones(len(self.df), dtype=float)
        if bpm_range:
            lo, hi = bpm_range
            inr = (self.df["tempo"] >= lo) & (self.df["tempo"] <= hi)
            soft *= np.where(inr, 1.0, 0.85)
        if energy_range:
            lo, hi = energy_range
            inr = (self.df["energy"] >= lo) & (self.df["energy"] <= hi)
            soft *= np.where(inr, 1.0, 0.85)

        # Final hybrid score (numeric only)
        score = 0.35 * tempo_align + 0.30 * energy_align + 0.25 * val_align + 0.10 * dance_align
        score *= soft

        out = self.df.copy()
        out["score"] = score
        out = out.sort_values("score", ascending=False).head(k).copy()
        out["spotify_url"] = out["uri"].apply(spotify_link_from_uri)
        return out

    @staticmethod
    def mk_links_from_uri(uri: str) -> Dict[str, str]:
        sp = spotify_link_from_uri(uri)
        return {"spotify": sp} if sp else {}
