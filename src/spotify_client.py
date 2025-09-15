import os
import requests
import time


class SpotifyClient:
    def __init__(self):
        self.client_id = os.getenv("SPOTIFY_CLIENT_ID")
        self.client_secret = os.getenv("SPOTIFY_CLIENT_SECRET")
        self._token = None
        self._expires = 0

    def _get_token(self) -> str:
        now = time.time()
        if self._token and now < self._expires:
            return self._token

        resp = requests.post(
            "https://accounts.spotify.com/api/token",
            data={"grant_type": "client_credentials"},
            auth=(self.client_id, self.client_secret),
        )
        resp.raise_for_status()
        data = resp.json()
        self._token = data["access_token"]
        self._expires = now + data["expires_in"] - 60
        return self._token

    def get_track_info(self, uri: str) -> dict | None:
        if not uri:
            return None
        track_id = None
        if uri.startswith("spotify:track:"):
            track_id = uri.split(":")[-1]
        elif "open.spotify.com/track/" in uri:
            track_id = uri.split("track/")[-1].split("?")[0]
        elif len(uri) == 22:
            track_id = uri
        if not track_id:
            return None

        headers = {"Authorization": f"Bearer {self._get_token()}"}
        resp = requests.get(f"https://api.spotify.com/v1/tracks/{track_id}", headers=headers)
        if resp.status_code != 200:
            return None
        data = resp.json()
        return {
            "name": data.get("name"),
            "artists": ", ".join(a.get("name") for a in data.get("artists", [])),
        }
