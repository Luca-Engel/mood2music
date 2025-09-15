import os
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()


@dataclass
class Settings:
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    openai_model: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

    # knobs
    max_movies: int = int(os.getenv("MAX_MOVIES", "8"))
    song_metadata_path: str = os.getenv("SONG_METADATA", "data/seeds/tracks.csv")

    def validate(self):
        if not self.openai_api_key:
            raise RuntimeError("OPENAI_API_KEY is required. Set it in your environment or .env file.")


settings = Settings()
