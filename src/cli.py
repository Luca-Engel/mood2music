from __future__ import annotations
import argparse
import json
from .agent import Mood2MusicAgent


def main():
    parser = argparse.ArgumentParser(description="Mood → Music recommender")
    parser.add_argument("--mood", required=True, help="Free text mood, e.g. 'rainy night focus'")
    parser.add_argument("-k", type=int, default=6, help="How many tracks to return")
    args = parser.parse_args()

    agent = Mood2MusicAgent()
    result = agent.recommend(mood=args.mood, k=args.k)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
