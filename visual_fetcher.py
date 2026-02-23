"""
Visual Fetcher — DiaryOfJazbaat
Emotion → Mood Router → Pexels vertical cinematic fetch
"""

import os
import random
import requests

PEXELS_API_KEY = os.getenv("PEXELS_API_KEY", "")
SAVE_PATH = "assets/backgrounds/background.mp4"

# ─── MOOD ROUTER ──────────────────────────────────────────────────────────────
# 50 moods total, weighted by brand philosophy
# All environment-driven — NO portraits, models, products

PAIN_MOODS = [
    "dark stormy ocean waves cinematic vertical",
    "heavy rain on window cinematic vertical",
    "foggy abandoned forest path vertical",
    "lone tree in winter blizzard vertical",
    "dark thunderstorm lightning sky vertical",
    "rough ocean at night vertical cinematic",
    "wilting flowers in rain vertical",
    "flooded empty street night vertical",
    "dead leaves falling in wind vertical",
    "misty graveyard fog dawn vertical",
]

REALIZATION_MOODS = [
    "golden sunrise over misty mountains vertical",
    "single candle in dark room vertical",
    "first light breaking through storm clouds vertical",
    "calm lake at dawn reflection vertical",
    "moonrise over dark ocean vertical",
    "soft morning fog lifting forest vertical",
    "stars appearing at dusk vertical",
    "quiet river flowing through dark forest vertical",
    "dim lantern in foggy night vertical",
    "open door leading to sunlit field vertical",
]

GROWTH_MOODS = [
    "timelapse clouds moving over mountains vertical",
    "sunlight streaming through forest canopy vertical",
    "spring flowers blooming in meadow vertical",
    "waterfall in lush green forest vertical",
    "green plants growing in sunlight vertical",
    "birds flying over open landscape vertical",
    "sunrise over rolling hills vertical",
    "fresh rain on green leaves vertical",
    "mountain peak above clouds vertical",
    "river flowing toward open horizon vertical",
]

EGO_MOODS = [
    "dramatic storm clouds building vertical",
    "lightning strikes over open desert vertical",
    "volcano eruption cinematic vertical",
    "dark mountain cliff at twilight vertical",
    "powerful waterfall crashing rocks vertical",
    "raging river white water vertical",
    "dramatic rocky coastline waves vertical",
    "dark canyon depth vertical cinematic",
    "thunderstorm over open ocean vertical",
    "high altitude wind swept plateau vertical",
]

CALM_MOODS = [
    "still lake misty morning vertical",
    "gentle snowfall in pine forest vertical",
    "soft clouds moving sunset vertical",
    "peaceful meadow golden hour vertical",
    "slow river through autumn forest vertical",
    "candle flame in dark room vertical",
    "starry night sky vertical cinematic",
    "gentle ocean waves at sunset vertical",
    "soft rain on still pond vertical",
    "quiet snowy mountain valley vertical",
]

MOOD_MAP = {
    "Pain":        PAIN_MOODS,
    "Realization": REALIZATION_MOODS,
    "Growth":      GROWTH_MOODS,
    "Ego":         EGO_MOODS,
    "Calm":        CALM_MOODS,
}

# ─── PEXELS FETCH ─────────────────────────────────────────────────────────────

def fetch_background(emotion: str, max_retries: int = 3) -> str:
    if not PEXELS_API_KEY:
        raise EnvironmentError("PEXELS_API_KEY not set.")

    mood_pool = MOOD_MAP.get(emotion)
    if not mood_pool:
        raise ValueError(f"Unknown emotion: {emotion}")

    os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)

    for attempt in range(max_retries):
        query = random.choice(mood_pool)
        page  = random.randint(1, 5)
        print(f"  Query: '{query}' | Page {page} (attempt {attempt+1})")

        video_url = _search_pexels(query, page)
        if video_url:
            _download_video(video_url, SAVE_PATH)
            return SAVE_PATH

    raise RuntimeError(f"Failed to fetch a valid video after {max_retries} attempts for emotion: {emotion}")


def _search_pexels(query: str, page: int) -> str | None:
    headers = {"Authorization": PEXELS_API_KEY}
    params = {
        "query": query,
        "orientation": "portrait",
        "size": "large",
        "per_page": 15,
        "page": page,
    }

    try:
        resp = requests.get(
            "https://api.pexels.com/videos/search",
            headers=headers,
            params=params,
            timeout=15,
        )
        resp.raise_for_status()
        data = resp.json()
    except requests.RequestException as e:
        print(f"  Pexels API error: {e}")
        return None

    videos = data.get("videos", [])
    candidates = []

    for video in videos:
        w = video.get("width", 0)
        h = video.get("height", 0)
        duration = video.get("duration", 0)

        # Must be vertical and tall enough and long enough
        if h <= w:
            continue
        if h < 1280:
            continue
        if duration < 6:
            continue

        # Pick highest-res file
        files = video.get("video_files", [])
        best = _best_file(files)
        if best:
            candidates.append((h, best))

    if not candidates:
        print("  No suitable vertical videos found for this query.")
        return None

    # Pick the highest resolution candidate
    candidates.sort(key=lambda x: x[0], reverse=True)
    return candidates[0][1]


def _best_file(files: list) -> str | None:
    """Return URL of highest-quality vertical video file."""
    portrait_files = [
        f for f in files
        if f.get("height", 0) > f.get("width", 0)
        and f.get("link", "").endswith(".mp4")
    ]
    if not portrait_files:
        portrait_files = [f for f in files if f.get("link", "").endswith(".mp4")]

    if not portrait_files:
        return None

    portrait_files.sort(key=lambda f: f.get("height", 0), reverse=True)
    return portrait_files[0].get("link")


def _download_video(url: str, path: str):
    print(f"  Downloading video...")
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        with open(path, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                f.write(chunk)
    size_mb = os.path.getsize(path) / (1024 * 1024)
    print(f"  Downloaded: {size_mb:.1f} MB → {path}")
