"""
AI Cinematic Shorts Automation Engine
DiaryOfJazbaat Emotional Brand System
Main pipeline orchestrator
"""

import os
import sys
import json
import google.generativeai as genai
from visual_fetcher import fetch_background
from cinematic_engine import render_short

# ─── CONFIG ───────────────────────────────────────────────────────────────────
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL   = "models/gemini-2.5-flash"

SCRIPT_PROMPT = """
You are a cinematic emotional micro-script writer for a dark, reflective YouTube Shorts brand.

Write a 2-line emotional script. Follow these strict rules:
- Total ≤ 16 words across both lines
- Line 1: strong hook that creates tension or curiosity
- Line 2: emotional payoff that makes Line 1 feel complete on replay
- Loop-friendly: Line 2 should subtly echo Line 1 so replay feels intentional
- Language: Urdu/Hindi transliteration OR English (brand: DiaryOfJazbaat)
- Tone: dark, reflective, cinematic — NO motivational clichés

Then output metadata in EXACTLY this format (no extra lines, no markdown):

Line1: <line 1>
Line2: <line 2>
Emotion: <one of: Pain | Realization | Growth | Ego | Calm>
Scene: <brief visual scene description>
Lighting: <lighting description>
Motion: <slow zoom | slow drift | gentle parallax>
"""

# ─── GEMINI SCRIPT GENERATION ─────────────────────────────────────────────────

def generate_script() -> dict:
    if not GEMINI_API_KEY:
        raise EnvironmentError("GEMINI_API_KEY not set.")
    genai.configure(api_key=GEMINI_API_KEY)
    model = genai.GenerativeModel(GEMINI_MODEL)

    response = model.generate_content(SCRIPT_PROMPT)
    raw = response.text.strip()

    print("\n── RAW GEMINI OUTPUT ──────────────────────────────")
    print(raw)
    print("───────────────────────────────────────────────────\n")

    metadata = parse_metadata(raw)
    return metadata


def parse_metadata(raw: str) -> dict:
    """Parse strict key: value format from Gemini output."""
    fields = ["Line1", "Line2", "Emotion", "Scene", "Lighting", "Motion"]
    result = {}
    for line in raw.splitlines():
        for field in fields:
            if line.startswith(f"{field}:"):
                result[field] = line[len(field)+1:].strip()
                break

    missing = [f for f in fields if f not in result]
    if missing:
        raise ValueError(f"Metadata parsing failed. Missing fields: {missing}\nRaw:\n{raw}")

    valid_emotions = {"Pain", "Realization", "Growth", "Ego", "Calm"}
    if result["Emotion"] not in valid_emotions:
        raise ValueError(f"Invalid emotion: {result['Emotion']}. Must be one of {valid_emotions}")

    return result


# ─── MAIN PIPELINE ────────────────────────────────────────────────────────────

def run_pipeline():
    print("=" * 55)
    print("  DiaryOfJazbaat — Cinematic Shorts Engine")
    print("=" * 55)

    # STEP 1: Generate script
    print("\n[1/3] Generating emotional script via Gemini...")
    metadata = generate_script()

    print("── SCRIPT ─────────────────────────────────────────")
    print(f"  {metadata['Line1']}")
    print(f"  {metadata['Line2']}")
    print(f"\n  Emotion  : {metadata['Emotion']}")
    print(f"  Scene    : {metadata['Scene']}")
    print(f"  Lighting : {metadata['Lighting']}")
    print(f"  Motion   : {metadata['Motion']}")
    print("───────────────────────────────────────────────────\n")

    # STEP 2: Fetch visual
    print("[2/3] Fetching cinematic background...")
    video_path = fetch_background(metadata["Emotion"])
    print(f"  Saved → {video_path}\n")

    # STEP 3: Render short
    print("[3/3] Rendering cinematic short...")
    output_path = render_short(
        video_path=video_path,
        line1=metadata["Line1"],
        line2=metadata["Line2"],
        emotion=metadata["Emotion"],
    )
    print(f"\n✅ Final short saved → {output_path}")
    print("\n[Audio] Add trending audio manually in YouTube Shorts editor.")
    print("=" * 55)

    # Save metadata log
    os.makedirs("output", exist_ok=True)
    with open("output/last_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    print("  Metadata logged → output/last_metadata.json")


if __name__ == "__main__":
    run_pipeline()
