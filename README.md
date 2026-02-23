# DiaryOfJazbaat — AI Cinematic Shorts Studio (Full Stack)

Automated emotional YouTube Shorts pipeline.  
**One command → emotional script → cinematic visual → rendered 9:16 short.**

---

## File Structure

```
project/
├── ai_pipeline.py        ← Run this
├── visual_fetcher.py     ← Pexels mood router + download
├── cinematic_engine.py   ← MoviePy grading + zoom + text
├── assets/backgrounds/   ← Downloaded background video (auto-created)
└── output/               ← final_short.mp4 + last_metadata.json (auto-created)
```

---

## Setup

### 1. Install dependencies

```bash
pip install google-generativeai moviepy requests pillow
```

MoviePy requires **FFmpeg**:
```bash
# Ubuntu / Debian
sudo apt install ffmpeg

# macOS
brew install ffmpeg

# Windows: download from https://ffmpeg.org/download.html
```

### 2. Set API keys

```bash
export GEMINI_API_KEY="your_gemini_api_key_here"
export PEXELS_API_KEY="your_pexels_api_key_here"
```

Or create a `.env` file and load it:
```bash
pip install python-dotenv
```

Then add to the top of `ai_pipeline.py`:
```python
from dotenv import load_dotenv
load_dotenv()
```

**Where to get keys:**
- Gemini API: https://aistudio.google.com/app/apikey
- Pexels API: https://www.pexels.com/api/ (free tier, 200 req/hour)

---

## Run

```bash
python ai_pipeline.py
```

**What happens:**
1. Gemini generates a 2-line emotional script + metadata
2. Pexels fetches a vertical cinematic nature video matching the emotion
3. MoviePy renders: slow zoom + color grade + text overlay → 8-second 9:16 short
4. Output saved to `output/final_short.mp4`

---

## Audio

The short exports **silent** intentionally.  
Add trending audio manually inside the YouTube Shorts editor.

This avoids copyright issues and lets you pick trending audio for the algorithm.

---

## Posting Strategy

- **1–2 Shorts per day** — consistency beats volume
- Focus on **retention loops**, not frequency
- Add trending audio in YouTube editor before uploading

---

## Emotion Distribution (Brand)

| Emotion     | Weight |
|-------------|--------|
| Pain        | 50%    |
| Realization | 20%    |
| Growth      | 15%    |
| Ego         | 10%    |
| Calm        | 5%     |

Gemini picks the emotion per script; the mood router maps it to a curated pool of 10 cinematic nature search queries per category.

---

## Environment Variables

| Variable         | Required | Description              |
|------------------|----------|--------------------------|
| `GEMINI_API_KEY` | ✅        | Google AI Studio key     |
| `PEXELS_API_KEY` | ✅        | Pexels free API key      |

---

## Cost Notes

- **Gemini 2.5 Flash** — very low cost per call
- **Pexels** — free tier (200 req/hour, no download limits)
- At 1–2 Shorts/day: Gemini credits last 60–90 days easily

---

## Output

- `output/final_short.mp4` — ready to upload (add audio in YouTube)
- `output/last_metadata.json` — script + emotion metadata log
