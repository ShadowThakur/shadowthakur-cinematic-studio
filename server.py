"""
DiaryOfJazbaat - Cinematic Shorts Studio
Flask backend: script gen, video fetch, render, preview, YouTube upload
Supports multiple users via session-based job management
"""

import os
import sys
import uuid
import threading
import traceback

from flask import Flask, request, jsonify, send_file, session
from flask_cors import CORS
from pathlib import Path

from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

# Ensure project root is on sys.path so cinematic_engine imports cleanly
BASE = Path(__file__).resolve().parent
if str(BASE) not in sys.path:
    sys.path.insert(0, str(BASE))

from google import genai
import requests as req_lib

from google_auth_oauthlib.flow import Flow
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload
from flask import send_from_directory

app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET", "jazbaat-secret-2024")
app.config['MAX_CONTENT_LENGTH'] = 5 * 1024 * 1024  # 5MB limit

limiter = Limiter(
    get_remote_address,
    app=app,
)

# Allow all origins - covers file://, localhost variants, VS Code Live Server, etc.
CORS(app, supports_credentials=True, origins="*")

# Directories
JOBS_DIR   = BASE / "jobs"
ASSETS_DIR = BASE / "assets" / "backgrounds"
JOBS_DIR.mkdir(parents=True, exist_ok=True)
ASSETS_DIR.mkdir(parents=True, exist_ok=True)
BASE = Path(__file__).resolve().parent
# In-memory job store
JOBS: dict = {}

GEMINI_MODEL = "gemini-2.5-flash"

SCRIPT_PROMPT_SINGLE = """
You are a cinematic emotional micro-script writer for a dark, reflective YouTube Shorts brand called DiaryOfJazbaat.

Write a 2-line emotional script. Strict rules:
- Total 16 words or fewer across both lines
- Line 1: strong hook - tension or curiosity
- Line 2: emotional payoff that makes Line 1 feel complete on replay
- Loop-friendly: Line 2 subtly echoes Line 1 so replay feels intentional
- Tone: dark, reflective, cinematic - NO motivational cliches

Output ONLY in this exact format (no markdown, no extra lines):

Line1: <line 1>
Line2: <line 2>
Emotion: <Pain | Realization | Growth | Ego | Calm>
Scene: <brief visual scene description>
Lighting: <lighting description>
Motion: slow zoom
"""

SCRIPT_PROMPT_MULTI = """
You are writing ultra-relatable emotional micro-scripts for YouTube Shorts and Instagram Reels.

GOAL:
Maximum growth.
Lines must be instantly understandable.
No heavy English. No deep psychology. No abstract vocabulary.

Use:
- Simple English
- OR light Hinglish
- OR Hindi + English mix (but readable globally)

Tone:
Relatable. Direct. Emotional. Screenshot-worthy.

DO NOT:
- Use philosophical language
- Use metaphors that require thinking
- Sound like a novel
- Sound like therapy

Each script must focus on one of these universal themes:
• Relationship tension
• Friendship betrayal
• Self-worth
• Expectations
• Effort imbalance
• Overthinking
• Letting go
• One-sided love

RULES:
- EXACTLY 5 scripts
- Each script under 12 total words
- Line1 must be strong and direct (≤5 words preferred)
- Line2 must hit emotionally
- Must feel like everyday life
- Loop-friendly (Line2 should subtly connect back to Line1)

STRICT OUTPUT FORMAT (no extra text):

Line1: <short direct statement>
Line2: <emotional punch>
Emotion: <Pain | Realization | Growth | Ego | Calm>
Scene: <simple relatable visual>
Lighting: <soft | dark | golden | moody>
Motion: slow zoom
---
(repeat for all 5)
"""
MOOD_MAP = {
    "Pain": [
        "dark stormy ocean waves cinematic vertical",
        "heavy rain on window cinematic vertical",
        "foggy abandoned forest path vertical",
        "lone tree in winter blizzard vertical",
        "dark thunderstorm lightning sky vertical",
        "rough ocean at night vertical cinematic",
        "flooded empty street night vertical",
        "dead leaves falling in wind vertical",
        "misty fog dawn vertical cinematic",
        "dark rain soaked ground vertical",
    ],
    "Realization": [
        "golden sunrise over misty mountains vertical",
        "single candle in dark room vertical",
        "first light breaking through storm clouds vertical",
        "calm lake at dawn reflection vertical",
        "moonrise over dark ocean vertical",
        "soft morning fog lifting forest vertical",
        "stars appearing at dusk vertical",
        "quiet river flowing dark forest vertical",
        "dim lantern foggy night vertical",
        "open field at sunrise vertical",
    ],
    "Growth": [
        "timelapse clouds moving over mountains vertical",
        "sunlight streaming through forest canopy vertical",
        "spring flowers blooming meadow vertical",
        "waterfall lush green forest vertical",
        "green plants growing sunlight vertical",
        "birds flying open landscape vertical",
        "sunrise rolling hills vertical",
        "fresh rain green leaves vertical",
        "mountain peak above clouds vertical",
        "river flowing open horizon vertical",
    ],
    "Ego": [
        "dramatic storm clouds building vertical",
        "lightning strikes open desert vertical",
        "dark mountain cliff twilight vertical",
        "powerful waterfall crashing rocks vertical",
        "raging river white water vertical",
        "dramatic rocky coastline waves vertical",
        "dark canyon depth vertical cinematic",
        "thunderstorm open ocean vertical",
        "wind swept plateau vertical cinematic",
        "volcanic lightning storm vertical",
    ],
    "Calm": [
        "still lake misty morning vertical",
        "gentle snowfall pine forest vertical",
        "soft clouds moving sunset vertical",
        "peaceful meadow golden hour vertical",
        "slow river autumn forest vertical",
        "candle flame dark room vertical",
        "starry night sky vertical cinematic",
        "gentle ocean waves sunset vertical",
        "soft rain still pond vertical",
        "quiet snowy mountain valley vertical",
    ],
}

from shutil import rmtree

def clean_jobs_dir():
    global JOBS
    try:
        if JOBS_DIR.exists():
            rmtree(JOBS_DIR)
        JOBS_DIR.mkdir(parents=True, exist_ok=True)
        JOBS = {}
        print("[CLEANUP] Jobs directory reset.")
    except Exception as e:
        print("[CLEANUP ERROR]", e)

def set_status(job_id, status, progress=None, error=None, extra=None, log=None):
    if job_id not in JOBS:
        JOBS[job_id] = {"logs": []}
    JOBS[job_id]["status"] = status
    if progress is not None:
        JOBS[job_id]["progress"] = progress
    if error:
        JOBS[job_id]["error"] = error
    if extra:
        JOBS[job_id].update(extra)
    if log:
        if "logs" not in JOBS[job_id]:
            JOBS[job_id]["logs"] = []
        import time as _time
        JOBS[job_id]["logs"].append({"msg": log, "ts": _time.time()})
        print(f"  [{job_id[:8]}] {log}")


def add_log(job_id, msg):
    set_status(job_id, JOBS.get(job_id, {}).get("status", "running"), log=msg)


def _parse_one_script(block: str) -> dict:
    """Parse one Line1/Line2/Emotion/... block. Returns dict or raises."""
    fields = ["Line1", "Line2", "Emotion", "Scene", "Lighting", "Motion"]
    result = {}
    for line in block.strip().splitlines():
        for field in fields:
            if line.startswith(f"{field}:"):
                result[field] = line[len(field)+1:].strip()
    missing = [f for f in fields if f not in result]
    if missing:
        raise ValueError(f"Missing fields {missing} in block: {block[:80]}")
    if result["Emotion"] not in MOOD_MAP:
        for k in MOOD_MAP:
            if k.lower() in result["Emotion"].lower():
                result["Emotion"] = k
                break
        else:
            result["Emotion"] = "Pain"
    return result


def parse_multi_scripts(raw: str) -> list:
    """Split on --- and parse up to 5 script blocks."""
    blocks = [b.strip() for b in raw.split("---") if b.strip()]
    scripts = []
    for block in blocks[:5]:
        try:
            scripts.append(_parse_one_script(block))
        except Exception as e:
            print(f"  [parse] Skipping malformed block: {e}")
    if not scripts:
        raise ValueError(f"No valid scripts parsed from Gemini output:\n{raw[:300]}")
    return scripts


def render_video(video_path, line1, line2, emotion, output_path, job_id):
    """Delegate to cinematic_engine."""
    from cinematic_engine import render_short
    render_short(
        video_path=str(video_path),
        line1=line1,
        line2=line2,
        emotion=emotion,
        output_path=str(output_path),
        job_id=job_id,
        jobs_dict=JOBS,
    )




# ---------------------------------------------------------------------------
# Pipeline Functions
# ---------------------------------------------------------------------------

def _fetch_one_video(job_id: str, pexels_key: str, emotion: str, slot: int, job_dir) -> str | None:
    """Fetch one vertical video for a given emotion slot. Returns local path or None."""
    import random, time as _time
    mood_pool = MOOD_MAP.get(emotion, MOOD_MAP["Pain"])
    for attempt in range(4):
        query = random.choice(mood_pool)
        page  = random.randint(1, 5)
        add_log(job_id, f"  [visual {slot+1}] Searching: \"{query}\" p{page}")
        try:
            r = req_lib.get("https://api.pexels.com/videos/search",
                            headers={"Authorization": pexels_key},
                            params={"query": query, "orientation": "portrait",
                                    "size": "large", "per_page": 15, "page": page},
                            timeout=20)
            r.raise_for_status()
        except Exception as e:
            add_log(job_id, f"  \u26a0\ufe0f  Pexels error slot {slot+1}: {e}")
            continue
        for v in r.json().get("videos", []):
            vw, vh, dur = v.get("width", 0), v.get("height", 0), v.get("duration", 0)
            if vh <= vw or vh < 1080 or dur < 6:
                continue
            
            files = [
                f for f in v.get("video_files", [])
                if f.get("link", "").endswith(".mp4")
                and f.get("height", 0) > f.get("width", 0)     # must be vertical
                and 1080 <= f.get("height", 0) <= 1920        # enforce range
                ]
            if not files:
                continue
            # Choose file closest to 1920 (but not exceeding)
            files.sort(key=lambda f: abs(f.get("height", 0) - 1920))
            best = files[0]
            add_log(job_id, f"  [pexels] Selected height: {best.get('height')}")


            #files = [f for f in v.get("video_files", []) if f.get("link", "").endswith(".mp4")]
            #if not files:
             #   continue
            #files.sort(key=lambda f: f.get("height", 0), reverse=True)
            #best = files[0]
            from pathlib import Path as _Path
            out_path = _Path(job_dir) / f"visual_{slot}.mp4"
            try:
                t0 = _time.time()
                dl = req_lib.get(best["link"], stream=True, timeout=120)
                dl.raise_for_status()
                total = int(dl.headers.get("content-length", 0))
                done = 0
                with open(out_path, "wb") as fh:
                    for chunk in dl.iter_content(512 * 1024):
                        fh.write(chunk)
                        done += len(chunk)
                        if total:
                            JOBS[job_id][f"dl_pct_{slot}"] = int(done / total * 100)
                elapsed = _time.time() - t0
                size_mb = os.path.getsize(out_path) / (1024 * 1024)
                res = f"{best.get('width','?')}x{best.get('height','?')}"
                add_log(job_id, f"  \u2705 Visual {slot+1} ({emotion}): {res} {size_mb:.1f}MB in {elapsed:.1f}s")
                return str(out_path)
            except Exception as e:
                add_log(job_id, f"  \u26a0\ufe0f Download failed slot {slot+1}: {e}")
    add_log(job_id, f"  \u274c Could not fetch visual {slot+1} for {emotion}")
    return None


def run_options_pipeline(job_id: str, gemini_key: str, pexels_key: str):
    """Generate 5 scripts + fetch 5 visuals in parallel. Sets status options_ready."""
    import time as _time, threading as _th
    try:
        job_dir = JOBS_DIR / job_id
        job_dir.mkdir(exist_ok=True)
        JOBS[job_id] = {"status": "queued", "progress": 0, "logs": []}

        # STEP 1: Generate 5 scripts
        set_status(job_id, "generating_scripts", 5,
                   log="\U0001f9e0 Connecting to Gemini — generating 5 cinematic scripts...")
        client = genai.Client(api_key=gemini_key)
        t0 = _time.time()
        resp = client.models.generate_content(model=GEMINI_MODEL, contents=SCRIPT_PROMPT_MULTI)
        elapsed = _time.time() - t0
        scripts = parse_multi_scripts(resp.text.strip())
        add_log(job_id, f"\u2705 {len(scripts)} scripts generated in {elapsed:.1f}s")
        for i, s in enumerate(scripts):
            add_log(job_id, f"  [{i+1}] {s['Emotion']}: \"{s['Line1']}\" / \"{s['Line2']}\"")
        set_status(job_id, "generating_scripts", 20, extra={"scripts": scripts},
                   log=f"\U0001f4dd Scripts ready — fetching {len(scripts)} cinematic visuals in parallel...")

        # STEP 2: Fetch visuals (parallel)
        set_status(job_id, "fetching_visuals", 25,
                   log="\U0001f3ac Fetching vertical cinematic footage from Pexels (parallel)...")
        visuals_raw = [None] * len(scripts)

        def fetch_worker(i, emotion):
            visuals_raw[i] = _fetch_one_video(job_id, pexels_key, emotion, i, job_dir)

        threads = [_th.Thread(target=fetch_worker, args=(i, s["Emotion"]), daemon=True)
                   for i, s in enumerate(scripts)]
        for t in threads: t.start()
        for t in threads: t.join()

        visual_list = [
            {"slot": i, "emotion": scripts[i]["Emotion"], "path": p}
            for i, p in enumerate(visuals_raw) if p and Path(p).exists()
        ]
        if not visual_list:
            raise RuntimeError("No visuals could be fetched. Check your PEXELS_API_KEY.")

        add_log(job_id, f"\u2705 {len(visual_list)}/{len(scripts)} visuals ready")
        set_status(job_id, "options_ready", 100,
                   extra={"scripts": scripts, "visuals": visual_list},
                   log="\U0001f389 All options ready — select your script + visual to render!")

    except Exception:
        err = traceback.format_exc()
        short_err = err.strip().split("\n")[-1]
        set_status(job_id, "error", error=err, log=f"\U0001f4a5 Pipeline failed: {short_err}")
        print(f"[{job_id[:8]}] OPTIONS ERROR:\n{err}", file=sys.stderr)


def run_render_pipeline(job_id: str, script: dict, visual_path: str):
    import time as _time

    try:
        job_dir = JOBS_DIR / job_id
        output_path = job_dir / "final_short.mp4"

        set_status(job_id, "rendering", 62,
                   log="🎨 Starting cinematic render pipeline...")
        add_log(job_id, f"🎭 Script: \"{script['Line1']}\" / \"{script['Line2']}\"")
        add_log(job_id, f"🎨 Emotion grade: {script['Emotion']}")
        add_log(job_id, "✂ Burning cinematic serif captions...")
        add_log(job_id, "⚙ Encoding H.264 @ 24fps...")

        render_video(
            visual_path,
            script["Line1"],
            script["Line2"],
            script["Emotion"],
            output_path,
            job_id
        )

        size_mb = os.path.getsize(output_path) / (1024 * 1024)

        set_status(job_id, "done", 100,
                   extra={"output": str(output_path), "metadata": script},
                   log=f"🎬 Done! {size_mb:.1f} MB — ready to preview")

    except Exception:
        err = traceback.format_exc()
        short_err = err.strip().split("\n")[-1]
        set_status(job_id, "error", error=err,
                   log=f"💥 Render failed: {short_err}")
        print(f"[{job_id[:8]}] RENDER ERROR:\n{err}", file=sys.stderr)



# ---------------------------------------------------------------------------
# API Routes
# ---------------------------------------------------------------------------
@app.route("/")
def serve_index():
    return send_from_directory(BASE, "index.html")

@app.route("/api/generate", methods=["POST"])
@limiter.limit("5 per minute")
def generate():
    data = request.json or {}
    gemini_key = data.get("gemini_key") or os.getenv("GEMINI_API_KEY", "")
    pexels_key = data.get("pexels_key") or os.getenv("PEXELS_API_KEY", "")
    if not gemini_key or not pexels_key:
        return jsonify({"error": "Both gemini_key and pexels_key are required."}), 400
    clean_jobs_dir()
    job_id = str(uuid.uuid4())
    JOBS[job_id] = {"status": "queued", "progress": 0, "logs": []}
    t = threading.Thread(target=run_options_pipeline, args=(job_id, gemini_key, pexels_key), daemon=True)
    t.start()
    return jsonify({"job_id": job_id})


@app.route("/api/regenerate", methods=["POST"])
@limiter.limit("5 per minute")
def regenerate():
    """Alias for /api/generate — starts a fresh options pipeline."""
    return generate()


@app.route("/api/render-selected", methods=["POST"])
@limiter.limit("5 per minute")
def render_selected():
    data = request.json or {}
    job_id         = data.get("job_id", "")
    selected_script = data.get("selected_script")   # full script dict
    selected_visual = data.get("selected_visual")   # {slot, emotion, path}

    if not job_id or not selected_script or not selected_visual:
        return jsonify({"error": "job_id, selected_script, and selected_visual required."}), 400

    job = JOBS.get(job_id)
    if not job:
        return jsonify({"error": "Job not found."}), 404

    visual_path = selected_visual.get("path", "")
    if not visual_path or not Path(visual_path).exists():
        return jsonify({"error": f"Visual file not found: {visual_path}"}), 400

    # Reset to rendering state (preserve logs)
    JOBS[job_id]["status"]   = "rendering"
    JOBS[job_id]["progress"] = 60
    JOBS[job_id]["output"]   = None

    t = threading.Thread(
        target=run_render_pipeline,
        args=(job_id, selected_script, visual_path),
        daemon=True
    )
    t.start()
    return jsonify({"ok": True, "job_id": job_id})


@app.route("/api/visual/<job_id>/<int:slot>")
def serve_visual(job_id, slot):
    """Stream a pre-fetched visual preview for the selection grid."""
    job = JOBS.get(job_id)
    if not job:
        return jsonify({"error": "Job not found"}), 404
    visuals = job.get("visuals", [])
    match = next((v for v in visuals if v["slot"] == slot), None)
    if not match:
        return jsonify({"error": "Visual not found"}), 404
    path = match["path"]
    if not Path(path).exists():
        return jsonify({"error": "File missing"}), 404
    return send_file(path, mimetype="video/mp4", as_attachment=False)


@app.route("/api/status/<job_id>")
def status(job_id):
    job = JOBS.get(job_id)
    if not job:
        return jsonify({"error": "Job not found"}), 404
    return jsonify(job)


@app.route("/api/preview/<job_id>")
def preview(job_id):
    job = JOBS.get(job_id)
    if not job or job.get("status") != "done":
        return jsonify({"error": "Video not ready yet"}), 404
    path = job.get("output")
    if not path or not Path(path).exists():
        return jsonify({"error": "Output file missing"}), 404
    return send_file(path, mimetype="video/mp4", as_attachment=False)


from flask import after_this_request

@app.route("/api/download/<job_id>")
def download(job_id):
    job = JOBS.get(job_id)
    if not job or job.get("status") != "done":
        return jsonify({"error": "Video not ready"}), 404

    path = job.get("output")
    if not path or not Path(path).exists():
        return jsonify({"error": "Output file missing"}), 404

    job_dir = JOBS_DIR / job_id

    @after_this_request
    def cleanup(response):
        import shutil
        try:
            if job_dir.exists():
                shutil.rmtree(job_dir)
            JOBS.pop(job_id, None)
        except Exception as e:
            print("Cleanup error:", e)
        return response

    return send_file(
        path,
        mimetype="video/mp4",
        as_attachment=True,
        download_name="jazbaat_short.mp4"
    )


# ---------------------------------------------------------------------------
# YouTube OAuth
# ---------------------------------------------------------------------------

YT_SCOPES = ["https://www.googleapis.com/auth/youtube.upload"]
REDIRECT_URI = "http://localhost:5000/api/oauth/callback"


def get_flow(client_id, client_secret):
    return Flow.from_client_config(
        {"web": {
            "client_id": client_id,
            "client_secret": client_secret,
            "redirect_uris": [REDIRECT_URI],
            "auth_uri": "https://accounts.google.com/o/oauth2/auth",
            "token_uri": "https://oauth2.googleapis.com/token",
        }},
        scopes=YT_SCOPES,
        redirect_uri=REDIRECT_URI,
    )


@app.route("/api/oauth/start", methods=["POST"])
def oauth_start():
    data = request.json or {}
    client_id     = data.get("client_id", "")
    client_secret = data.get("client_secret", "")
    if not client_id or not client_secret:
        return jsonify({"error": "client_id and client_secret required"}), 400
    session["yt_client_id"]     = client_id
    session["yt_client_secret"] = client_secret
    flow = get_flow(client_id, client_secret)
    auth_url, state = flow.authorization_url(access_type="offline", include_granted_scopes="true")
    session["oauth_state"] = state
    return jsonify({"auth_url": auth_url})


@app.route("/api/oauth/callback")
def oauth_callback():
    flow = get_flow(session.get("yt_client_id", ""), session.get("yt_client_secret", ""))
    flow.fetch_token(authorization_response=request.url)
    creds = flow.credentials
    session["yt_credentials"] = {
        "token":         creds.token,
        "refresh_token": creds.refresh_token,
        "token_uri":     creds.token_uri,
        "client_id":     creds.client_id,
        "client_secret": creds.client_secret,
        "scopes":        list(creds.scopes),
    }
    return """<html><body>
    <script>
      window.opener && window.opener.postMessage({type:'yt_auth_done'}, '*');
      setTimeout(function(){ window.close(); }, 500);
    </script>
    <p style="font-family:sans-serif;padding:2rem">YouTube connected! You may close this window.</p>
    </body></html>"""


@app.route("/api/yt/status")
def yt_status():
    return jsonify({"connected": "yt_credentials" in session})


@app.route("/api/upload", methods=["POST"])
def upload_to_youtube():
    if "yt_credentials" not in session:
        return jsonify({"error": "Not authenticated with YouTube. Please connect first."}), 401
    data      = request.json or {}
    job_id    = data.get("job_id", "")
    title     = data.get("title",       "DiaryOfJazbaat Short")
    desc      = data.get("description", "#Shorts #DiaryOfJazbaat")
    tags      = data.get("tags",        ["Shorts", "emotional", "cinematic"])
    privacy   = data.get("privacy",     "public")

    job = JOBS.get(job_id)
    if not job or job.get("status") != "done":
        return jsonify({"error": "Video not ready."}), 400
    video_path = job.get("output", "")
    if not Path(video_path).exists():
        return jsonify({"error": "Video file missing on disk."}), 400

    try:
        cd = session["yt_credentials"]
        creds = Credentials(
            token=cd["token"], refresh_token=cd["refresh_token"],
            token_uri=cd["token_uri"], client_id=cd["client_id"],
            client_secret=cd["client_secret"], scopes=cd["scopes"],
        )
        youtube = build("youtube", "v3", credentials=creds)
        body = {
            "snippet": {
                "title":       title,
                "description": desc,
                "tags":        tags,
                "categoryId":  "22",
            },
            "status": {"privacyStatus": privacy},
        }
        media = MediaFileUpload(video_path, mimetype="video/mp4", resumable=True)
        insert_req = youtube.videos().insert(
            part=",".join(body.keys()), body=body, media_body=media
        )
        response = None
        while response is None:
            _, response = insert_req.next_chunk()
        video_id = response.get("id", "")
        return jsonify({
            "success":  True,
            "video_id": video_id,
            "url":      f"https://youtube.com/shorts/{video_id}",
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


#if __name__ == "__main__":
    #os.environ["OAUTHLIB_INSECURE_TRANSPORT"] = "1"
    #print("🔥 OPTIONS ARCHITECTURE ACTIVE")
    #print("DiaryOfJazbaat Studio starting on http://localhost:5000")
    #app.run(debug=True, port=5000, threaded=True)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))

    # Only allow insecure transport locally
    if os.environ.get("RAILWAY_ENVIRONMENT") is None:
        os.environ["OAUTHLIB_INSECURE_TRANSPORT"] = "1"

    print(f"DiaryOfJazbaat Studio starting on port {port}")
    app.run(host="0.0.0.0", port=port)