import json
import os
import re
import sqlite3
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from google import genai

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
_gemini_client = genai.Client(api_key=GEMINI_API_KEY) if GEMINI_API_KEY else None

app = FastAPI(title="Extreme Top Ten API")

DB_NAME = os.getenv("CACHE_DB", "cache.db")


def init_db() -> None:
    conn = sqlite3.connect(DB_NAME)
    try:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS top_tens (
                topic TEXT PRIMARY KEY,
                data_json TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        conn.commit()
    finally:
        conn.close()


init_db()


@contextmanager
def db():
    conn = sqlite3.connect(DB_NAME)
    try:
        yield conn
    finally:
        conn.close()


WIKI_API = "https://commons.wikimedia.org/w/api.php"


def _strip_html(s):
    return re.sub(r"<[^>]+>", "", s or "").strip()


def fetch_wikimedia_image(name):
    """Best-effort: thumbnail + credit for a topic name. Returns dict or None."""
    try:
        params = {
            "action": "query",
            "format": "json",
            "generator": "search",
            "gsrsearch": "filetype:bitmap " + name,
            "gsrnamespace": 6,
            "gsrlimit": 3,
            "prop": "pageimages|imageinfo|info",
            "piprop": "thumbnail",
            "pithumbsize": 900,
            "iiprop": "url|extmetadata",
            "iiextmetadatafilter": "Artist|LicenseShortName",
            "inprop": "url",
        }
        url = WIKI_API + "?" + urllib.parse.urlencode(params)
        req = urllib.request.Request(
            url, headers={"User-Agent": "ExtremeTopTen/1.0 (extremetop10.com)"}
        )
        with urllib.request.urlopen(req, timeout=8) as resp:
            data = json.load(resp)
        pages = (data.get("query") or {}).get("pages") or {}
        for p in pages.values():
            thumb = (p.get("thumbnail") or {}).get("source")
            if not thumb:
                continue
            ii = (p.get("imageinfo") or [{}])[0]
            meta = ii.get("extmetadata") or {}
            artist = _strip_html((meta.get("Artist") or {}).get("value"))
            lic = _strip_html((meta.get("LicenseShortName") or {}).get("value"))
            return {
                "thumb": thumb,
                "page_url": p.get("fullurl"),
                "credit": artist or "Wikimedia Commons contributor",
                "license": lic or "",
            }
    except Exception:
        return None
    return None


def enrich_with_images(data):
    """Attach a Wikimedia image+credit to every item, in parallel."""
    items = data.get("items") or []
    names = [it.get("name", "") for it in items]
    with ThreadPoolExecutor(max_workers=5) as ex:
        images = list(ex.map(fetch_wikimedia_image, names))
    for it, img in zip(items, images):
        if img:
            it["image"] = img
    return data


class QueryRequest(BaseModel):
    topic: str


@app.get("/api/health")
def health():
    return {"status": "ok", "generator_configured": bool(GEMINI_API_KEY)}


@app.get("/api/lists")
def list_lists(q: str = ""):
    """The record archive: every list ever generated, searchable."""
    with db() as conn:
        if q.strip():
            like = f"%{q.strip().lower()}%"
            rows = conn.execute(
                "SELECT topic, data_json, created_at FROM top_tens "
                "WHERE topic LIKE ? ORDER BY created_at DESC LIMIT 100",
                (like,),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT topic, data_json, created_at FROM top_tens "
                "ORDER BY created_at DESC LIMIT 100"
            ).fetchall()
    out = []
    for topic, data_json, created_at in rows:
        try:
            data = json.loads(data_json)
        except Exception:
            continue
        out.append(
            {
                "topic": topic,
                "title": data.get("title") or topic,
                "items": len(data.get("items") or []),
                "on_record_since": (created_at or "")[:10],
            }
        )
    return {"lists": out, "count": len(out)}


@app.post("/api/get-top-ten")
async def get_top_ten(req: QueryRequest):
    topic_clean = req.topic.strip().lower()
    if not topic_clean:
        raise HTTPException(status_code=400, detail="Topic cannot be empty.")
    if len(topic_clean) > 200:
        raise HTTPException(status_code=400, detail="Topic is too long.")

    # 1. Check cache first
    with db() as conn:
        row = conn.execute(
            "SELECT data_json FROM top_tens WHERE topic = ?", (topic_clean,)
        ).fetchone()
    if row:
        data = json.loads(row[0])
        # Backfill images for entries cached before images existed
        items = data.get("items") or []
        if items and "image" not in items[0]:
            data = enrich_with_images(data)
            with db() as conn:
                conn.execute(
                    "UPDATE top_tens SET data_json = ? WHERE topic = ?",
                    (json.dumps(data), topic_clean),
                )
                conn.commit()
        return {"source": "cache", "data": data}

    # 2. No live generation without a key
    if not GEMINI_API_KEY:
        raise HTTPException(
            status_code=503,
            detail=(
                "List generation is not configured yet — the backend has no "
                "Gemini API key. Set the GEMINI_API_KEY environment variable "
                "on the service and try again."
            ),
        )

    # 3. Generate via Gemini with strict factual schema
    prompt = f"""
    You are an impartial, factual data record-keeper for extremetop10.com.
    Provide the top 10 verified items for the topic: "{topic_clean}".
    Focus strictly on objective metrics, documented records, or recognized facts.
    Do not invent figures; if a figure is not widely documented, say so in the summary.

    Return pure JSON with this exact schema:
    {{
        "title": "Title of the Top 10 List",
        "description": "Short 1-2 sentence overview of the measurement parameters.",
        "items": [
            {{
                "rank": 1,
                "name": "Item Name",
                "key_stat": "The main quantifiable stat (e.g., '8,848 meters' or '501 kg')",
                "location_or_year": "Relevant context (e.g., 'Nepal / 1953')",
                "summary": "2-3 sentences explaining what makes this item rank here.",
                "source": "Name of public dataset or record authority",
                "source_url": "Direct URL to the cited source (e.g. the Wikipedia article or official record page)"
            }}
        ]
    }}
    """

    try:
        response = _gemini_client.models.generate_content(
            model="gemini-flash-lite-latest",
            contents=prompt,
            config={"response_mime_type": "application/json"},
        )
        data = json.loads(response.text)
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error generating list: {str(e)}"
        )

    # 4. Attach images, then cache the result
    data = enrich_with_images(data)
    with db() as conn:
        conn.execute(
            "INSERT OR REPLACE INTO top_tens (topic, data_json) VALUES (?, ?)",
            (topic_clean, json.dumps(data)),
        )
        conn.commit()

    return {"source": "live", "data": data}


# Mount static folder for frontend assets
app.mount("/", StaticFiles(directory="static", html=True), name="static")
