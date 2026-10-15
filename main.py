import json
import os
import sqlite3
from contextlib import contextmanager

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import google.generativeai as genai

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

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


class QueryRequest(BaseModel):
    topic: str


@app.get("/api/health")
def health():
    return {"status": "ok", "generator_configured": bool(GEMINI_API_KEY)}


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
        return {"source": "cache", "data": json.loads(row[0])}

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
                "source": "Name of public dataset or record authority"
            }}
        ]
    }}
    """

    try:
        model = genai.GenerativeModel("gemini-2.5-flash")
        response = model.generate_content(
            prompt, generation_config={"response_mime_type": "application/json"}
        )
        data = json.loads(response.text)
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error generating list: {str(e)}"
        )

    # 4. Cache the result
    with db() as conn:
        conn.execute(
            "INSERT OR REPLACE INTO top_tens (topic, data_json) VALUES (?, ?)",
            (topic_clean, json.dumps(data)),
        )
        conn.commit()

    return {"source": "live", "data": data}


# Mount static folder for frontend assets
app.mount("/", StaticFiles(directory="static", html=True), name="static")
