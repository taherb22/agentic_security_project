#!/usr/bin/env python3
"""
agent.py
Run with: uvicorn agent:app --host 0.0.0.0 --port 8000
"""
import os
for key in ["HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy"]:
    if key in os.environ:
        del os.environ[key]
# Monkey patch httpx to ignore proxies argument
import httpx
_original_client_init = httpx.Client.__init__
def patched_client_init(self, *args, **kwargs):
    kwargs.pop("proxies", None)  # remove unsupported argument
    return _original_client_init(self, *args, **kwargs)
httpx.Client.__init__ = patched_client_init        
import json
import uuid
import logging
from typing import Any, Dict, List, Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
# Vector DB + Embeddings
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
# Groq LLM
from groq import Groq
# Env variables
from dotenv import load_dotenv
# ------------------------------------------------------
# Initialization
# ------------------------------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("CyberAgent")

app = FastAPI(title="Cyber Agent & Vector Store")

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    raise ValueError("Missing GROQ_API_KEY — add it to your .env")

# Initialize Groq client
groq_client = Groq(api_key=GROQ_API_KEY,  http_client=None)


from chromadb import PersistentClient
from sentence_transformers import SentenceTransformer


try:
    logger.info("Initializing Vector Store...")

    # New Chroma persistent client (0.5+)
    chroma_client = PersistentClient(path="./chroma_data")

    # Create or load collection
    collection = chroma_client.get_or_create_collection(
        name="cyber_events",
        metadata={"hnsw:space": "cosine"}   # still allowed
    )

    # Load embedding model
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    logger.info("Vector Store Ready.")

except Exception as e:
    logger.error(f"Critical Error in Vector Store Initialization: {e}")
    model = None
    collection = None



# ------------------------------------------------------
# Models
# ------------------------------------------------------
class IngestRequest(BaseModel):
    texts: List[str]

class AnalyzeRequest(BaseModel):
    text: str
    top_k: int = 3

class AnalyzeResponse(BaseModel):
    id: str
    groq_extraction: Dict[str, Any]
    similar: Dict[str, Any]
    final_report: Optional[str]


# ------------------------------------------------------
# Functions
# ------------------------------------------------------

def groq_extract_json(text: str):
    """
    Extract structured JSON from a log using Groq.
    Groq SDK does NOT support response_format — so we enforce JSON in the prompt.
    """
    prompt = f"""
You are a cyber threat log parser.
Extract strictly the following fields in JSON format only:

- attack_type
- severity
- explanation
- iocs (list)

Log:
{text}

Return only valid JSON.
"""

    try:
        resp = groq_client.chat.completions.create(
            model="openai/gpt-oss-20b",
            messages=[
                {"role": "system", "content": "Always output valid JSON, no explanations."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.0
        )

        raw = resp.choices[0].message.content.strip()   

        return json.loads(raw)

    except Exception as e:
        return {"error": str(e), "raw": raw if "raw" in locals() else None}


def groq_report_stream(prompt: str):
    """
    Streaming report generation.
    """
    try:
        stream = groq_client.chat.completions.create(
            model="openai/gpt-oss-20b",
            messages=[{"role": "user", "content": prompt}],
            max_completion_tokens=2048,
            stream=True
        )

        for chunk in stream:
            if chunk.choices and chunk.choices[0].delta and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content

    except Exception as e:
        yield f"Error: {e}"


# ------------------------------------------------------
# Endpoints
# ------------------------------------------------------

from fastapi import BackgroundTasks

@app.post("/ingest")
async def ingest_logs(payload: IngestRequest, background: BackgroundTasks):
    if not model or not collection:
        raise HTTPException(status_code=503, detail="Vector DB not initialized")

    # Schedule ingestion in background
    background.add_task(process_batch, payload.texts)

    return {"status": "queued", "count": len(payload.texts)}


def process_batch(texts):
    try:
        ids = [str(uuid.uuid4()) for _ in texts]

        # Run embedding in a dedicated CPU thread
        embeddings = model.encode(texts).tolist()

        # Write to Chroma (blocking, so kept off the event loop)
        collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=texts
        )

        if hasattr(chroma_client, "persist"):
            chroma_client.persist()

        logger.info(f"Ingested {len(texts)} logs")

    except Exception as e:
        logger.error(f"Batch processing failed: {e}")


@app.post("/agent/analyze")
async def analyze(req: AnalyzeRequest):

    # 1 — Structured extraction
    extraction = groq_extract_json(req.text)

    # 2 — Vector search
    query_vec = model.encode(req.text).tolist()
    results = collection.query(
        query_embeddings=[query_vec],
        n_results=req.top_k
    )

    # 3 — Build context
    context = []
    if results.get("documents"):
        for doc in results["documents"][0]:
            context.append(f"Similar Log: {doc}")

    # 4 — Full report
    prompt = f"""
Analyze the following cybersecurity event.

Log:
{req.text}

Extraction:
{json.dumps(extraction, indent=2)}

Context:
{context}

Generate a detailed cybersecurity analysis report.
"""

    report = "".join(list(groq_report_stream(prompt)))

    return {
        "id": str(uuid.uuid4()),
        "groq_extraction": extraction,
        "similar": results,
        "final_report": report
    }

@app.get("/health")
def health():
    return {"status": "ok"}