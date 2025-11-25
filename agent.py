#!/usr/bin/env python3
"""
agent.py
Run with: uvicorn agent:app --host 0.0.0.0 --port 8000

"""

import os
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
groq_client = Groq(api_key=GROQ_API_KEY)



# --- Vector Store Setup (Single Source of Truth) ---
try:
    logger.info("Initializing Vector Store...")
    chroma_client = chromadb.Client(Settings(
        chroma_db_impl="duckdb+parquet",
        persist_directory="./chroma_data"
    ))
    collection = chroma_client.get_or_create_collection(
        name="cyber_events",
        metadata={"hnsw:space": "cosine"}
    )
    # Load model once at startup
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    logger.info("Vector Store Ready.")
except Exception as e:
    logger.error(f"Critical Error: {e}")
    model = None
    collection = None

# --- Models ---
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

# --- Core Functions ---

def groq_extract_json(text: str):
    prompt = f"""
    Extract cyber fields (attack_type, severity, explanation, iocs) as JSON.
    Log: {text}
    """
    try:
        resp = groq_client.chat.completions.create(
            model="openai/gpt-oss-120b",
            messages=[{"role": "system", "content": "JSON only."}, {"role": "user", "content": prompt}],
            temperature=0.0,
            response_format={"type": "json_object"}
        )
        return json.loads(resp.choices[0].message.content)
    except Exception as e:
        return {"error": str(e)}

def groq_report_stream(prompt: str):
    try:
        stream = groq_client.chat.completions.create(
            model="openai/gpt-oss-120b",
            messages=[{"role": "user", "content": prompt}],
            max_completion_tokens=4096,
            stream=True
        )
        for chunk in stream:
            yield chunk.choices[0].delta.content or ""
    except Exception as e:
        yield f"Error: {e}"

# --- Endpoints ---

@app.post("/ingest")
async def ingest_logs(payload: IngestRequest):
    """
    Endpoint used by the ETL Pipeline to load data.
    """
    if not model or not collection:
        raise HTTPException(status_code=503, detail="DB not initialized")
    
    try:
        # Batch processing
        ids = [str(uuid.uuid4()) for _ in payload.texts]
        # Encode all at once (faster than loop)
        embeddings = model.encode(payload.texts).tolist()
        collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=payload.texts
        )
        # Persist to disk
        if hasattr(chroma_client, 'persist'):
            chroma_client.persist()
            
        return {"status": "success", "count": len(ids)}
    except Exception as e:
        logger.error(f"Ingest failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/agent/analyze")
async def analyze(req: AnalyzeRequest):
    # 1. Groq Extraction
    extraction = groq_extract_json(req.text)
    
    # 2. Vector Search
    query_vec = model.encode(req.text).tolist()
    results = collection.query(query_embeddings=[query_vec], n_results=req.top_k)
    
    # 3. Format Context
    context = []
    if results['documents']:
        for i, doc in enumerate(results['documents'][0]):
            context.append(f"Similar Log: {doc}")
            
    # 4. Generate Report
    prompt = f"Analyze this log based on context:\nLog: {req.text}\nContext: {context}\nExtraction: {extraction}"
    
    # Non-streaming wrapper for simplicity in this example response
    report_gen = "".join(list(groq_report_stream(prompt)))
    
    return {
        "id": str(uuid.uuid4()),
        "groq_extraction": extraction,
        "similar": results,
        "final_report": report_gen
    }