#!/usr/bin/env python3
"""
ingest.py – Load a SIEM dataset from HuggingFace and ingest it into your FastAPI backend.
"""

import requests
from datasets import load_dataset
import time
import json
import argparse


# -----------------------------------------------
# Configuration
# -----------------------------------------------
API_URL = "http://localhost:8000/ingest"   # your FastAPI endpoint
BATCH_SIZE = 100                           # number of records per API call
VERIFY_SSL = True                          # set False if using self-signed SSL


# -----------------------------------------------
# Function: Load dataset from HuggingFace
# -----------------------------------------------
def load_siem_dataset(name="darkknight25/Advanced_SIEM_Dataset", split="train"):
    print(f" Loading dataset: {name} ({split}) ...")
    ds = load_dataset(name, split=split)
    print(f" Loaded {len(ds)} records.")
    return ds


# -----------------------------------------------
# Function: Convert dataset record → text blob
# This is what will be embedded and stored in Chroma.
# -----------------------------------------------
def normalize_record(row):
    """
    Build a clean, human-readable text representation of the log.
    This text is what FastAPI will send to Chroma for embedding.
    """
    text = (
        f"Event ID: {row.get('event_id')}\n"
        f"Timestamp: {row.get('timestamp')}\n"
        f"Severity: {row.get('severity')}\n"
        f"Type: {row.get('event_type')}\n"
        f"Description: {row.get('description')}\n"
        f"Raw Log: {row.get('raw_log')}\n"
    )

    # Include metadata if exists
    if "advanced_metadata" in row and isinstance(row["advanced_metadata"], dict):
        meta = row["advanced_metadata"]
        for k, v in meta.items():
            text += f"{k}: {v}\n"

    return text


# -----------------------------------------------
# Function: Send batch to FastAPI
# -----------------------------------------------
def send_batch(batch):
    try:
        payload = {"texts": batch}
        response = requests.post(API_URL, json=payload, verify=VERIFY_SSL)

        if response.status_code == 200:
            print(f" Successfully ingested {len(batch)} records.")
        else:
            print(f" Error {response.status_code}: {response.text}")

    except Exception as e:
        print(f"Exception sending batch: {e}")


# -----------------------------------------------
# Main ETL Process
# -----------------------------------------------
def run_ingestion(dataset_name, split):
    ds = load_siem_dataset(dataset_name, split)

    batch = []
    total = len(ds)

    print(" Starting ingestion...")

    for i, row in enumerate(ds):

        text = normalize_record(row)
        batch.append(text)

        # If batch full → send
        if len(batch) == BATCH_SIZE:
            send_batch(batch)
            batch = []
            time.sleep(0.1)   # avoid spamming API

        # Progress display
        if (i + 1) % 1000 == 0:
            print(f"Progress: {i+1}/{total}")

    # Send final partial batch
    if batch:
        send_batch(batch)

    print(" Ingestion complete!")


# -----------------------------------------------
# CLI Arguments
# -----------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest SIEM dataset into FastAPI.")
    parser.add_argument("--dataset", default="darkknight25/Advanced_SIEM_Dataset")
    parser.add_argument("--split", default="train")
    parser.add_argument("--batch", default=100, type=int)
    parser.add_argument("--api", default=API_URL)

    args = parser.parse_args()

    BATCH_SIZE = args.batch
    API_URL = args.api

    run_ingestion(args.dataset, args.split)
