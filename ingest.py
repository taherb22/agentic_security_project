#!/usr/bin/env python3
import asyncio
import aiohttp
from datasets import load_dataset
import argparse

API_URL = "http://agent:8000/ingest"


# -----------------------------------------
# Normalize function (same as before)
# -----------------------------------------
def normalize_record(row):
    text = (
        f"Event ID: {row.get('event_id')}\n"
        f"Timestamp: {row.get('timestamp')}\n"
        f"Severity: {row.get('severity')}\n"
        f"Type: {row.get('event_type')}\n"
        f"Description: {row.get('description')}\n"
        f"Raw Log: {row.get('raw_log')}\n"
    )
    if "advanced_metadata" in row and isinstance(row["advanced_metadata"], dict):
        for k, v in row["advanced_metadata"].items():
            text += f"{k}: {v}\n"
    return text


# -----------------------------------------
# Async batch sender
# -----------------------------------------
async def send_batch(session, batch):
    try:
        async with session.post(API_URL, json={"texts": batch}) as resp:
            if resp.status == 200:
                print(f" Ingested {len(batch)} records")
            else:
                print(f" Error {resp.status}: {await resp.text()}")
    except Exception as e:
        print(f" Exception sending batch: {e}")


# -----------------------------------------
# Main async ETL
# -----------------------------------------
async def run_ingestion(dataset_name, split, batch_size):
    ds = load_dataset(dataset_name, split=split)
    total = len(ds)
    print(f"Loaded {total} records")

    tasks = []
    connector = aiohttp.TCPConnector(limit=50)  # up to 50 parallel requests
    async with aiohttp.ClientSession(connector=connector) as session:

        batch = []
        for i, row in enumerate(ds):
            batch.append(normalize_record(row))

            if len(batch) == batch_size:
                tasks.append(asyncio.create_task(send_batch(session, batch)))
                batch = []

                # Optional: print progress every 5k
                if (i + 1) % 5000 == 0:
                    print(f"Progress: {i+1}/{total}")

        # send remaining batch
        if batch:
            tasks.append(asyncio.create_task(send_batch(session, batch)))

        print("Waiting for all batches to complete...")
        await asyncio.gather(*tasks)

    print(" Ingestion complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="darkknight25/Advanced_SIEM_Dataset")
    parser.add_argument("--split", default="train")
    parser.add_argument("--batch", type=int, default=1000)
    args = parser.parse_args()

    asyncio.run(run_ingestion(args.dataset, args.split, args.batch))
