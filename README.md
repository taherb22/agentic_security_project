# Cybersecurity AI Agent

This project provides a FastAPI-based cybersecurity analysis agent powered by:

* **LLM reasoning (Groq)**
* **Structured extraction**
* **Vector search (ChromaDB)**
* **ETL ingestion pipeline**
* **Containerized deployment (Docker Compose)**

---

## 🧪 Testing the `/agent/analyze` Endpoint

You can test the analysis agent using the following `curl` command:

```bash
curl -X POST http://localhost:8000/agent/analyze \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Multiple failed SSH login attempts detected from IP 185.100.87.12",
    "top_k": 3
  }'
```

### ✔ Sample Response

The API returns:

* A unique **id** for the analysis
* A **Groq-based extraction** (attack_type, severity, explanation, IOCs)
* A **similarity search** from Chroma (top‑k similar events)
* A full **cybersecurity analysis report**

Example output:

```json
{
  "id": "9a14d942-b167-4b1b-acf2-0395c5603d06",
  "groq_extraction": {
    "attack_type": "SSH Brute Force",
    "severity": "High",
    "explanation": "Multiple failed SSH login attempts from IP 185.100.87.12 indicate a potential brute force attack targeting the SSH service.",
    "iocs": ["185.100.87.12"]
  },
  "similar": {
    "ids": [["..."]],
    "documents": [["..."]],
    "distances": [["..."]]
  },
  "final_report": "# Cybersecurity Analysis Report ... (truncated)"
}
```

---

Add more endpoints or sections as needed!
