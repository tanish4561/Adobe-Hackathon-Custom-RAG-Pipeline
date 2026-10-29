## 📘 Part 1B — Persona-Driven Document Intelligence

### 🚀 Overview

This script performs **semantic document analysis** to extract and prioritize the most relevant content from a set of PDFs based on a given **persona** and **job-to-be-done**. It supports **multilingual documents** and is designed to run fully offline with CPU-only constraints.

---

### 🧠 What It Does

* Accepts a JSON input describing:

  * A persona (user role)
  * Their job-to-be-done (task)
  * A list of documents to analyze
* Reads all relevant PDFs from a given folder
* Embeds content using a local multilingual MiniLM model
* Uses cosine similarity to rank sections most relevant to the persona's job
* Outputs a structured JSON containing:

  * Metadata
  * Ranked documents
  * Relevance scores
  * Extracted high-value sections (raw content)

---

### 📁 Folder Structure

```
project/
├── input/
│   ├── pdfs/                         # Contains all PDF documents
│   └── challenge1b_input.json        # Specifies persona, job, and document list
├── output/
│   └── result.json                   # Output JSON (can be renamed)
├── models/
│   └── multilingual_model/          # Local multilingual MiniLM model (~678MB)
├── part_1b.py                        # Main entry point for analysis
├── requirements.txt                 # All dependencies
```

---

### ⚖️ How to Run

#### 1. 📦 Install dependencies

```bash
pip install -r requirements.txt
```

Ensure PyMuPDF (`fitz`) and `sentence-transformers` are installed, and Python ≥ 3.8 is recommended.

#### 2. 📂 Prepare Input

* Place your PDFs inside `input/pdfs/`
* Place your input config as `input/challenge1b_input.json`

Example `challenge1b_input.json`:

```json
{
  "challenge_info": {
    "challenge_id": "round_1b_002",
    "test_case_name": "travel_planner"
  },
  "persona": "Travel Planner",
  "job": "Plan a trip of 4 days for a group of 10 college friends.",
  "documents": [
    { "filename": "South of France - Cities.pdf", "title": "South of France - Cities" },
    { "filename": "South of France - Cuisine.pdf", "title": "South of France - Cuisine" }
  ]
}
```

#### 3. ▶️ Run the script

```bash
python part_1b.py \
  --input_json input/challenge1b_input.json \   
  --pdf_folder input/pdfs \
  --output output/result.json \
  --multilingual
```

* `--multilingual` ensures the multilingual MiniLM model is used.
* Top 10 most relevant documents/sections are extracted by default (can be adjusted with `--top_n`).

---

### ✅ Output Format (result.json)

```json
{
  "challenge_info": { ... },
  "persona": "Travel Planner",
  "job": "Plan a trip of 4 days...",
  "relevant_sections": [
    {
      "document": "South of France - Cuisine.pdf",
      "title": "South of France - Cuisine",
      "relevance_score": 0.823,
      "content": "A Culinary Journey..."
    }
  ],
  "metadata": {
    "input_documents": [...],
    "timestamp": "2025-07-23T00:00:00Z",
    "model_version": "1.0.0"
  }
}
```

---

### 💡 Notes

* Supports **any persona and job** dynamically.
* Fully **offline-compatible** — no internet is used.
* Multilingual support via `paraphrase-multilingual-MiniLM-L12-v2` saved locally.
* Designed to handle academic, business, or educational PDFs alike.

---
