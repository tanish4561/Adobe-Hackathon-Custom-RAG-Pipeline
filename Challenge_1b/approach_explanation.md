    ## 📘 Approach Explanation — Challenge 1B

In this solution, we built a multilingual, offline-compatible system that extracts and ranks relevant PDF sections based on a persona and job-to-be-done context.

### 🎯 Objective
To simulate a document analyst that identifies the most contextually relevant sections from a set of PDFs using only CPU, with no internet access.

---

### 🧱 Architecture

- **Input**: A `challenge1b_input.json` describing:
  - A persona (e.g., Travel Planner)
  - A job (e.g., Plan a 4-day trip)
  - A list of PDF documents with metadata

- **Preprocessing**:
  - PDFs are loaded using **PyMuPDF**
  - Each document’s text is extracted and paired with its title

- **Semantic Matching**:
  - A multilingual **MiniLM** model (from `sentence-transformers`) encodes both:
    - The persona + job query
    - The document sections
  - Cosine similarity scores rank how relevant each section is to the job

- **Output**:
  - A JSON with:
    - Ranked relevant sections
    - Document metadata
    - Extraction timestamp

---

### ⚙️ Technical Highlights

- Fully **offline**, Docker-compatible, **CPU-only** solution
- Processes 3–10 PDFs in under 60 seconds
- Supports **multilingual documents**
- Pretrained model embedded inside Docker (≤ 1GB)

---

### 📌 Key Assumptions

- All PDFs reside in `input/pdfs/`
- Input JSON follows Adobe’s specification
- Output is written to `output/challenge1b_output.json`

---

### 🧪 Testing & Performance

The system was tested on academic, travel, and educational content. Accuracy was validated by ensuring the top-ranked sections consistently matched the persona’s goal.

---

### 🧠 Future Enhancements

- Fine-tuning model embeddings using labeled domain-specific data
- Section chunking for long documents
- GUI for easier persona-job-PDF configuration

---

### ✅ Summary

This pipeline emulates a personalized research assistant that "reads" a document the way a human would — by aligning its contents to the intent of the reader.
