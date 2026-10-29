
import os
import json
import time
import fitz  # PyMuPDF
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
import hashlib
import csv
import argparse
from sentence_transformers import SentenceTransformer

# Multilingual support
USE_MULTILINGUAL = True  # Set to True to use multilingual model by default
MULTILINGUAL_MODEL_DIR = './models/multilingual_model'
ENGLISH_MODEL_DIR = './models/fine_tuned_heading_model'

# CLI support for multilingual
parser = argparse.ArgumentParser(description="PDF Outline Extractor (Part 1A) with optional multilingual support.")
parser.add_argument('--multilingual', action='store_true', help='Use multilingual heading classifier (if available)')
args, unknown = parser.parse_known_args()
if args.multilingual:
    USE_MULTILINGUAL = True

# Model loading logic (always runs)
model_dir = MULTILINGUAL_MODEL_DIR if USE_MULTILINGUAL else ENGLISH_MODEL_DIR
print("Checking for SentenceTransformer model at:", os.path.join(model_dir, 'config_sentence_transformers.json'))
print("Exists?", os.path.isfile(os.path.join(model_dir, 'config_sentence_transformers.json')))
if os.path.isfile(os.path.join(model_dir, 'config_sentence_transformers.json')):
    print(f"[INFO] Loading SentenceTransformer model from {model_dir}")
    st_model = SentenceTransformer(model_dir)
    USE_SENTENCE_TRANSFORMER = True
else:
    print(f"[INFO] Loading HuggingFace sequence classification model from {model_dir}")
    model = AutoModelForSequenceClassification.from_pretrained(model_dir, local_files_only=True)
    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    model.eval()
    USE_SENTENCE_TRANSFORMER = False

USE_LLM = True  # Set to False for ultra-fast, rule-based heading extraction

TRAIN_MODE = True # Set to True to collect training data for model improvement
TRAIN_DATA_PATH = './training/training_data.csv'

def classify_heading_semantic(texts):
    # Use semantic similarity to heading templates
    all_templates = []
    template_labels = []
    for label, templates in HEADING_TEMPLATES:
        for t in templates:
            all_templates.append(t)
            template_labels.append(label)
    template_embs = st_model.encode(all_templates, convert_to_numpy=True)
    text_embs = st_model.encode(texts, convert_to_numpy=True)
    import numpy as np
    results = []
    for emb in text_embs:
        sims = np.dot(template_embs, emb) / (np.linalg.norm(template_embs, axis=1) * np.linalg.norm(emb) + 1e-8)
        best_idx = np.argmax(sims)
        best_label = template_labels[best_idx]
        # Threshold: if similarity is too low, label as None
        if sims[best_idx] < 0.45:
            results.append("None")
        else:
            results.append(best_label)
    return results

def classify_heading_llm(texts):
    if USE_SENTENCE_TRANSFORMER:
        return classify_heading_semantic(texts)
    batch = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**batch)
        predictions = torch.argmax(outputs.logits, dim=1)
    label_map = {0: "H1", 1: "H2", 2: "H3", 3: "None"}
    return [label_map.get(p.item(), "None") for p in predictions]

def is_likely_heading(text, font_size):
    # Heuristic rules for likely headings
    if len(text) > 80 or len(text) < 2:
        return False
    lower = text.lower()
    if any(x in lower for x in ["copyright", "all rights reserved", "orcid", "email", "@", "doi", "page "]):
        return False
    if lower.startswith(("tip:", "note:", "warning:", "caution:", "important:", "sidebar:", "tooltip:")):
        return False
    if lower.startswith(("margin note", "side note", "footnote")):
        return False
    if text.replace(" ", "").isdigit():
        return False
    if lower.startswith("table ") or lower.startswith("figure "):
        return False
    if sum(c.isalpha() for c in text) / max(1, len(text)) < 0.5:
        return False
    if font_size > 13 or text.isupper() or text.istitle() or text[:2].isdigit():
        return True
    return False

def get_pdf_hash(pdf_path):
    # Compute a hash of the PDF file for cache key
    hasher = hashlib.md5()
    with open(pdf_path, 'rb') as f:
        while True:
            chunk = f.read(8192)
            if not chunk:
                break
            hasher.update(chunk)
    return hasher.hexdigest()


def load_font_cache(cache_path):
    if os.path.exists(cache_path):
        with open(cache_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}


def save_font_cache(cache_path, cache):
    with open(cache_path, 'w', encoding='utf-8') as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)


def extract_headings_and_content(pdf_path, font_cache_global=None, cache_key=None, cache_path=None):
    doc = fitz.open(pdf_path)
    data = []
    candidates = []
    pages = []
    blocks_list = []
    font_cache = font_cache_global.get(cache_key, {}) if font_cache_global and cache_key else {}
    font_cache_updated = False
    for page_num in range(len(doc)):
        page = doc[page_num]
        blocks = page.get_text("blocks")
        blocks.sort(key=lambda b: (b[1], b[0]))
        # Cache font sizes for this page
        if page_num not in font_cache:
            font_cache[page_num] = set()
            for block in blocks:
                font_size = block[3] - block[1]
                font_cache[page_num].add(font_size)
            font_cache_updated = True
        for block in blocks:
            text = block[4].strip()
            if not text:
                continue
            font_size = block[3] - block[1]
            if is_likely_heading(text, font_size):
                candidates.append(text)
                pages.append(page_num + 1)
                blocks_list.append(block)
    # Save updated font cache if needed
    if font_cache_global is not None and cache_key and font_cache_updated:
        font_cache_global[cache_key] = {k: list(v) for k, v in font_cache.items()}
        save_font_cache(cache_path, font_cache_global)
    # Training mode: save all candidates and their predicted labels
    if TRAIN_MODE:
        rows = []
        if USE_LLM and candidates:
            labels = classify_heading_llm(candidates)
        else:
            labels = []
            for idx, text in enumerate(candidates):
                font_size = blocks_list[idx][3] - blocks_list[idx][1]
                if font_size > 20:
                    labels.append("H1")
                elif font_size > 16:
                    labels.append("H2")
                else:
                    labels.append("H3")
        for idx, text in enumerate(candidates):
            rows.append({
                'pdf': os.path.basename(pdf_path),
                'page': pages[idx],
                'text': text,
                'label': labels[idx]
            })
        # Write to CSV (append mode)
        file_exists = os.path.exists(TRAIN_DATA_PATH)
        with open(TRAIN_DATA_PATH, 'a', newline='', encoding='utf-8') as csvfile:
            fieldnames = ['pdf', 'page', 'text', 'label']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            if not file_exists:
                writer.writeheader()
            for row in rows:
                writer.writerow(row)
    if not USE_LLM:
        for idx, text in enumerate(candidates):
            font_size = blocks_list[idx][3] - blocks_list[idx][1]
            if font_size > 20:
                level = "H1"
            elif font_size > 16:
                level = "H2"
            else:
                level = "H3"
            data.append({
                "level": level,
                "text": text,
                "page": pages[idx],
                "content": ""
            })
        return data
    if candidates:
        labels = classify_heading_llm(candidates)
        current_heading = None
        current_content = ""
        current_level = ""
        for idx, label in enumerate(labels):
            text = candidates[idx]
            page_num = pages[idx]
            if label != "None":
                if current_heading:
                    data.append({
                        "level": current_level,
                        "text": current_heading,
                        "page": page_num,
                        "content": current_content.strip()
                    })
                current_heading = text
                current_level = label
                current_content = ""
            else:
                current_content += " " + text
        if current_heading:
            data.append({
                "level": current_level,
                "text": current_heading,
                "page": page_num,
                "content": current_content.strip()
            })
    return data

def postprocess_headings(data):
    # Collapse adjacent headings of the same level and validate heading hierarchy
    if not data:
        return data
    processed = []
    last = None
    for entry in data:
        if last and entry['level'] == last['level'] and entry['page'] == last['page']:
            # Merge fragments of the same heading
            last['text'] += ' ' + entry['text']
            last['content'] += ' ' + entry.get('content', '')
        else:
            # Validate heading hierarchy: H3 should not follow H1 directly
            if last and last['level'] == 'H1' and entry['level'] == 'H3':
                entry['level'] = 'H2'  # Downgrade to H2
            processed.append(entry)
            last = entry
    return processed

def extract_pdf_bookmarks(pdf_path):
    doc = fitz.open(pdf_path)
    toc = doc.get_toc()
    outline = []
    for item in toc:
        level, text, page = item
        outline.append({
            "level": f"H{level}",
            "text": text,
            "page": page,
            "content": ""  # Bookmarks don't have content, but keep format
        })
    return outline

def build_outline(pdf_path):
    cache_path = os.path.join("./cache", "font_cache.json")
    font_cache_global = load_font_cache(cache_path)
    cache_key = get_pdf_hash(pdf_path)
    bookmarks = extract_pdf_bookmarks(pdf_path)
    if bookmarks:
        title = bookmarks[0]['text'] if bookmarks else "Document"
        return {"title": title, "outline": bookmarks}
    extracted = extract_headings_and_content(pdf_path, font_cache_global, cache_key, cache_path)
    extracted = postprocess_headings(extracted)
    title = extracted[0]['text'] if extracted else "Document"
    return {"title": title, "outline": extracted}

def process_pdf_file(filename, input_dir, output_dir):
    if filename.endswith(".pdf"):
        pdf_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, filename.replace(".pdf", ".json"))
        outline = build_outline(pdf_path)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(outline, f, ensure_ascii=False, indent=2)


def main():
    input_dir = "./input"
    output_dir = "./output"
    os.makedirs(output_dir, exist_ok=True)
    start_time = time.time()
    pdf_files = [f for f in os.listdir(input_dir) if f.endswith(".pdf")]
    with ThreadPoolExecutor() as executor:
        for _ in executor.map(lambda f: process_pdf_file(f, input_dir, output_dir), pdf_files):
            pass
    end_time = time.time()  # End timer
    print(f"Execution Time: {end_time - start_time:.2f} seconds")

if __name__ == "__main__":
    main()
