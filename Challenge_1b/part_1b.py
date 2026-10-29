import os
import sys
import json
import argparse
import datetime
import numpy as np
from tqdm import tqdm
import torch
import fitz  # PyMuPDF
from sentence_transformers import SentenceTransformer

DEFAULT_MODEL_DIR = "./models/multilingual_model/"

class SentenceEmbedder:
    def __init__(self, model_dir=DEFAULT_MODEL_DIR):
        if os.path.isfile(os.path.join(model_dir, 'config_sentence_transformers.json')):
            self.st_model = SentenceTransformer(model_dir)
        else:
            print(f"ERROR: Model directory '{model_dir}' not found or not a SentenceTransformer model.")
            sys.exit(1)

    def encode(self, texts, batch_size=16):
        if isinstance(texts, str):
            texts = [texts]
        return self.st_model.encode(texts, batch_size=batch_size, convert_to_numpy=True, show_progress_bar=False)

def cosine_similarity(a, b):
    a = a / (np.linalg.norm(a, axis=-1, keepdims=True) + 1e-8)
    b = b / (np.linalg.norm(b, axis=-1, keepdims=True) + 1e-8)
    return np.dot(a, b.T)

def extract_pdf_text(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text() + "\n"
    return text

def analyze_collection(input_json_path, pdf_folder, output_path=None, model_dir=DEFAULT_MODEL_DIR, top_n=10):
    with open(input_json_path, 'r', encoding='utf-8') as f:
        input_data = json.load(f)

    documents = input_data.get('documents', [])
    persona = input_data.get('persona', {}).get('role', '')
    job = input_data.get('job_to_be_done', {}).get('task', '')
    challenge_info = input_data.get('challenge_info', {})

    if not documents or not persona or not job:
        print("Input JSON must contain 'documents', 'persona', and 'job_to_be_done'.")
        return None

    embedder = SentenceEmbedder(model_dir)
    query = f"Persona: {persona}. Job: {job}"
    query_emb = embedder.encode([query])[0]

    section_entries = []
    section_texts = []
    doc_names = []

    for doc in tqdm(documents, desc="Processing documents"):
        filename = doc.get('filename')
        title = doc.get('title', '')
        pdf_path = os.path.join(pdf_folder, filename)
        if not os.path.isfile(pdf_path):
            print(f"PDF file not found: {pdf_path}")
            continue

        text = extract_pdf_text(pdf_path)
        section_entries.append({
            'document': filename,
            'title': title,
            'raw_text': text
        })
        section_texts.append(f"{title}. {text}")
        doc_names.append(filename)

    if not section_entries:
        print("No valid PDFs found for analysis.")
        return None

    section_embs = embedder.encode(section_texts)
    sims = cosine_similarity(section_embs, query_emb[None, :]).flatten()
    ranked_idx = np.argsort(sims)[::-1][:top_n]

    relevant_sections = []
    for rank, idx in enumerate(ranked_idx, 1):
        entry = section_entries[idx]
        relevant_sections.append({
            'document': entry['document'],
            'title': entry['title'],
            'relevance_score': float(sims[idx]),
            'content': entry['raw_text'][:1000]
        })

    output = {
        'challenge_info': challenge_info,
        'persona': persona,
        'job': job,
        'relevant_sections': relevant_sections,
        'metadata': {
            'input_documents': doc_names,
            'timestamp': datetime.datetime.utcnow().isoformat() + 'Z',
            'model_version': '1.0.0',
        }
    }

    if output_path:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output, f, ensure_ascii=False, indent=2)
    return output
def main():
    parser = argparse.ArgumentParser(description="Analyze PDF collections for relevant content based on persona and job. Fully offline.")
    parser.add_argument('--input_json', type=str, help='Path to input JSON file (challenge1b_input.json)')
    parser.add_argument('--pdf_folder', type=str, help='Path to folder containing PDFs')
    parser.add_argument('--output', type=str, default="output/challenge1b_output.json", help='Output JSON file')
    parser.add_argument('--top_n', type=int, default=10, help='Number of top sections to return')
    parser.add_argument('--model_dir', type=str, default=DEFAULT_MODEL_DIR, help='Path to local model directory')
    parser.add_argument('--multilingual', action='store_true', help='Use multilingual model')
    parser.add_argument('--collection_path', type=str, help='If specified, overrides input_json and pdf_folder with collection_path layout')

    args = parser.parse_args()

    # If collection_path is provided, automatically resolve paths
    if args.collection_path:
        args.input_json = os.path.join(args.collection_path, "challenge1b_input.json")
        args.pdf_folder = os.path.join(args.collection_path, "pdfs")

    if not args.input_json or not args.pdf_folder:
        parser.error("You must specify --input_json and --pdf_folder, or --collection_path.")

    model_dir = args.model_dir
    if args.multilingual:
        model_dir = './models/multilingual_model/'

    analyze_collection(args.input_json, args.pdf_folder, args.output, model_dir, args.top_n)


if __name__ == '__main__':
    main()
