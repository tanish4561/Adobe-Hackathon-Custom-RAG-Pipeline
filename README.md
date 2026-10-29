# Adobe India Hackathon - PDF Processing Solution

## Project Overview
This repository contains two independent solutions for PDF document processing:

### Challenge 1A: PDF Outline Extraction
Located in `/Challenge_1a/`
- Automatically extracts headings and subheadings from PDF documents
- Creates hierarchical document outlines
- Supports multilingual documents
- CPU-only processing (no GPU required)
- Memory efficient (model size < 200MB)
- Fast processing (≤ 10 seconds for 50-page PDF)

### Challenge 1B: Persona-Based Content Filter
Located in `/Challenge_1b/`
- Filters and ranks content based on user personas and job roles
- Takes JSON input with document sections
- Provides relevance scores and summaries
- Supports multilingual content
- CPU-only processing (no GPU required)
- Memory efficient (model size < 1GB)
- Fast processing (≤ 60 seconds for 3-5 documents)

## Project Structure
```
.
├── Challenge_1a/           # PDF Outline Extraction
│   ├── input/             # Place PDF files here
│   ├── output/            # Generated JSON outlines
│   ├── models/            # Pre-trained models
│   ├── training/          # Training code (optional)
│   └── ...               # Other component files
│
└── Challenge_1b/          # Content Filter
    ├── input/            # Place JSON files here
    ├── output/           # Filtered results
    ├── models/           # Pre-trained models
    └── ...              # Other component files
```

## Getting Started
Each challenge is completely independent. Please refer to the individual README files in each challenge directory for specific setup and usage instructions:
- [Challenge 1A - PDF Outline Extraction](./Challenge_1a/README.md)
- [Challenge 1B - Persona-Based Content Filter](./Challenge_1b/README.md) 