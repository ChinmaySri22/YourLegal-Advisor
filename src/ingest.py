# src/ingest.py

import os
import sys
import re
import pickle
from pathlib import Path

import pandas as pd
import numpy as np
import faiss
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer

def split_into_chunks(text, max_chars=1000, overlap=200):
    """
    Split `text` into chunks of ~max_chars, overlapping by `overlap`, 
    but break on sentence boundaries.
    """
    # Simple sentence splitter
    sentences = re.split(r'(?<=[\.!?])\s+', text.strip())
    chunks = []
    current = ""
    for sent in sentences:
        if len(current) + len(sent) + 1 <= max_chars:
            current = (current + " " + sent).strip()
        else:
            if current:
                chunks.append(current)
            # start new chunk (include overlap tail)
            tail = current[-overlap:] if overlap < len(current) else current
            current = (tail + " " + sent).strip()
    if current:
        chunks.append(current)
    return chunks

def extract_section_label(text):
    """
    Look for "Section <number>" or "[s <number>]" in text.
    Returns the first number found, else None.
    """
    # common patterns in IPC docs
    patterns = [
        r'section\s+(\d+)',         # Section 306
        r'\[s\.?\s*(\d+)\]',        # [s 309]
        r'\[\s*(\d+)\]'             # [306]
    ]
    for pat in patterns:
        m = re.search(pat, text, flags=re.IGNORECASE)
        if m:
            return m.group(1)
    return None

def main():
    project_root = Path(__file__).parent.parent
    data_dir     = project_root / "data"
    pdf_path     = data_dir / "ipc_law.pdf"
    if not pdf_path.exists():
        print(f"❌ {pdf_path} not found.")
        sys.exit(1)

    # 1. Read PDF
    reader = PdfReader(str(pdf_path))
    print(f"✅ Loaded PDF with {len(reader.pages)} pages.")

    records = []
    chunks = []

    # 2. Iterate pages, split & label
    for page_num, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        if not text.strip():
            continue

        page_chunks = split_into_chunks(text, max_chars=1000, overlap=200)
        for chunk in page_chunks:
            label = extract_section_label(chunk) or str(page_num)
            records.append({"text": chunk, "section": label})
            chunks.append(chunk)

    print(f"✅ Extracted {len(records)} labeled text chunks.")

    # 3. Save CSV for supervised training
    df = pd.DataFrame(records)
    csv_path = data_dir / "ipc_sections_dataset.csv"
    df.to_csv(csv_path, index=False)
    print(f"🚀 Wrote dataset to {csv_path} (columns: text, section)")

    # 4. Embed all chunks
    model_name = "all-MiniLM-L6-v2"
    print(f"⏳ Loading embedder: {model_name}")
    embedder = SentenceTransformer(model_name)
    embeddings = embedder.encode(chunks, show_progress_bar=True)
    embeddings = np.array(embeddings)
    dim = embeddings.shape[1]
    print(f"✅ Generated embeddings shape: {embeddings.shape}")

    # 5. Build FAISS index
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    print(f"✅ FAISS index contains {index.ntotal} vectors.")

    # 6. Persist index + original chunk texts
    models_dir = project_root / "models" / "Retrival"
    models_dir.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(models_dir / "index.faiss"))
    with open(models_dir / "index.pkl", "wb") as f:
        pickle.dump(chunks, f)
    print(f"🚀 Saved FAISS index and chunks to {models_dir}")

if __name__ == "__main__":
    main()
