# AI Legal Advisor

An AI-powered Legal Assistant designed to help users easily navigate and understand sections of the Indian Penal Code (IPC). This tool enables users to query the IPC in natural language, retrieve relevant sections, and receive clear summaries. The system also features an emotion-aware assistant to provide a more human-friendly user experience.

## Key Features

* 🔍 **Semantic Search**: Retrieve IPC sections based on natural language queries.
* 📖 **Section Classification**: Predicts the most relevant IPC section from a user query.
* 📈 **Summarization**: Provides concise summaries of retrieved legal sections.
* 🧐 **Emotion-aware Assistant**: Detects user emotions in queries to personalize responses.

## Tech Stack

* **Python**
* **PyPDF2** - Text extraction from IPC PDFs
* **SentenceTransformers (SBERT)** - Text embedding
* **FAISS** - Approximate nearest-neighbor search
* **DistilBERT** - IPC section classification
* **BiLSTM + Attention** - Emotion detection
* **TextRank** - Extractive summarization

## How It Works

1. **Preprocessing**: IPC text is extracted, tokenized into overlapping chunks, and embedded using SentenceTransformer.
2. **Indexing**: Embeddings are stored in a FAISS index for fast similarity search.
3. **Query Handling**:

   * Semantic search identifies top-matching IPC chunks.
   * Classification model predicts the most relevant IPC section.
   * Summarizer extracts key sentences from the section.
   * Emotion-aware module personalizes the response tone.

## Setup Instructions

### Prerequisites

* Python 3.8+
* pip

### Installation

```bash
git clone https://github.com/yourusername/ai-legal-advisor.git
cd ai-legal-advisor
pip install -r requirements.txt
```

### Running the Project

```bash
# Run the main app (example)
python app.py
```

### Files Overview

* `ipc_text_processing.py` - Preprocesses IPC PDF.
* `embedding_search.py` - Builds and queries FAISS index.
* `section_classifier.py` - IPC section prediction.
* `summarizer.py` - Generates summaries.
* `emotion_detector.py` - Detects user emotion.
* `app.py` - Main app entry point.

## Example Usage

```text
User: "What is the punishment for assaulting a public servant?"

AI Legal Advisor:
- Relevant Section: IPC Section 353
- Summary: Whoever assaults or uses criminal force to deter a public servant from discharge of his duty shall be punished with imprisonment of up to 2 years, fine, or both.
- Detected Emotion: Neutral
```

## Known Limitations

* IPC section prediction accuracy can be improved (\~32% accuracy).
* Summarization sometimes includes redundant sentences.
* Emotion detection is limited to 6 classes.

## Future Work

* Expand to include other legal codes (CRPC, Evidence Act, etc.).
* Improve section classification with focal loss.
* Add abstractive summarization.
* Build a web-based UI.

## License

This project is licensed under the MIT License.

## Acknowledgements

* HuggingFace Transformers & Datasets
* FAISS by Facebook AI Research
* Indian Kanoon for IPC dataset inspiration

---

Feel free to star the repo if you find it useful!

---

*Disclaimer: This tool is for informational purposes only and should not be considered a substitute for professional legal advice.*
