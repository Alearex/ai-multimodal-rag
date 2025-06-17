# AI Multimodal RAG

This project demonstrates a lightweight multimodal retrieval-augmented generation (RAG) pipeline. It can ingest text, images and audio, encode them into a FAISS vector store and query the Mixtral model hosted on Together.

## Requirements

Install the Python dependencies listed in `requirements.txt`:

```bash
pip install -r requirements.txt
```

Each dependency is pinned to a specific version for reproducibility.

### Main Python packages

- **streamlit** – web interface
- **faiss-cpu** – vector store backend
- **sentence-transformers** – text embeddings
- **open_clip_torch** – CLIP model for images
- **faster-whisper** – fast audio transcription
- **pillow** – image handling
- **transformers** – BLIP image captioning
- **python-dotenv** – environment configuration
- **requests** – Mixtral API calls
- **PyMuPDF** – PDF text extraction
