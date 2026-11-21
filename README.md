# Spotify Architecture — Hybrid RAG + Knowledge Graph RAG

This project builds two indexes from `spotify_web_app_architecture.pdf`:
- **Vector RAG:** PDF chunks embedded with OpenAI `text-embedding-3-small` and stored locally in FAISS.
- **Knowledge Graph RAG:** LLM-extracted entities and typed relationships stored in Neo4j.
- **Hybrid answer:** Retrieves vector passages and graph neighbourhoods, then asks GPT-4o-mini to answer using both contexts.

## 1. Prerequisites
- Python 3.11 recommended
- OpenAI API key
- Neo4j AuraDB database (URI, username, password)

## 2. Setup (Windows PowerShell)
```powershell
cd "C:\Users\vdiwaka\Downloads\Gen AI program\spotify_hybrid_rag"
py -3.11 -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## 3. Configure environment
```powershell
Copy-Item .env.example .env
```
Edit `.env` and enter your OpenAI and Neo4j credentials. Do not commit `.env`.

Copy the source PDF to `data\spotify_web_app_architecture.pdf`.

## 4. Build both indexes
```powershell
python ingestion.py
```
This creates `storage/faiss_index` and loads extracted graph nodes/relationships into Neo4j.

## 5. Run the application
Terminal 1:
```powershell
uvicorn backend:app --reload
```
Terminal 2:
```powershell
streamlit run frontend.py
```
Open the Streamlit URL shown in the terminal.

## 6. Example questions
- What does the Playback & Session Service manage?
- Which technologies are used for search and autocomplete?
- Explain the track upload and media processing flow.
- How are playback events stored and used by recommendations?
- What security controls are listed in the architecture?

## 7. Project structure
```text
spotify_hybrid_rag/
├── data/
│   └── spotify_web_app_architecture.pdf
├── storage/
│   └── faiss_index/
├── RAG.py
├── ingestion.py
├── backend.py
├── frontend.py
├── requirements.txt
├── .env.example
├── .gitignore
└── README.md
```

## Notes
- The graph extraction uses an LLM and only stores relationships supported by the PDF text.
- The first ingestion can take several minutes because each chunk is processed for graph extraction and embeddings.
- FAISS loading uses local pickle deserialization. Only load an index you created yourself.
- Do not commit API keys, Neo4j passwords, virtual environments, or generated indexes unless intentionally required.
