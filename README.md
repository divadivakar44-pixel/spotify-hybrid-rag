# Spotify Hybrid RAG – AI-Powered Document Question Answering System

## 1. Project Overview

Spotify Hybrid RAG is an AI-powered document question-answering application that combines **Vector Retrieval-Augmented Generation (Vector RAG)** and **Graph Retrieval-Augmented Generation (Graph RAG)**.

The application processes Spotify Web Application Architecture documentation and answers user questions by retrieving relevant information from two knowledge sources:

- FAISS – Vector similarity search.
- Neo4j – Knowledge graph and relationship-based retrieval.

The retrieved context is combined and passed to OpenAI GPT-4o-mini to generate relevant, contextual answers.

The application is developed using Python, LangChain, FastAPI, and Streamlit, and deployed on a Hostinger Ubuntu VPS using Docker and Docker Compose.

## 2. Project Objectives

- Build an intelligent document question-answering system.
- Implement semantic search using vector embeddings.
- Extract entities and relationships from technical documentation.
- Store and retrieve connected information using a knowledge graph.
- Combine vector and graph retrieval into a hybrid RAG pipeline.
- Develop REST APIs for backend communication.
- Build an interactive web interface.
- Containerize and deploy the application on a cloud VPS.

## 3. Technology Stack

| Technology | Purpose |
|---|---|
| Python | Core application development |
| LangChain | RAG pipeline orchestration |
| PyPDFLoader | PDF text extraction |
| RecursiveCharacterTextSplitter | Document chunking |
| OpenAI GPT-4o-mini | LLM-based answer generation |
| OpenAI Embeddings | Text embedding generation |
| FAISS | Vector database and similarity search |
| Neo4j Aura | Graph database |
| FastAPI | Backend REST API |
| Streamlit | Interactive frontend |
| Requests | Frontend-backend communication |
| Docker | Application containerization |
| Docker Compose | Multi-container management |
| Ubuntu 24.04 | VPS operating system |
| Hostinger VPS | Cloud deployment platform |
| Git & GitHub | Version control and source management |

## 4. Project Architecture

The application follows a document ingestion, dual knowledge-building, hybrid retrieval, and answer-generation architecture.

### High-Level Workflow

1. Upload the Spotify architecture PDF.
2. Extract text using PyPDFLoader.
3. Split the document into chunks.
4. Generate embeddings using OpenAI.
5. Store embeddings in FAISS.
6. Extract entities and relationships using an LLM.
7. Store the knowledge graph in Neo4j.
8. Accept user questions through Streamlit.
9. Send questions to FastAPI.
10. Retrieve relevant chunks from FAISS.
11. Retrieve related entities and relationships from Neo4j.
12. Combine the retrieved context.
13. Generate answers using GPT-4o-mini.
14. Display answers in Streamlit.

## 5. End-to-End Implementation Phases

### Phase 1 – Environment Setup

- Installed and configured Python.
- Created a project directory.
- Created and activated a Python virtual environment.
- Installed required Python dependencies.
- Configured environment variables using `.env`.

### Phase 2 – PDF Document Processing

- Loaded the Spotify Web Application Architecture PDF.
- Extracted document text using PyPDFLoader.
- Implemented RecursiveCharacterTextSplitter.
- Configured chunk size to 900 characters.
- Configured chunk overlap to 150 characters.

### Phase 3 – Vector RAG Implementation

- Integrated OpenAI `text-embedding-3-small`.
- Converted document chunks into vector embeddings.
- Created a FAISS vector index.
- Implemented semantic similarity retrieval.
- Retrieved relevant document chunks based on user questions.

### Phase 4 – Graph RAG Implementation

- Integrated LLM-based entity extraction.
- Identified entities and relationships from document chunks.
- Created graph nodes and relationships.
- Connected the application to Neo4j Aura.
- Stored and retrieved graph information.
- Implemented graph-based contextual retrieval.

### Phase 5 – Hybrid Retrieval Implementation

- Integrated FAISS vector retrieval.
- Integrated Neo4j graph retrieval.
- Retrieved relevant document context and graph relationships.
- Combined retrieval results.
- Passed the combined context to the LLM.
- Generated contextual answers using GPT-4o-mini.

### Phase 6 – Backend Development

- Developed the backend using FastAPI.
- Implemented the `/ask` API endpoint.
- Accepted user questions through HTTP requests.
- Integrated hybrid retrieval with the backend.
- Returned generated answers and retrieval results to the frontend.

### Phase 7 – Frontend Development

- Developed an interactive interface using Streamlit.
- Implemented question input functionality.
- Integrated frontend with FastAPI using Requests.
- Displayed generated answers.
- Displayed vector and graph retrieval results.

### Phase 8 – Local Application Testing

- Tested PDF ingestion.
- Validated FAISS index creation.
- Tested Neo4j graph connectivity.
- Verified vector retrieval.
- Verified graph retrieval.
- Tested hybrid question answering.
- Validated frontend-backend communication.

### Phase 9 – Docker Containerization

Created the following deployment files:

- `Dockerfile`
- `docker-compose.yml`
- `.dockerignore`
- `.gitignore`

Implemented:

- Python application image creation.
- FastAPI backend container.
- Streamlit frontend container.
- Docker Compose service configuration.
- Environment variable injection.
- Persistent storage mapping for FAISS.
- Container restart policies.

### Phase 10 – GitHub Integration

- Created the GitHub repository.
- Initialized Git version control.
- Added application source code.
- Added Docker configuration files.
- Configured `.gitignore` and `.dockerignore`.
- Committed and pushed project files.
- Verified repository contents.

### Phase 11 – Hostinger VPS Deployment

- Configured Hostinger VPS.
- Used Ubuntu 24.04 as the deployment environment.
- Connected to the VPS terminal.
- Verified Docker installation.
- Cloned the GitHub repository.
- Configured production environment variables.
- Secured `.env` permissions.
- Built Docker images.
- Executed the ingestion process.
- Started frontend and backend containers.
- Verified container status and application logs.

### Phase 12 – Production Connectivity and Troubleshooting

- Connected the backend container to Neo4j Aura.
- Checked Neo4j environment configuration.
- Investigated database connection failures.
- Restarted the Neo4j database.
- Re-established database connectivity.
- Restarted Docker services.
- Verified application functionality.

## 6. Application Components

### Document Ingestion

Extracts and processes PDF content into manageable text chunks.

### Vector Retrieval – FAISS

Uses embeddings and semantic similarity search to retrieve relevant document information.

### Graph Retrieval – Neo4j

Stores entities and their relationships to provide relationship-based context.

### Hybrid RAG Engine

Combines vector and graph retrieval results to improve contextual answer generation.

### FastAPI Backend

Handles incoming questions and coordinates retrieval and LLM processing.

### Streamlit Frontend

Provides the interactive user interface for asking questions and viewing responses.

## 7. Project Folder Structure

```text
spotify-hybrid-rag/
│
├── data/
│   └── spotify_web_app_architecture.pdf
│
├── storage/
│   └── faiss_index/
│
├── backend.py
├── frontend.py
├── RAG.py
├── ingestion.py
│
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
├── .dockerignore
├── .gitignore
├── .env
└── README.md
```

Note: `.env` is a local/VPS configuration file and must not be committed to GitHub.

## 8. Deployment Architecture

**Cloud Platform:** Hostinger VPS

**Operating System:** Ubuntu 24.04

**Containerization:** Docker and Docker Compose

**Application Services:**

- FastAPI Backend – Port 8000 (internal Docker network).
- Streamlit Frontend – Port 8501.
- FAISS – Persistent vector storage.
- Neo4j Aura – External graph database.

### Application Access

Streamlit application:

`http://YOUR_VPS_PUBLIC_IP:8501`

Backend API:

`http://YOUR_VPS_PUBLIC_IP:8000`

The backend port should remain private in production unless explicitly protected.

## 9. Docker Commands Used

Build Docker images:

```bash
docker compose build
```

Start containers:

```bash
docker compose up -d
```

Check running containers:

```bash
docker compose ps
```

View backend logs:

```bash
docker compose logs --tail=100 backend
```

View frontend logs:

```bash
docker compose logs --tail=100 frontend
```

Restart containers:

```bash
docker compose restart
```

Stop containers:

```bash
docker compose down
```

Run ingestion:

```bash
docker compose run --rm backend python ingestion.py
```

## 10. Key Learning Outcomes

- Understanding Retrieval-Augmented Generation architecture.
- Implementing vector embeddings and semantic retrieval.
- Working with FAISS vector databases.
- Building knowledge graphs using Neo4j.
- Implementing hybrid retrieval pipelines.
- Integrating OpenAI LLMs with LangChain.
- Building REST APIs using FastAPI.
- Developing interactive AI applications using Streamlit.
- Containerizing Python applications using Docker.
- Managing multi-container services using Docker Compose.
- Deploying applications on Ubuntu-based VPS infrastructure.
- Troubleshooting cloud database connectivity and application deployment issues.
- Managing source code and deployment configuration using GitHub.

## 11. Future Enhancements

- Implement HTTPS using Nginx or Caddy.
- Add user authentication and access control.
- Implement graceful fallback when Neo4j is unavailable.
- Add source citations to generated answers.
- Integrate logging and application monitoring.
- Implement automated testing.
- Set up CI/CD using GitHub Actions.
- Add retrieval evaluation and performance monitoring.
- Improve scalability and production security.

## 12. Conclusion

The Spotify Hybrid RAG project demonstrates an end-to-end implementation of an AI-powered document question-answering application.

By combining FAISS vector search and Neo4j knowledge graph retrieval, the system can retrieve both semantically relevant information and connected entity context.

The project covers the complete lifecycle, from PDF ingestion and AI integration to backend/frontend development, Docker containerization, GitHub version control, and cloud VPS deployment.

**Repository:** https://github.com/divadivakar44-pixel/spotify-hybrid-rag
