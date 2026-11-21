import streamlit as st
import requests

st.set_page_config(page_title="Spotify Hybrid RAG", page_icon="🎧", layout="wide")
st.title("🎧 Spotify Architecture — Hybrid RAG + Knowledge Graph")
st.caption("Ask questions across the PDF using semantic retrieval and Neo4j relationships.")

API_URL = st.sidebar.text_input("FastAPI URL", "http://127.0.0.1:8000")
question = st.text_area("Ask a question", placeholder="How does the Media Processor interact with Object Storage?")
k = st.slider("Vector chunks to retrieve", 2, 10, 5)

if st.button("Ask", type="primary"):
    if not question.strip():
        st.warning("Enter a question first.")
    else:
        with st.spinner("Searching vector index and knowledge graph..."):
            try:
                response = requests.post(
                    f"{API_URL}/ask", json={"question": question, "k": k}, timeout=180
                )
                response.raise_for_status()
                data = response.json()
                st.subheader("Answer")
                st.write(data["answer"])
                with st.expander("PDF source chunks"):
                    for source in data.get("sources", []):
                        st.markdown(f"**Page {source['page']} · Chunk {source['chunk_id']}**")
                        st.write(source["preview"])
                with st.expander("Knowledge graph retrieval context"):
                    st.text(data.get("graph_context", ""))
            except requests.RequestException as exc:
                st.error(f"API request failed: {exc}")
