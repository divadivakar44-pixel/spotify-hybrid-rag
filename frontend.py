
import os
import requests
import streamlit as st

# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")

st.set_page_config(
    page_title="Spotify Hybrid RAG Assistant",
    page_icon="🎵",
    layout="wide"
)

# --------------------------------------------------
# HEADER
# --------------------------------------------------

st.title("🎵 Spotify Architecture Assistant")

st.markdown(
    """
    Ask questions about the Spotify Web App Architecture.

    This application uses **Hybrid RAG**:
    - Vector Search using FAISS
    - Knowledge Graph Search using Neo4j
    - LLM-powered answers using OpenAI
    """
)

st.divider()

# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------

with st.sidebar:
    st.header("⚙️ Retrieval Settings")

    k = st.slider(
        "Number of Vector Results",
        min_value=1,
        max_value=10,
        value=5
    )

    st.markdown("---")

    st.subheader("Architecture")

    st.write("📄 PDF Document")
    st.write("🔎 FAISS Vector Search")
    st.write("🕸️ Neo4j Knowledge Graph")
    st.write("🤖 OpenAI LLM")

    st.markdown("---")

    st.caption("Spotify Hybrid RAG Application")

# --------------------------------------------------
# QUESTION INPUT
# --------------------------------------------------

st.subheader("💬 Ask Your Question")

question = st.text_area(
    "Enter your question:",
    placeholder="Example: Explain the Spotify application architecture.",
    height=100
)

ask_button = st.button(
    "🚀 Get Answer",
    type="primary",
    use_container_width=True
)

# --------------------------------------------------
# HELPER FUNCTIONS
# --------------------------------------------------

def display_vector_results(vector_results):

    st.subheader("📄 Retrieved PDF Source Chunks")

    if not vector_results:
        st.info("No vector search results found.")
        return

    for index, item in enumerate(vector_results, start=1):

        if isinstance(item, dict):
            content = item.get("content", str(item))
            page = item.get("page", "Not available")
            score = item.get("score", "Not available")
        else:
            content = str(item)
            page = "Not available"
            score = "Not available"

        with st.expander(f"Source {index} | Page: {page}"):

            st.write(content)

            st.caption(f"Similarity Score: {score}")


def display_graph_results(graph_results):

    st.subheader("🕸️ Knowledge Graph Retrieval")

    if not graph_results:
        st.info("No knowledge graph results found.")
        return

    for index, entity in enumerate(graph_results, start=1):

        if isinstance(entity, dict):

            entity_name = entity.get("entity", "Unknown Entity")
            entity_type = entity.get("entity_type", "Unknown")
            description = entity.get("description", "")

            with st.expander(f"Entity {index}: {entity_name}"):

                st.write(f"**Entity Type:** {entity_type}")

                if description:
                    st.write(f"**Description:** {description}")

                connections = entity.get("connections", [])

                if connections:
                    st.markdown("**Connected Entities:**")

                    for connection in connections:

                        if isinstance(connection, dict):
                            relationship = connection.get(
                                "relationship", "RELATED_TO"
                            )
                            connected_entity = connection.get(
                                "connected_entity", "Unknown"
                            )
                            connected_type = connection.get(
                                "connected_type", "Unknown"
                            )

                            st.write(
                                f"- {entity_name} "
                                f"--[{relationship}]--> "
                                f"{connected_entity} "
                                f"({connected_type})"
                            )
                        else:
                            st.write(f"- {connection}")

        else:
            st.write(entity)


# --------------------------------------------------
# API REQUEST AND RESPONSE
# --------------------------------------------------

if ask_button:

    if not question.strip():

        st.warning("Please enter a question before submitting.")

    else:

        try:

            with st.spinner(
                "Searching FAISS and Neo4j and generating your answer..."
            ):

                response = requests.post(
                    f"{API_URL.rstrip('/')}/ask",
                    json={
                        "question": question,
                        "k": k
                    },
                    timeout=120
                )

            # Check API response status
            response.raise_for_status()

            result = response.json()

            # ------------------------------------------
            # DISPLAY LLM ANSWER
            # ------------------------------------------

            st.success("Answer generated successfully!")

            st.subheader("🤖 AI Generated Answer")

            answer = result.get(
                "answer",
                "No answer returned by the backend."
            )

            st.markdown(answer)

            st.divider()

            # ------------------------------------------
            # DISPLAY VECTOR RESULTS
            # ------------------------------------------

            vector_results = result.get("vector_results", [])

            display_vector_results(vector_results)

            st.divider()

            # ------------------------------------------
            # DISPLAY GRAPH RESULTS
            # ------------------------------------------

            graph_results = result.get("graph_results", [])

            display_graph_results(graph_results)

            st.divider()

            # ------------------------------------------
            # RETRIEVAL SUMMARY
            # ------------------------------------------

            st.subheader("📊 Retrieval Summary")

            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric(
                    "Vector Results",
                    len(vector_results)
                    if isinstance(vector_results, list)
                    else 0
                )

            with col2:
                st.metric(
                    "Graph Results",
                    len(graph_results)
                    if isinstance(graph_results, list)
                    else 0
                )

            with col3:
                st.metric("Top K", k)

        except requests.exceptions.ConnectionError:

            st.error(
                "Cannot connect to the FastAPI backend. "
                "Please check whether the backend container is running."
            )

        except requests.exceptions.Timeout:

            st.error(
                "The request timed out. Please try again."
            )

        except requests.exceptions.HTTPError as error:

            st.error(f"Backend HTTP error: {error}")

            st.code(
                response.text if "response" in locals()
                else "No response details available."
            )

        except Exception as error:

            st.error(f"Unexpected error: {error}")

# --------------------------------------------------
# FOOTER
# --------------------------------------------------

st.divider()

st.caption(
    "Built with Streamlit | FastAPI | FAISS | Neo4j | OpenAI"
)
