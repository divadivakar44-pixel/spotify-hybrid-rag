
import traceback

from RAG import (
    load_pdf_chunks,
    build_vector_index,
    extract_graph_data,
    save_graph,
    test_neo4j_connection
)


def main():

    print("\n======================================", flush=True)
    print("SPOTIFY HYBRID RAG INGESTION STARTED", flush=True)
    print("======================================\n", flush=True)

    # STEP 1: Load and split PDF.
    print("1/4 Loading and splitting PDF...", flush=True)

    chunks = load_pdf_chunks()

    print(
        f"Created {len(chunks)} chunks.",
        flush=True
    )

    if not chunks:
        raise ValueError("No chunks were created from the PDF.")

    # STEP 2: Build FAISS vector index.
    print("\n2/4 Building vector index...", flush=True)

    build_vector_index(chunks)

    print("Vector index saved successfully.", flush=True)

    # STEP 3: Extract graph entities and relationships.
    print(
        "\n3/4 Extracting graph entities and relationships...",
        flush=True
    )

    entities, relationships = extract_graph_data(chunks)

    print(
        f"\nExtracted {len(entities)} entities and "
        f"{len(relationships)} relationships.",
        flush=True
    )

    if not entities:
        raise ValueError(
            "No entities were extracted. Check the OpenAI API "
            "key and extraction responses."
        )

    # STEP 4: Connect to Neo4j and save graph.
    print("\n4/4 Connecting to Neo4j...", flush=True)

    test_neo4j_connection()

    print("\nSaving graph to Neo4j...", flush=True)

    save_graph(entities, relationships)

    print("\n======================================", flush=True)
    print("INGESTION COMPLETED SUCCESSFULLY!", flush=True)
    print("======================================", flush=True)


if __name__ == "__main__":

    try:
        main()

    except Exception:

        print("\nINGESTION FAILED!", flush=True)

        traceback.print_exc()
