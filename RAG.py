
import os
import re
import json
from pathlib import Path

from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from langchain_community.vectorstores import FAISS

from langchain_core.prompts import ChatPromptTemplate

from neo4j import GraphDatabase


# ============================================================
# 1. LOAD ENVIRONMENT VARIABLES
# ============================================================

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")
NEO4J_DATABASE = os.getenv("NEO4J_DATABASE", "neo4j")

PDF_PATH = os.getenv(
    "PDF_PATH",
    "data/spotify_web_app_architecture.pdf"
)

VECTOR_DIR = os.getenv(
    "VECTOR_DIR",
    "storage/faiss_index"
)

if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY is missing in .env")

if not NEO4J_URI or not NEO4J_PASSWORD:
    raise ValueError("Neo4j credentials are missing in .env")


# ============================================================
# 2. INITIALIZE OPENAI MODELS
# ============================================================

llm = ChatOpenAI(
    model=OPENAI_MODEL,
    temperature=0
)

embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small"
)


# ============================================================
# 3. LOAD AND SPLIT PDF
# ============================================================

def load_pdf_chunks():

    pdf_file = Path(PDF_PATH)

    if not pdf_file.exists():
        raise FileNotFoundError(
            f"PDF not found: {pdf_file.resolve()}"
        )

    print("Loading Spotify architecture PDF...", flush=True)

    loader = PyPDFLoader(str(pdf_file))

    documents = loader.load()

    print(
        f"Total pages loaded: {len(documents)}",
        flush=True
    )

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=900,
        chunk_overlap=150
    )

    chunks = splitter.split_documents(documents)

    print(
        f"Total chunks created: {len(chunks)}",
        flush=True
    )

    return chunks


# ============================================================
# 4. VECTOR RAG - FAISS
# ============================================================

def build_vector_index(chunks):

    if not chunks:
        raise ValueError("No chunks available for indexing.")

    print("Generating embeddings...", flush=True)

    vector_db = FAISS.from_documents(
        documents=chunks,
        embedding=embeddings
    )

    os.makedirs(VECTOR_DIR, exist_ok=True)

    vector_db.save_local(VECTOR_DIR)

    print(
        f"FAISS index saved successfully: {VECTOR_DIR}",
        flush=True
    )

    return vector_db


def load_vector_index():

    if not os.path.exists(VECTOR_DIR):
        raise FileNotFoundError(
            "FAISS index not found. Run ingestion.py first."
        )

    vector_db = FAISS.load_local(
        VECTOR_DIR,
        embeddings,
        allow_dangerous_deserialization=True
    )

    return vector_db


# ============================================================
# 5. KNOWLEDGE GRAPH EXTRACTION PROMPT
# ============================================================

# IMPORTANT:
# JSON braces are doubled as {{ and }}.
# This prevents LangChain template formatting errors.

EXTRACTION_PROMPT = ChatPromptTemplate.from_messages([

    (
        "system",
        """
You are an information extraction system.

Extract entities and relationships from the provided
Spotify web application architecture document.

Return ONLY valid JSON.

Use this exact structure:

{{
    "entities": [
        {{
            "name": "Spotify Web Application",
            "type": "APPLICATION",
            "description": "Main Spotify web application"
        }},
        {{
            "name": "Frontend",
            "type": "COMPONENT",
            "description": "User interface component"
        }}
    ],

    "relationships": [
        {{
            "source": "Spotify Web Application",
            "target": "Frontend",
            "type": "HAS_COMPONENT",
            "evidence": "The application contains a frontend"
        }}
    ]
}}

Extraction rules:

1. Extract important technologies.
2. Extract microservices.
3. Extract databases and storage systems.
4. Extract frontend and backend components.
5. Extract APIs and integrations.
6. Extract security components.
7. Extract application features.
8. Extract meaningful relationships.
9. Use consistent entity names.
10. Relationship types must contain only uppercase
    letters, numbers and underscores.
11. Do not invent information.
12. Return empty arrays if nothing relevant is found.
13. Return JSON only without Markdown code fences.
        """
    ),

    (
        "human",
        """
Extract entities and relationships from this document text:

{text}
        """
    )

])


# ============================================================
# 6. PARSE LLM JSON RESPONSE
# ============================================================

def parse_json_response(response_text):

    cleaned = response_text.strip()

    # Remove Markdown code fences if present.
    cleaned = re.sub(
        r"^```(?:json)?\s*",
        "",
        cleaned
    )

    cleaned = re.sub(
        r"\s*```$",
        "",
        cleaned
    )

    try:
        return json.loads(cleaned)

    except json.JSONDecodeError:
        pass

    # Try to extract a JSON object from the response.
    match = re.search(
        r"\{.*\}",
        cleaned,
        re.DOTALL
    )

    if match:

        return json.loads(match.group(0))

    raise ValueError(
        "Unable to parse LLM JSON response: " + response_text
    )


# ============================================================
# 7. EXTRACT ENTITIES AND RELATIONSHIPS
# ============================================================

def extract_graph_data(chunks):

    all_entities = {}

    all_relationships = {}

    total_chunks = len(chunks)

    print(
        f"Starting graph extraction for {total_chunks} chunks...",
        flush=True
    )

    for index, chunk in enumerate(chunks, start=1):

        print(
            f"\nProcessing chunk {index}/{total_chunks}...",
            flush=True
        )

        try:

            response = llm.invoke(
                EXTRACTION_PROMPT.format_messages(
                    text=chunk.page_content
                )
            )

            response_text = response.content

            # Handle text blocks returned as a list.
            if isinstance(response_text, list):

                text_parts = []

                for item in response_text:

                    if isinstance(item, dict):
                        text_parts.append(
                            str(item.get("text", ""))
                        )
                    else:
                        text_parts.append(str(item))

                response_text = "\n".join(text_parts)

            extracted = parse_json_response(response_text)

            entities = extracted.get("entities", [])

            relationships = extracted.get(
                "relationships",
                []
            )

            # --------------------------------------------
            # STORE UNIQUE ENTITIES
            # --------------------------------------------

            for entity in entities:

                name = str(
                    entity.get("name", "")
                ).strip()

                if not name:
                    continue

                entity_type = str(
                    entity.get("type", "UNKNOWN")
                ).strip().upper()

                description = str(
                    entity.get("description", "")
                ).strip()

                key = name.lower()

                all_entities[key] = {
                    "name": name,
                    "type": entity_type,
                    "description": description
                }

            # --------------------------------------------
            # STORE UNIQUE RELATIONSHIPS
            # --------------------------------------------

            for relationship in relationships:

                source = str(
                    relationship.get("source", "")
                ).strip()

                target = str(
                    relationship.get("target", "")
                ).strip()

                rel_type = str(
                    relationship.get("type", "RELATED_TO")
                ).strip().upper()

                evidence = str(
                    relationship.get("evidence", "")
                ).strip()

                rel_type = re.sub(
                    r"[^A-Z0-9_]",
                    "_",
                    rel_type
                )

                if not source or not target:
                    continue

                if not rel_type:
                    rel_type = "RELATED_TO"

                relationship_key = (
                    source.lower(),
                    target.lower(),
                    rel_type
                )

                all_relationships[relationship_key] = {
                    "source": source,
                    "target": target,
                    "type": rel_type,
                    "evidence": evidence
                }

            print(
                f"Chunk {index} completed | "
                f"Entities: {len(all_entities)} | "
                f"Relationships: {len(all_relationships)}",
                flush=True
            )

        except Exception as error:

            print(
                f"Error processing chunk {index}: {error}",
                flush=True
            )

            continue

    entities_list = list(all_entities.values())

    relationships_list = list(
        all_relationships.values()
    )

    print("\nGraph extraction completed.", flush=True)

    print(
        f"Total unique entities: {len(entities_list)}",
        flush=True
    )

    print(
        f"Total unique relationships: {len(relationships_list)}",
        flush=True
    )

    return entities_list, relationships_list


# ============================================================
# 8. NEO4J DATABASE CONNECTION
# ============================================================

def get_neo4j_driver():

    driver = GraphDatabase.driver(
        NEO4J_URI,
        auth=(
            NEO4J_USERNAME,
            NEO4J_PASSWORD
        )
    )

    driver.verify_connectivity()

    return driver


def test_neo4j_connection():

    driver = get_neo4j_driver()

    try:

        with driver.session(
            database=NEO4J_DATABASE
        ) as session:

            result = session.run(
                "RETURN 'Neo4j Connected Successfully' AS message"
            )

            record = result.single()

            print(
                record["message"],
                flush=True
            )

    finally:

        driver.close()


# ============================================================
# 9. SAVE KNOWLEDGE GRAPH TO NEO4J
# ============================================================

def save_graph(entities, relationships):

    print("\nConnecting to Neo4j...", flush=True)

    driver = get_neo4j_driver()

    try:

        with driver.session(
            database=NEO4J_DATABASE
        ) as session:

            # --------------------------------------------
            # CREATE UNIQUE ENTITY CONSTRAINT
            # --------------------------------------------

            print(
                "Creating entity constraint...",
                flush=True
            )

            session.run("""
                CREATE CONSTRAINT kg_entity_name_unique IF NOT EXISTS
                FOR (e:KGEntity)
                REQUIRE e.name IS UNIQUE
            """).consume()

            # --------------------------------------------
            # CLEAR PREVIOUS PROJECT GRAPH
            # --------------------------------------------

            print(
                "Clearing existing KGEntity graph...",
                flush=True
            )

            session.run("""
                MATCH (n:KGEntity)
                DETACH DELETE n
            """).consume()

            # --------------------------------------------
            # INSERT ENTITIES
            # --------------------------------------------

            print(
                f"Saving {len(entities)} entities...",
                flush=True
            )

            if entities:

                session.run("""
                    UNWIND $entities AS item

                    MERGE (e:KGEntity {
                        name: item.name
                    })

                    SET
                        e.entity_type = item.type,
                        e.description = item.description

                """, entities=entities).consume()

            print(
                f"Successfully saved {len(entities)} entities.",
                flush=True
            )

            # --------------------------------------------
            # INSERT RELATIONSHIPS
            # --------------------------------------------

            print(
                f"Saving {len(relationships)} relationships...",
                flush=True
            )

            for index, relationship in enumerate(
                relationships,
                start=1
            ):

                source = relationship["source"]

                target = relationship["target"]

                rel_type = re.sub(
                    r"[^A-Z0-9_]",
                    "_",
                    relationship["type"].upper()
                )

                if not rel_type:
                    rel_type = "RELATED_TO"

                query = f"""
                    MATCH (a:KGEntity {{name: $source}})
                    MATCH (b:KGEntity {{name: $target}})

                    MERGE (a)-[r:{rel_type}]->(b)

                    SET r.evidence = $evidence
                """

                session.run(
                    query,
                    source=source,
                    target=target,
                    evidence=relationship.get(
                        "evidence",
                        ""
                    )
                ).consume()

                print(
                    f"Relationship {index}/{len(relationships)} saved",
                    flush=True
                )

            # --------------------------------------------
            # VERIFY GRAPH COUNTS
            # --------------------------------------------

            entity_result = session.run("""
                MATCH (n:KGEntity)
                RETURN count(n) AS total
            """)

            total_entities = entity_result.single()["total"]

            relationship_result = session.run("""
                MATCH (:KGEntity)-[r]->(:KGEntity)
                RETURN count(r) AS total
            """)

            total_relationships = (
                relationship_result.single()["total"]
            )

            print("\nNeo4j graph insertion completed!", flush=True)

            print(
                f"Verified entities: {total_entities}",
                flush=True
            )

            print(
                f"Verified relationships: {total_relationships}",
                flush=True
            )

    finally:

        driver.close()

        print(
            "Neo4j connection closed.",
            flush=True
        )


# ============================================================
# 10. KNOWLEDGE GRAPH RETRIEVAL
# ============================================================

def graph_retrieve(question):

    driver = get_neo4j_driver()

    try:

        words = [
            word.lower()
            for word in re.findall(
                r"[a-zA-Z0-9_]+",
                question
            )
            if len(word) > 2
        ]

        if not words:
            return []

        with driver.session(
            database=NEO4J_DATABASE
        ) as session:

            query = """
                MATCH (e:KGEntity)

                WHERE any(
                    word IN $words
                    WHERE
                        toLower(e.name) CONTAINS word
                        OR toLower(
                            coalesce(e.description, '')
                        ) CONTAINS word
                )

                OPTIONAL MATCH (e)-[r]-(connected:KGEntity)

                RETURN
                    e.name AS entity,
                    e.entity_type AS entity_type,
                    e.description AS description,

                    collect(
                        CASE
                            WHEN connected IS NOT NULL
                            THEN {
                                relationship: type(r),
                                connected_entity: connected.name,
                                connected_type: connected.entity_type
                            }
                            ELSE null
                        END
                    ) AS connections

                LIMIT 15
            """

            result = session.run(
                query,
                words=words
            )

            graph_results = []

            for record in result:

                graph_results.append({

                    "entity": record["entity"],

                    "entity_type": record["entity_type"],

                    "description": record["description"],

                    "connections": [
                        connection
                        for connection in record["connections"]
                        if connection is not None
                    ]

                })

            return graph_results

    finally:

        driver.close()


# ============================================================
# 11. HYBRID RAG - VECTOR + KNOWLEDGE GRAPH
# ============================================================

def hybrid_answer(question, k=5):

    # --------------------------------------------
    # VECTOR SEARCH
    # --------------------------------------------

    print(
        "\nSearching FAISS Vector Database...",
        flush=True
    )

    vector_db = load_vector_index()

    vector_results = vector_db.similarity_search_with_score(
        question,
        k=k
    )

    vector_context = []

    for document, score in vector_results:

        vector_context.append({

            "content": document.page_content,

            "page": document.metadata.get(
                "page",
                "Unknown"
            ),

            "score": float(score)

        })

    print(
        f"Vector results found: {len(vector_context)}",
        flush=True
    )

    # --------------------------------------------
    # KNOWLEDGE GRAPH SEARCH
    # --------------------------------------------

    print(
        "Searching Neo4j Knowledge Graph...",
        flush=True
    )

    graph_results = graph_retrieve(question)

    print(
        f"Graph results found: {len(graph_results)}",
        flush=True
    )

    # --------------------------------------------
    # PREPARE VECTOR CONTEXT
    # --------------------------------------------

    vector_text = "\n\n".join(

        [
            (
                f"Page: {item['page']}\n"
                f"Content: {item['content']}"
            )

            for item in vector_context
        ]

    )

    # --------------------------------------------
    # PREPARE GRAPH CONTEXT
    # --------------------------------------------

    graph_text = json.dumps(
        graph_results,
        indent=2,
        ensure_ascii=False
    )

    # --------------------------------------------
    # HYBRID PROMPT
    # --------------------------------------------

    hybrid_prompt = ChatPromptTemplate.from_messages([

        (
            "system",
            """
You are a technical assistant for a Spotify web
application architecture project.

Answer questions using the supplied Vector RAG
and Knowledge Graph RAG context.

Instructions:

1. Combine information from both retrieval methods.
2. Use relevant technical details.
3. Explain components and their relationships.
4. Do not invent information.
5. If information is missing, clearly say so.
6. Give structured and easy-to-understand answers.
            """
        ),

        (
            "human",
            """
QUESTION:

{question}

VECTOR RAG CONTEXT:

{vector_context}

KNOWLEDGE GRAPH CONTEXT:

{graph_context}

Generate an accurate answer using both contexts.
            """
        )

    ])

    # --------------------------------------------
    # GENERATE FINAL ANSWER
    # --------------------------------------------

    print(
        "Generating Hybrid RAG answer...",
        flush=True
    )

    response = llm.invoke(

        hybrid_prompt.format_messages(

            question=question,

            vector_context=vector_text,

            graph_context=graph_text

        )

    )

    return {

        "answer": response.content,

        "vector_results": vector_context,

        "graph_results": graph_results

    }


# ============================================================
# 12. END OF RAG MODULE
# ============================================================

if __name__ == "__main__":

    print("Spotify Hybrid RAG module loaded successfully.")

