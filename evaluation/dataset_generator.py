import os
import argparse
import logging
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

from docling_core.types.doc import DoclingDocument

from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings

from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper

from ragas.testset.graph import KnowledgeGraph, Node, NodeType
from ragas.testset.transforms import default_transforms, apply_transforms
from ragas.testset import TestsetGenerator
from ragas.testset.synthesizers import default_query_distribution


# --------------------------------------------------
# 1. CONFIGURATION
# --------------------------------------------------

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

LLM_MODEL = "openai/gpt-oss-120b"

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


# --------------------------------------------------
# 2. LOGGING
# --------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

logger = logging.getLogger(__name__)


# --------------------------------------------------
# 3. LOAD PARSED DOCLING DOCUMENTS
# --------------------------------------------------

def load_parsed_documents(input_dir: Path):

    documents = []

    json_files = list(input_dir.rglob("*.json"))

    if not json_files:
        raise FileNotFoundError(
            f"No JSON files found inside: {input_dir}"
        )

    logger.info("Found %d JSON files", len(json_files))

    for file_path in json_files:

        try:
            # Ignore JSON files that are not Docling documents.
            import json

            with open(file_path, "r", encoding="utf-8") as f:
                raw_data = json.load(f)

            if raw_data.get("schema_name") != "DoclingDocument":
                logger.warning(
                    "Skipping non-Docling JSON: %s",
                    file_path.name
                )
                continue

            # Reconstruct the DoclingDocument object.
            docling_doc = DoclingDocument.load_from_json(file_path)

            # Export structured document content as Markdown.
            text_content = docling_doc.export_to_markdown()

            if not text_content or not text_content.strip():
                logger.warning(
                    "No textual content found: %s",
                    file_path.name
                )
                continue

            # Extract original filename if available.
            original_filename = (
                raw_data.get("origin", {}).get("filename")
                or file_path.name
            )

            # Create a LangChain Document.
            from langchain_core.documents import Document

            document = Document(
                page_content=text_content,
                metadata={
                    "source": str(file_path.resolve()),
                    "file_name": original_filename,
                    "docling_name": raw_data.get("name", file_path.stem)
                }
            )

            documents.append(document)

            logger.info(
                "Loaded: %s | Characters: %d",
                original_filename,
                len(text_content)
            )

        except Exception as e:
            logger.exception(
                "Failed to load %s: %s",
                file_path.name,
                e
            )

    if not documents:
        raise ValueError(
            "No valid Docling documents could be loaded."
        )

    logger.info(
        "Successfully loaded %d documents",
        len(documents)
    )

    return documents


# --------------------------------------------------
# 4. INITIALIZE LLM AND EMBEDDING MODELS
# --------------------------------------------------

def initialize_models():

    if not GROQ_API_KEY:
        raise EnvironmentError(
            "GROQ_API_KEY is missing. Add it to your .env file."
        )

    logger.info("Initializing Groq LLM...")

    langchain_llm = ChatGroq(
        model=LLM_MODEL,
        api_key=GROQ_API_KEY,
        temperature=0
    )

    logger.info("Initializing embedding model...")

    langchain_embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )

    # Convert LangChain models into RAGAS-compatible wrappers.
    ragas_llm = LangchainLLMWrapper(langchain_llm)

    ragas_embeddings = LangchainEmbeddingsWrapper(
        langchain_embeddings
    )

    logger.info("Models initialized successfully.")

    return ragas_llm, ragas_embeddings


# --------------------------------------------------
# 5. CREATE KNOWLEDGE GRAPH
# --------------------------------------------------

def create_knowledge_graph(documents, ragas_llm, ragas_embeddings):

    logger.info("Creating Knowledge Graph...")

    kg = KnowledgeGraph()

    # Add each parsed document as a DOCUMENT node.
    for doc in documents:

        node = Node(
            type=NodeType.DOCUMENT,
            properties={
                "page_content": doc.page_content,
                "document_metadata": doc.metadata
            }
        )

        kg.nodes.append(node)

    logger.info(
        "Added %d document nodes to Knowledge Graph",
        len(kg.nodes)
    )

    # Define default transformations.
    transformations = default_transforms(
        documents=documents,
        llm=ragas_llm,
        embedding_model=ragas_embeddings
    )

    logger.info("Applying default transformations...")

    # Enrich the Knowledge Graph.
    apply_transforms(
        kg,
        transformations
    )

    logger.info(
        "Knowledge Graph created: %d nodes, %d relationships",
        len(kg.nodes),
        len(kg.relationships)
    )

    return kg


# --------------------------------------------------
# 6. GENERATE TESTSET
# --------------------------------------------------

def generate_testset(
    kg,
    ragas_llm,
    ragas_embeddings,
    testset_size: int
):

    logger.info("Initializing TestsetGenerator...")

    generator = TestsetGenerator(
        llm=ragas_llm,
        embedding_model=ragas_embeddings,
        knowledge_graph=kg
    )

    # Default distribution:
    # 50% Single-hop specific
    # 25% Multi-hop abstract
    # 25% Multi-hop specific

    query_distribution = default_query_distribution(
        ragas_llm
    )

    logger.info(
        "Generating %d synthetic questions...",
        testset_size
    )

    testset = generator.generate(
        testset_size=testset_size,
        query_distribution=query_distribution
    )

    return testset


# --------------------------------------------------
# 7. SAVE KNOWLEDGE GRAPH AND CSV
# --------------------------------------------------

def save_outputs(kg, testset, output_dir: Path):

    kg_dir = output_dir / "knowledge_graphs"
    testset_dir = output_dir / "testsets"

    kg_dir.mkdir(parents=True, exist_ok=True)
    testset_dir.mkdir(parents=True, exist_ok=True)

    # Save Knowledge Graph.
    kg_path = kg_dir / "electronics_knowledge_graph.json"

    kg.save(str(kg_path))

    logger.info("Knowledge Graph saved: %s", kg_path)

    # Convert generated testset to Pandas DataFrame.
    df = testset.to_pandas()

    # Save CSV.
    csv_path = testset_dir / "electronics_testset.csv"

    df.to_csv(
        csv_path,
        index=False,
        encoding="utf-8-sig"
    )

    logger.info("Testset CSV saved: %s", csv_path)

    logger.info("Generated dataset shape: %s", df.shape)

    print("\nGenerated Testset Preview:")
    print(df.head().to_string(index=False))

    print("\nCSV columns:")
    print(df.columns.tolist())


# --------------------------------------------------
# 8. MAIN PIPELINE
# --------------------------------------------------

def main():

    parser = argparse.ArgumentParser(
        description="Generate RAGAS testset from Docling JSON files."
    )

    parser.add_argument(
        "--input-dir",
        type=str,
        required=True,
        help="Folder containing parsed Docling JSON documents."
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="evaluation",
        help="Folder where KG and testset CSV will be saved."
    )

    parser.add_argument(
        "--testset-size",
        type=int,
        default=50,
        help="Number of synthetic questions to generate."
    )

    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    if not input_dir.exists() or not input_dir.is_dir():
        raise NotADirectoryError(
            f"Invalid input directory: {input_dir}"
        )

    if args.testset_size <= 0:
        raise ValueError("Testset size must be greater than zero.")

    logger.info("Starting RAGAS testset generation pipeline.")

    # Step 1: Load parsed documents.
    documents = load_parsed_documents(input_dir)

    # Step 2: Initialize models.
    ragas_llm, ragas_embeddings = initialize_models()

    # Step 3: Build Knowledge Graph.
    kg = create_knowledge_graph(
        documents,
        ragas_llm,
        ragas_embeddings
    )

    # Step 4: Generate synthetic testset.
    testset = generate_testset(
        kg,
        ragas_llm,
        ragas_embeddings,
        args.testset_size
    )

    # Step 5: Save KG and CSV.
    save_outputs(
        kg,
        testset,
        output_dir
    )

    logger.info("Pipeline completed successfully.")


if __name__ == "__main__":
    main()



#Need to change the llm to accomodate multiple requests to generate testset data.