
# """
# RAG Evaluator using LLM-as-a-Judge (Qwen via Groq)

# Evaluates:
#   1. Context Accuracy: Measures how accurately and relevantly the retrieved context captures the
#      information required by the query and aligns with the ground truth reference context.
#   2. Answer Accuracy: Measures how accurately the LLM generated response matches the ground truth reference.
#   3. Faithfulness: Measures whether the LLM generated response is strictly grounded in the retrieved context
#      without hallucination or unsupported claims.

# Workflow:
#   - Loads queries, ground truth references, and reference contexts from ragas_evaluation_dataset.json.
#   - Queries Qdrant vector database using context_retriever to obtain top-k retrieved chunks.
#   - Generates responses with Mistral via Ollama using the retrieved context.
#   - Prompts Llama 3.3 70B on Groq to evaluate Context Accuracy, Answer Accuracy, and Faithfulness.
#   - Outputs formatted score tables, dataset averages, and saves results to CSV and JSON.
# """

# import argparse
# import csv
# import json
# import os
# import re
# import sys
# import time
# from pathlib import Path
# from typing import Any, Dict, List, Optional

# # Ensure project root is in sys.path
# PROJECT_ROOT = Path(__file__).resolve().parent.parent
# if str(PROJECT_ROOT) not in sys.path:
#     sys.path.insert(0, str(PROJECT_ROOT))


# def load_env():
#     """Load environment variables from .env file if present."""
#     env_paths = [
#         PROJECT_ROOT / ".env",
#         Path.cwd() / ".env",
#         Path(__file__).parent / ".env",
#     ]
#     for env_path in env_paths:
#         if env_path.is_file():
#             with open(env_path, "r", encoding="utf-8") as f:
#                 for line in f:
#                     line = line.strip()
#                     if line and not line.startswith("#") and "=" in line:
#                         key, val = line.split("=", 1)
#                         key = key.strip()
#                         val = val.strip().strip("'\"")
#                         if key and key not in os.environ:
#                             os.environ[key] = val
#             break


# load_env()

# DEFAULT_DATASET = Path(__file__).parent / "ragas_evaluation_dataset.json"
# DEFAULT_JUDGE_MODEL = "openai/gpt-oss-20b"
# DEFAULT_QWEN_MODEL = "qwen/qwen3.8-27b"


# def generate_qwen_response(
#     groq_client: Any,
#     query: str,
#     retrieved_context: str,
#     model: str = DEFAULT_QWEN_MODEL,
# ) -> str:
#     """Generate an answer from Qwen via Groq using retrieved context."""
#     prompt = f"""You are an assistant for Standard Operating Procedures (SOP).
# Use ONLY the following context to answer the question. If the answer cannot be found in the context, say "I don't know based on the provided documents." Do not invent information.

# Context:
# {retrieved_context}

# Question: {query}
# Answer:"""

#     response = groq_client.chat.completions.create(
#         messages=[
#             {"role": "user", "content": prompt},
#         ],
#         model=model,
#     )
#     return response.choices[0].message.content.strip()


# def build_judge_prompt(
#     query: str,
#     reference_answer: str,
#     reference_context: str,
#     retrieved_context: str,
#     generated_answer: str,
# ) -> str:
#     """Construct a rigorous prompt for Llama 3.3 70B to evaluate the 3 metrics."""
#     return f"""You are an expert AI evaluator judging the quality of a Retrieval-Augmented Generation (RAG) system.
# Carefully evaluate the system based on the provided inputs and the following three metrics:

# 1. Context Accuracy (Score 0.0 to 1.0):
#    - Definition: How accurately and adequately does the retrieved context capture the key information needed to answer the question, in comparison with the reference context and user question?
#    - 1.0: The retrieved context contains all necessary, relevant facts needed to answer the question accurately (matches reference context well).
#    - 0.5: The retrieved context contains partial information; some critical details are missing or there is significant irrelevant noise.
#    - 0.0: The retrieved context is completely irrelevant, missing all required information, or misleading.

# 2. Answer Accuracy (Score 0.0 to 1.0):
#    - Definition: How accurately does the generated response answer the user question compared to the reference (ground truth) answer?
#    - 1.0: Completely accurate and factually aligned with the reference answer.
#    - 0.5: Partially accurate; covers some correct points but is incomplete or has slight discrepancies.
#    - 0.0: Factually incorrect, contradicts the reference answer, or fails to answer the question.

# 3. Faithfulness (Score 0.0 to 1.0):
#    - Definition: Is the generated answer strictly grounded in the retrieved context? Does it avoid hallucinating facts not present in the retrieved context?
#    - 1.0: Every factual claim made in the generated answer can be directly verified from the retrieved context.
#    - 0.5: The answer is mostly grounded, but makes minor unverified assumptions or ungrounded claims.
#    - 0.0: The answer contains significant hallucinations or relies heavily on external knowledge not present in the retrieved context.

# EVALUATION INPUTS:
# User Question:
# {query}

# Reference (Ground Truth) Context:
# {reference_context}

# Reference (Ground Truth) Answer:
# {reference_answer}

# Retrieved Context from System:
# {retrieved_context}

# Generated Answer from System:
# {generated_answer}

# INSTRUCTIONS:
# You must respond with a valid JSON object ONLY. Do not include markdown code block backticks, explanations outside JSON, or conversational filler.
# Format your output exactly as follows:
# {{
#   "context_accuracy": {{
#     "score": <float between 0.0 and 1.0>,
#     "reasoning": "<brief explanation>"
#   }},
#   "answer_accuracy": {{
#     "score": <float between 0.0 and 1.0>,
#     "reasoning": "<brief explanation>"
#   }},
#   "faithfulness": {{
#     "score": <float between 0.0 and 1.0>,
#     "reasoning": "<brief explanation>"
#   }}
# }}"""


# def evaluate_with_llama_judge(
#     groq_client: Any,
#     query: str,
#     reference_answer: str,
#     reference_context: str,
#     retrieved_context: str,
#     generated_answer: str,
#     model: str = DEFAULT_JUDGE_MODEL,
#     max_retries: int = 3,
# ) -> Dict[str, Any]:
#     """Call Groq API with Llama 3.3 70B to evaluate metrics."""
#     prompt = build_judge_prompt(
#         query=query,
#         reference_answer=reference_answer,
#         reference_context=reference_context,
#         retrieved_context=retrieved_context,
#         generated_answer=generated_answer,
#     )

#     for attempt in range(1, max_retries + 1):
#         try:
#             chat_completion = groq_client.chat.completions.create(
#                 messages=[
#                     {
#                         "role": "system",
#                         "content": "You are an objective, precise LLM evaluation judge. Return only JSON.",
#                     },
#                     {"role": "user", "content": prompt},
#                 ],
#                 model=model,
#                 temperature=0.0,
#                 response_format={"type": "json_object"},
#             )
#             raw_text = chat_completion.choices[0].message.content.strip()

#             # Parse JSON
#             parsed = json.loads(raw_text)

#             # Validate expected fields
#             ctx_score = float(parsed["context_accuracy"]["score"])
#             ans_score = float(parsed["answer_accuracy"]["score"])
#             faith_score = float(parsed["faithfulness"]["score"])

#             return {
#                 "context_accuracy": {
#                     "score": round(max(0.0, min(1.0, ctx_score)), 3),
#                     "reasoning": parsed["context_accuracy"].get("reasoning", ""),
#                 },
#                 "answer_accuracy": {
#                     "score": round(max(0.0, min(1.0, ans_score)), 3),
#                     "reasoning": parsed["answer_accuracy"].get("reasoning", ""),
#                 },
#                 "faithfulness": {
#                     "score": round(max(0.0, min(1.0, faith_score)), 3),
#                     "reasoning": parsed["faithfulness"].get("reasoning", ""),
#                 },
#             }

#         except Exception as e:
#             if attempt == max_retries:
#                 print(f"  [Warning] Judge evaluation failed after {max_retries} attempts: {e}")
#                 return {
#                     "context_accuracy": {"score": 0.0, "reasoning": f"Error: {e}"},
#                     "answer_accuracy": {"score": 0.0, "reasoning": f"Error: {e}"},
#                     "faithfulness": {"score": 0.0, "reasoning": f"Error: {e}"},
#                 }
#             time.sleep(2 * attempt)


# def display_results_table(results: List[Dict[str, Any]]):
#     """Print a clean formatted table summarizing scores."""
#     print("\n" + "=" * 90)
#     print(" " * 32 + "EVALUATION RESULTS SUMMARY")
#     print("=" * 90)
#     header = f"{'#':<3} | {'Question':<42} | {'Context Acc':<11} | {'Answer Acc':<11} | {'Faithfulness':<12}"
#     print(header)
#     print("-" * 90)

#     total_ctx = 0.0
#     total_ans = 0.0
#     total_faith = 0.0

#     for i, res in enumerate(results, 1):
#         q = res["question"]
#         short_q = (q[:39] + "...") if len(q) > 42 else q
#         ctx_s = res["scores"]["context_accuracy"]["score"]
#         ans_s = res["scores"]["answer_accuracy"]["score"]
#         faith_s = res["scores"]["faithfulness"]["score"]

#         total_ctx += ctx_s
#         total_ans += ans_s
#         total_faith += faith_s

#         print(f"{i:<3} | {short_q:<42} | {ctx_s:<11.3f} | {ans_s:<11.3f} | {faith_s:<12.3f}")

#     n = len(results)
#     if n > 0:
#         avg_ctx = total_ctx / n
#         avg_ans = total_ans / n
#         avg_faith = total_faith / n
#         print("=" * 90)
#         print(f"{'OVERALL AVERAGE':<46} | {avg_ctx:<11.3f} | {avg_ans:<11.3f} | {avg_faith:<12.3f}")
#         print("=" * 90 + "\n")


# def save_results(results: List[Dict[str, Any]], output_dir: Path):
#     """Save full results to JSON and CSV files."""
#     output_dir.mkdir(parents=True, exist_ok=True)

#     json_path = output_dir / "evaluation_results.json"
#     with open(json_path, "w", encoding="utf-8") as f:
#         json.dump(results, f, indent=2, ensure_ascii=False)
#     print(f"Detailed JSON results saved to: {json_path}")

#     csv_path = output_dir / "evaluation_results.csv"
#     with open(csv_path, "w", encoding="utf-8", newline="") as f:
#         writer = csv.writer(f)
#         writer.writerow([
#             "index",
#             "question",
#             "context_accuracy_score",
#             "context_accuracy_reasoning",
#             "answer_accuracy_score",
#             "answer_accuracy_reasoning",
#             "faithfulness_score",
#             "faithfulness_reasoning",
#             "generated_answer",
#             "reference_answer",
#             "retrieved_context",
#             "reference_context",
#         ])
#         for i, r in enumerate(results, 1):
#             writer.writerow([
#                 i,
#                 r["question"],
#                 r["scores"]["context_accuracy"]["score"],
#                 r["scores"]["context_accuracy"]["reasoning"],
#                 r["scores"]["answer_accuracy"]["score"],
#                 r["scores"]["answer_accuracy"]["reasoning"],
#                 r["scores"]["faithfulness"]["score"],
#                 r["scores"]["faithfulness"]["reasoning"],
#                 r["generated_answer"],
#                 r["reference_answer"],
#                 r["retrieved_context"],
#                 r["reference_context"],
#             ])
#     print(f"Summary CSV saved to: {csv_path}")


# def run_evaluation(
#     dataset_path: Path,
#     limit: Optional[int] = None,
#     judge_model: str = DEFAULT_JUDGE_MODEL,
#     qwen_model: str = DEFAULT_QWEN_MODEL,
#     output_dir: Optional[Path] = None,
# ):
#     """Run full evaluation pipeline."""
#     groq_api_key = os.environ.get("GROQ_API_KEY")
#     if not groq_api_key or "your_groq_api_key_here" in groq_api_key:
#         raise ValueError(
#             "GROQ_API_KEY is not set. Please set GROQ_API_KEY in your environment or sop_qa_rag/.env file."
#         )

#     from groq import Groq
#     from reteival.retriever import context_retriever

#     groq_client = Groq(api_key=groq_api_key)

#     if not dataset_path.is_file():
#         raise FileNotFoundError(f"Evaluation dataset not found at: {dataset_path}")

#     print(f"Loading evaluation dataset from: {dataset_path}")
#     with open(dataset_path, "r", encoding="utf-8") as f:
#         dataset = json.load(f)

#     if limit and limit > 0:
#         dataset = dataset[:limit]
#         print(f"Evaluating subset of {limit} queries.")
#     else:
#         print(f"Evaluating full dataset ({len(dataset)} queries).")

#     results = []

#     for idx, item in enumerate(dataset, 1):
#         query = item.get("user_input") or item.get("question", "")
#         ref_answer = item.get("reference") or item.get("ground_truth", "")
#         ref_contexts = item.get("reference_contexts") or item.get("contexts", [])
#         ref_context_text = "\n\n".join(ref_contexts) if isinstance(ref_contexts, list) else str(ref_contexts)

#         print(f"\n[{idx}/{len(dataset)}] Query: {query}")

#         # 1. Retrieve context
#         print("  -> Retrieving context from Qdrant...")
#         retrieved_chunks = context_retriever(query, top_k=3)
#         retrieved_context_text = "\n\n---\n\n".join([c["text"] for c in retrieved_chunks])

#         # 2. Generate response from Qwen via Groq
#         print(f"  -> Generating response with Qwen ({qwen_model})...")
#         generated_answer = generate_qwen_response(
#             groq_client=groq_client,
#             query=query,
#             retrieved_context=retrieved_context_text,
#             model=qwen_model,
#         )

#         # 3. Evaluate with Llama 3.3 70B on Groq
#         print(f"  -> Evaluating with LLM Judge ({judge_model})...")
#         scores = evaluate_with_llama_judge(
#             groq_client=groq_client,
#             query=query,
#             reference_answer=ref_answer,
#             reference_context=ref_context_text,
#             retrieved_context=retrieved_context_text,
#             generated_answer=generated_answer,
#             model=judge_model,
#         )

#         print(f"     Context Accuracy : {scores['context_accuracy']['score']:.3f} | {scores['context_accuracy']['reasoning']}")
#         print(f"     Answer Accuracy  : {scores['answer_accuracy']['score']:.3f} | {scores['answer_accuracy']['reasoning']}")
#         print(f"     Faithfulness     : {scores['faithfulness']['score']:.3f} | {scores['faithfulness']['reasoning']}")

#         results.append({
#             "question": query,
#             "reference_answer": ref_answer,
#             "reference_context": ref_context_text,
#             "retrieved_context": retrieved_context_text,
#             "generated_answer": generated_answer,
#             "scores": scores,
#         })

#         # Small delay to adhere to Groq API rate limits
#         time.sleep(1.0)

#     # Display score summary
#     display_results_table(results)

#     # Save outputs
#     out_dir = output_dir or Path(__file__).parent
#     save_results(results, out_dir)


# def main():
#     parser = argparse.ArgumentParser(description="Evaluate RAG with Qwen and an LLM judge.")
#     parser.add_argument(
#         "--dataset",
#         type=Path,
#         default=DEFAULT_DATASET,
#         help="Path to evaluation dataset JSON file.",
#     )
#     parser.add_argument(
#         "--limit",
#         type=int,
#         default=None,
#         help="Limit number of queries to evaluate (e.g. --limit 3 for quick test).",
#     )
#     parser.add_argument(
#         "--judge-model",
#         type=str,
#         default=DEFAULT_JUDGE_MODEL,
#         help=f"Judge model name on Groq (default: {DEFAULT_JUDGE_MODEL}).",
#     )
#     parser.add_argument(
#         "--qwen-model",
#         type=str,
#         default=DEFAULT_QWEN_MODEL,
#         help=f"Qwen model name on Groq (default: {DEFAULT_QWEN_MODEL}).",
#     )
#     parser.add_argument(
#         "--output-dir",
#         type=Path,
#         default=None,
#         help="Directory to save evaluation results CSV and JSON.",
#     )

#     args = parser.parse_args()

#     run_evaluation(
#         dataset_path=args.dataset,
#         limit=args.limit,
#         judge_model=args.judge_model,
#         qwen_model=args.qwen_model,
#         output_dir=args.output_dir,
#     )


# if __name__ == "__main__":
#     main() 

"""
RAG Evaluator using RAGAS framework (Qwen via Groq for Generation)

Evaluates:
  1. Context Precision: Measures whether all ground-truth relevant items in the contexts are ranked high.
  2. Context Recall: Measures how well the retrieved context aligns with the annotated ground truth context.
  3. Faithfulness: Measures whether the LLM generated response is strictly grounded in the retrieved context.
  4. Answer Relevancy: Measures how relevant the generated answer is to the prompt/question.

Workflow:
  - Loads queries, ground truth references, and reference contexts from dataset.
  - Queries Qdrant vector database using context_retriever to obtain top-k retrieved chunks.
  - Generates responses with Qwen via Groq using the retrieved context.
  - Formats results into a Hugging Face Dataset.
  - Evaluates using RAGAS `evaluate` with Langchain's ChatGroq and HuggingFace Embeddings.
  - Outputs formatted score tables, dataset averages, and saves results to CSV and JSON.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Optional

import sys
import types

# Workaround for Ragas importing deprecated vertexai from langchain_community
if "langchain_community.chat_models.vertexai" not in sys.modules:
    dummy_vertex = types.ModuleType("langchain_community.chat_models.vertexai")
    dummy_vertex.ChatVertexAI = type("ChatVertexAI", (object,), {})
    sys.modules["langchain_community.chat_models.vertexai"] = dummy_vertex

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def load_env():
    """Load environment variables from .env file if present."""
    env_paths = [
        PROJECT_ROOT / ".env",
        Path.cwd() / ".env",
        Path(__file__).parent / ".env",
    ]
    for env_path in env_paths:
        if env_path.is_file():
            with open(env_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#") and "=" in line:
                        key, val = line.split("=", 1)
                        key = key.strip()
                        val = val.strip().strip("'\"")
                        if key and key not in os.environ:
                            os.environ[key] = val
            break

load_env()

DEFAULT_DATASET = Path(__file__).parent / "ragas_evaluation_dataset.json"
DEFAULT_JUDGE_MODEL = "openai/gpt-oss-120b"
DEFAULT_QWEN_MODEL = "qwen/qwen3.8-27b" # Updated to a valid Groq Qwen model id if necessary


def generate_qwen_response(
    groq_client: Any,
    query: str,
    retrieved_context: str,
    model: str = DEFAULT_QWEN_MODEL,
) -> str:
    """Generate an answer from Qwen via Groq using retrieved context."""
    prompt = f"""You are an assistant for Standard Operating Procedures (SOP).
Use ONLY the following context to answer the question. If the answer cannot be found in the context, say "I don't know based on the provided documents." Do not invent information.

Context:
{retrieved_context}

Question: {query}
Answer:"""

    response = groq_client.chat.completions.create(
        messages=[
            {"role": "user", "content": prompt},
        ],
        model=model,
    )
    return response.choices[0].message.content.strip()


def run_evaluation(
    dataset_path: Path,
    limit: Optional[int] = None,
    judge_model: str = DEFAULT_JUDGE_MODEL,
    qwen_model: str = DEFAULT_QWEN_MODEL,
    output_dir: Optional[Path] = None,
):
    """Run full evaluation pipeline using RAGAS."""
    groq_api_key = os.environ.get("GROQ_API_KEY")
    if not groq_api_key or "your_groq_api_key_here" in groq_api_key:
        raise ValueError(
            "GROQ_API_KEY is not set. Please set GROQ_API_KEY in your environment or sop_qa_rag/.env file."
        )

    # Imports scoped here so the script fails gracefully if they aren't installed yet
    from groq import Groq
    from datasets import Dataset
    from ragas import evaluate
    from ragas.metrics import (
        context_precision,
        context_recall,
        faithfulness,
        answer_relevancy,
    )
    from langchain_groq import ChatGroq
    from langchain_huggingface import HuggingFaceEmbeddings
    from reteival.retriever import context_retriever

    groq_client = Groq(api_key=groq_api_key)

    # Configure Judge LLM and Embeddings for Ragas
    print(f"Initializing Ragas Judge LLM ({judge_model}) and Embeddings...")
    ragas_llm = ChatGroq(model_name=judge_model, temperature=0.0, groq_api_key=groq_api_key)
    # Ragas answer_relevancy requires an embedding model. We use standard BGE embeddings locally.
    ragas_embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")

    if not dataset_path.is_file():
        raise FileNotFoundError(f"Evaluation dataset not found at: {dataset_path}")

    print(f"Loading evaluation dataset from: {dataset_path}")
    with open(dataset_path, "r", encoding="utf-8") as f:
        dataset_json = json.load(f)

    if limit and limit > 0:
        dataset_json = dataset_json[:limit]
        print(f"Evaluating subset of {limit} queries.")
    else:
        print(f"Evaluating full dataset ({len(dataset_json)} queries).")

    # Data structure required by RAGAS
    data = {
        "question": [],
        "answer": [],
        "contexts": [],
        "ground_truth": []
    }

    # Step 1: Run Generation loop
    for idx, item in enumerate(dataset_json, 1):
        query = item.get("user_input") or item.get("question", "")
        ref_answer = item.get("reference") or item.get("ground_truth", "")
        
        print(f"\n[{idx}/{len(dataset_json)}] Query: {query}")

        print("  -> Retrieving context from Qdrant...")
        retrieved_chunks = context_retriever(query, top_k=3)
        # Ragas expects `contexts` as a list of strings
        context_list = [c["text"] for c in retrieved_chunks]
        retrieved_context_text = "\n\n---\n\n".join(context_list)

        print(f"  -> Generating response with Qwen ({qwen_model})...")
        generated_answer = generate_qwen_response(
            groq_client=groq_client,
            query=query,
            retrieved_context=retrieved_context_text,
            model=qwen_model,
        )

        data["question"].append(query)
        data["answer"].append(generated_answer)
        data["contexts"].append(context_list)
        data["ground_truth"].append(ref_answer)
        
        time.sleep(1.0) # Small delay for Groq generation limits

    # Step 2: Convert to HuggingFace Dataset
    hf_dataset = Dataset.from_dict(data)

    # Step 3: Evaluate using RAGAS
    print("\nStarting RAGAS Evaluation. This may take some time...")
    metrics = [
        context_precision,
        context_recall,
        faithfulness,
        answer_relevancy
    ]
    
    result = evaluate(
        hf_dataset,
        metrics=metrics,
        llm=ragas_llm,
        embeddings=ragas_embeddings,
    )

    # Step 4: Display and Save Results
    print("\n" + "=" * 90)
    print(" " * 32 + "RAGAS EVALUATION RESULTS SUMMARY")
    print("=" * 90)
    print(result) # Prints aggregate dataset scores (mean across dataset)
    print("=" * 90 + "\n")

    # Convert results back to pandas dataframe for export
    df_results = result.to_pandas()
    
    out_dir = output_dir or Path(__file__).parent
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_path = out_dir / "ragas_evaluation_results.csv"
    df_results.to_csv(csv_path, index=False)
    
    json_path = out_dir / "ragas_evaluation_results.json"
    df_results.to_json(json_path, orient="records", indent=2)

    print(f"Detailed CSV results saved to: {csv_path}")
    print(f"Detailed JSON results saved to: {json_path}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate RAG with Qwen and RAGAS framework.")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=DEFAULT_DATASET,
        help="Path to evaluation dataset JSON file.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of queries to evaluate (e.g. --limit 3 for quick test).",
    )
    parser.add_argument(
        "--judge-model",
        type=str,
        default=DEFAULT_JUDGE_MODEL,
        help=f"Judge model name on Groq (default: {DEFAULT_JUDGE_MODEL}).",
    )
    parser.add_argument(
        "--qwen-model",
        type=str,
        default=DEFAULT_QWEN_MODEL,
        help=f"Qwen model name on Groq (default: {DEFAULT_QWEN_MODEL}).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory to save evaluation results CSV and JSON.",
    )

    args = parser.parse_args()

    run_evaluation(
        dataset_path=args.dataset,
        limit=args.limit,
        judge_model=args.judge_model,
        qwen_model=args.qwen_model,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()