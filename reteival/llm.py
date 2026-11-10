# # create an ollama end point to connect to mistral, the query is first sent to llm which should decided if vector_search is necessary, if yes, then convert the query to embedding using same embedding model, call the search_db.py to calculate similarity and retrieve top-n chunks  
# import ollama
# from reteival.retriever import context_retriever 

# model_name = "mistral:latest" 


# def should_perform_retrieval(query: str) -> bool:
#     """Ask Mistral to classify if the query requires SOP documentation search."""
#     router_prompt = f"""You are a query classifier. Decide if the user question requires searching standard operating procedure (SOP) documents or if it's a general greeting/chit-chat.
# Reply ONLY with 'YES' if it requires document lookup, or 'NO' if it does not.
# Question: {query}
# Answer:"""
    
#     response = ollama.generate(model=model_name, prompt=router_prompt)
#     decision = response.get("response", "").strip().upper()
#     return "YES" in decision


# def answer_query(query: str) -> dict:
#     needs_retrieval = should_perform_retrieval(query)
    
#     if needs_retrieval:
#         retrieved_chunks = context_retriever(query, top_k=3)
#         context_text = "\n\n---\n\n".join([c["text"] for c in retrieved_chunks])
        
#         prompt = f"""You are an assistant for Standard Operating Procedures (SOP).
# Use ONLY the following context to answer the question. If the answer cannot be found in the context, say "I don't know based on the provided documents." Do not invent information.
# Context:
# {context_text}
# Question: {query}
# Answer:"""
        
#         response = ollama.generate(model=model_name, prompt=prompt)
#         return {
#             "answer": response["response"],
#             "retrieval_performed": True,
#             "sources": [c["source"] for c in retrieved_chunks]
#         }
#     else:
#         # Direct conversational response from Mistral
#         response = ollama.chat(
#             model= model_name,
#             messages=[{"role": "user", "content": query}]
#         )
#         return {
#             "answer": response["message"]["content"],
#             "retrieval_performed": False,
#             "sources": []
#         }

import os
from groq import Groq
from reteival.retriever import context_retriever 

# Initialize the Groq client (it automatically picks up the GROQ_API_KEY environment variable)
client = Groq()
model_name = "qwen/qwen3.8-27b" # Note: verify the exact Qwen model ID listed in your Groq console (e.g., "qwen-2.5-32b")


def should_perform_retrieval(query: str) -> bool:
    """Ask the LLM to classify if the query requires SOP documentation search."""
    router_prompt = f"""You are a query classifier. Decide if the user question requires searching standard operating procedure (SOP) documents or if it's a general greeting/chit-chat.
Reply ONLY with 'YES' if it requires document lookup, or 'NO' if it does not.
Question: {query}
Answer:"""
    
    response = client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": router_prompt}],
        temperature=0.0, # Best for classification tasks
        max_tokens=10
    )
    decision = response.choices[0].message.content.strip().upper()
    return "YES" in decision


def answer_query(query: str) -> dict:
    needs_retrieval = should_perform_retrieval(query)
    
    if needs_retrieval:
        retrieved_chunks = context_retriever(query, top_k=3)
        context_text = "\n\n---\n\n".join([c["text"] for c in retrieved_chunks])
        
        prompt = f"""You are an assistant for Standard Operating Procedures (SOP).
Use ONLY the following context to answer the question. If the answer cannot be found in the context, say "I don't know based on the provided documents." Do not invent information.
Context:
{context_text}
Question: {query}
Answer:"""
        
        response = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}]
        )
        return {
            "answer": response.choices[0].message.content,
            "retrieval_performed": True,
            "sources": [c["source"] for c in retrieved_chunks]
        }
    else:
        # Direct conversational response from the LLM
        response = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": query}]
        )
        return {
            "answer": response.choices[0].message.content,
            "retrieval_performed": False,
            "sources": []
        }