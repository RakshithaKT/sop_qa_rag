from fastapi import FastAPI  
from pydantic import BaseModel
from reteival.llm import answer_query
# from db import document_registry

app = FastAPI(title="SOP QA RAG API")

class QueryRequest(BaseModel):
    query: str

@app.post("/")
def upload_doc(new_pdf):  
    #check if document was uploaded before using hash, if not, upload
    return {"Document successfully uploaded"} 

@app.post("/llm_endpoint") 
def user_query(request: QueryRequest):  
    result = answer_query(request.query) 
    return result
    #validate user query for data-type, prompt injection and then route it to llm end point 
    