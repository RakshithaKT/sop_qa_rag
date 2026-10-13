from fastapi import FastAPI
from sop_qa.api import router

app = FastAPI(
title="SOP Question Answering API",
description="An API for asking questions about standard operating procedures.",
version="0.1.0",
)

@app.get("/health")
def health_check():
    return {"status": "healthy"}

app.include_router(router)