from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Literal
from openai import OpenAI
import os
import json
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

class CommentRequest(BaseModel):
    comment: str

class SentimentResponse(BaseModel):
    sentiment: Literal["positive", "negative", "neutral"]
    rating: int

@app.get("/debug")
def debug():
    """CRITICAL: Test if env var works"""
    key = os.getenv("OPENAI_API_KEY")
    return {
        "key_exists": key is not None,
        "key_length": len(key) if key else 0,
        "starts_with": key[:10] + "..." if key else "None",
        "full_request_headers": dict(os.environ)  # Shows ALL env vars
    }

@app.get("/")
def root():
    return {"status": "Sentiment API Live"}

@app.post("/comment", response_model=SentimentResponse)
async def analyze_comment(request: CommentRequest):
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="NO API KEY FOUND")
    
    client = OpenAI(api_key=api_key)
    
    # Rest of your OpenAI code...
    response_schema = {
        "name": "sentiment_response",
        "schema": {
            "type": "object",
            "properties": {
                "sentiment": {"type": "string", "enum": ["positive", "negative", "neutral"]},
                "rating": {"type": "integer", "minimum": 1, "maximum": 5}
            },
            "required": ["sentiment", "rating"],
            "additionalProperties": False
        },
        "strict": True
    }
    
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": f"Analyze: {request.comment}"}],
        response_format={"type": "json_schema", "json_schema": response_schema}
    )
    
    result = json.loads(response.choices[0].message.content)
    return SentimentResponse(**result)
