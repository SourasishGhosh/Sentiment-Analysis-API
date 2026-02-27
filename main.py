from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Literal
from openai import OpenAI
import os
import json
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

class CommentRequest(BaseModel):
    comment: str

class SentimentResponse(BaseModel):
    sentiment: Literal["positive", "negative", "neutral"]
    rating: int

# FIXED: No proxies config + env var check
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY environment variable is required")

client = OpenAI(api_key=api_key)

@app.get("/")
def root():
    return {"status": "Sentiment API - Vercel Ready!"}

@app.post("/comment", response_model=SentimentResponse)
async def analyze_comment(request: CommentRequest):
    try:
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
            messages=[{"role": "user", "content": f"Analyze sentiment: {request.comment}"}],
            response_format={"type": "json_schema", "json_schema": response_schema}
        )
        
        result = json.loads(response.choices[0].message.content)
        return SentimentResponse(**result)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
