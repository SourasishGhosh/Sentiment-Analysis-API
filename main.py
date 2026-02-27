from pydantic import BaseModel
from typing import Literal
from fastapi import FastAPI, HTTPException
from openai import OpenAI
import os
from dotenv import load_dotenv
from fastapi.middleware.cors import CORSMiddleware 

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For testing
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS", "PUT", "DELETE"],
    allow_headers=["*"],
    expose_headers=["*"]
)

class CommentRequest(BaseModel):
    comment: str

class SentimentResponse(BaseModel):
    sentiment: Literal["positive", "negative", "neutral"]
    rating: int  # Must be 1-5


#load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("API key not found")

client = OpenAI(
    api_key=api_key,
    base_url="https://aipipe.org/openai/v1"
)
#client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

app = FastAPI()

@app.post("/comment", response_model=SentimentResponse)
async def analyze_comment(request: CommentRequest):
    try:
        # The JSON SCHEMA that forces OpenAI to return exact structure
        response_schema = {
            "name": "sentiment_response",  # ← REQUIRED
            "schema": {
                "type": "object",
                "properties": {
                    "sentiment": {
                        "type": "string",
                        "enum": ["positive", "negative", "neutral"]
                    },
                    "rating": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 5
                    }
                },
                "required": ["sentiment", "rating"],
                "additionalProperties": False
            },
            "strict": True  # ← REQUIRED
        }
        
        # Ask OpenAI with STRUCTURED OUTPUTS (the magic part!)
        response = client.chat.completions.create(
            model="gpt-4.1-mini",  # Supports structured outputs
            messages=[
                {
                    "role": "user",
                    "content": f"Analyze this customer comment for sentiment: {request.comment}"
                }
            ],
            response_format={
                "type": "json_schema",
                "json_schema": response_schema
            }
        )
        
        # OpenAI gives us PERFECT JSON - no parsing needed!
        result = eval(response.choices[0].message.content)  # Safe because of schema
        
        return SentimentResponse(**result)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

