import time
from collections import defaultdict
from dotenv import load_dotenv
from fastapi import FastAPI, Depends, HTTPException, status, Request
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel
from travel_llm import run_travel_llm
from mangum import Mangum

load_dotenv()

app = FastAPI(title="Travel Assistant API")


request_counts = defaultdict(list)
system_prompt = """
You are a travel planning assistant. 
You ONLY answer questions related to flights, travel itineraries, or trip planning. 
If the user asks something outside of this domain, politely respond with:
"I'm only able to help with travel itineraries and flights."
"""
format = "Always return the response in simple plaintext, no tables."


def rate_limit(request: Request):
    """
    Simple rate limiter: 20 requests per 5 minutes per IP.

    Args:
        request: FastAPI Request object

    Raises:
        HTTPException: If rate limit exceeded (429 Too Many Requests)
    """
    client_ip = request.client.host
    now = time.time()

    # Remove old requests (older than 5 minutes)
    request_counts[client_ip] = [
        req_time for req_time in request_counts[client_ip] if now - req_time < 300
    ]

    # Check if over limit (10 requests per 5 minutes)
    if len(request_counts[client_ip]) >= 20:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Rate limit: 20 requests per 5 minutes. Please wait.",
        )

    request_counts[client_ip].append(now)


class LLMRequest(BaseModel):
    """Schema for general LLM queries via the API."""

    prompt: str


# Endpoint
@app.post("/ask", response_class=PlainTextResponse, dependencies=[Depends(rate_limit)])
def api_ask(req: LLMRequest):
    """Endpoint: free-form question answered by the LLM with tools."""
    return run_travel_llm(req.prompt + system_prompt + format)


handler = Mangum(app)
