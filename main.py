import time
from collections import defaultdict
from dotenv import load_dotenv
from fastapi import FastAPI, Depends, HTTPException, status, Request
from pydantic import BaseModel
from flight_agent import search_flights
from itinerary_agent import query_city
from travel_llm import run_travel_llm
from mangum import Mangum

load_dotenv()

app = FastAPI(title="Travel Assistant API")


request_counts = defaultdict(list)


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


class FlightRequest(BaseModel):
    """Schema for flight search requests via the API."""

    origin: str
    destination: str
    date: str
    adults: int = 1


class ItineraryRequest(BaseModel):
    """Schema for city itinerary questions via the API."""

    city: str
    question: str


class LLMRequest(BaseModel):
    """Schema for general LLM queries via the API."""

    prompt: str


# Endpoints
@app.post("/search-flights", dependencies=[Depends(rate_limit)])
def api_search_flights(req: FlightRequest):
    """Endpoint: search for flights using Amadeus wrapper."""
    return search_flights(
        origin=req.origin, destination=req.destination, date=req.date, adults=req.adults
    )


@app.post("/query-city", dependencies=[Depends(rate_limit)])
def api_query_city(req: ItineraryRequest):
    """Endpoint: answer a city-related question using Pinecone + LLM."""
    return query_city(req.city, req.question)


@app.post("/ask", dependencies=[Depends(rate_limit)])
def api_ask(req: LLMRequest):
    """Endpoint: free-form question answered by the LLM with tools."""
    return run_travel_llm(req.prompt)


handler = Mangum(app)
