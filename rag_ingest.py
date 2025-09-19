import os
import requests
from dotenv import load_dotenv
from bs4 import BeautifulSoup
from pinecone import Pinecone

load_dotenv()


def chunk_text(text, size=500, overlap=50):
    """
    Split a text string into overlapping chunks.

    Args:
        text (str): The input text to chunk.
        size (int): Maximum size of each chunk (default: 500).
        overlap (int): Number of characters to overlap between chunks (default: 50).

    Returns:
        list[str]: A list of text chunks.
    """
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + size, len(text))
        chunk = text[start:end]
        if len(chunk) >= overlap:  # only include if big enough
            chunks.append(chunk)
        start += size - overlap
    return chunks


def scrape_city(city, size=500, overlap=50):
    """
    Scrape Wikivoyage for a given city and return cleaned text chunks.

    Args:
        city (str): The city name (must match Wikivoyage URL format, e.g. "Tokyo").

    Returns:
        list[str]: A list of text chunks extracted from the city page.
    """
    url = f"https://en.wikivoyage.org/wiki/{city}"
    html = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}).text
    soup = BeautifulSoup(html, "html.parser")

    paragraphs = [p.get_text() for p in soup.find_all("p")]
    text = " ".join(paragraphs)
    chunks = chunk_text(text, size=size, overlap=overlap)
    return chunks


def embed_db(city):
    """
    Scrape city information and embed it into Pinecone vector database.

    Args:
        city (str): The city name (e.g., "Tokyo").

    Returns:
        dict: A status message if successful, or an error dictionary on failure.
    """
    PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

    pc = Pinecone(api_key=PINECONE_API_KEY)
    index = pc.Index("travel-knowledge")

    try:
        chunks = scrape_city(city)

        for i, chunk in enumerate(chunks):
            index.upsert_records(
                city,
                [
                    {
                        "_id": f"{city}-{i}",
                        "text": chunk,
                    }
                ],
            )
        return {
            "status": "success",
            "message": f"Successfully embedded {len(chunks)} chunks for {city}",
        }

    except Exception as error:
        return {"error": str(error)}


result = embed_db("Taipei")
print(result)
