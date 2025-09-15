import os
from openai import OpenAI
from dotenv import load_dotenv
from pinecone import Pinecone

load_dotenv()


def query_city(city: str, question: str):
    """Query Pinecone for city context and ask the LLM to answer a question.

    Args:
        city: The city namespace to search in the vector index.
        question: The user's question about the city.

    Returns:
        Model-generated answer string, or a human-readable error string on failure.
    """
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

    client = OpenAI(base_url="https://api.groq.com/openai/v1", api_key=GROQ_API_KEY)
    pc = Pinecone(api_key=PINECONE_API_KEY)
    index = pc.Index("travel-knowledge")

    try:
        results = index.search(
            namespace=city,
            query={"inputs": {"text": question}, "top_k": 3},
            fields=["text"],
        )

        context = "\n".join(
            [hit["fields"]["text"] for hit in results["result"]["hits"]]
        )

        prompt = f"""
        Answer the question using this travel info:

        {context}

        Question: {question}
        Answer:
        """
        response = client.chat.completions.create(
            model="openai/gpt-oss-120b", messages=[{"role": "user", "content": prompt}]
        )

        return response.choices[0].message.content

    except Exception as e:
        return f"Error while querying {city}: {e}"
