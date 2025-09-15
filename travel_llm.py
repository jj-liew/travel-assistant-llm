import os
from openai import OpenAI
from dotenv import load_dotenv
from flight_agent import search_flights
from itinerary_agent import query_city

load_dotenv()

tools = [
    {
        "type": "function",
        "function": {
            "name": "search_flights",
            "description": "Search for flights using Amadeus API",
            "parameters": {
                "type": "object",
                "properties": {
                    "origin": {
                        "type": "string",
                        "description": "Origin airport/city IATA code",
                    },
                    "destination": {
                        "type": "string",
                        "description": "Destination airport/city IATA code",
                    },
                    "date": {
                        "type": "string",
                        "description": "Departure date in YYYY-MM-DD",
                    },
                    "adults": {
                        "type": "integer",
                        "description": "Number of adult passengers",
                    },
                },
                "required": ["origin", "destination", "date"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "query_city",
            "description": "Query Pinecone to retrieve travel/attraction/food information for a given city",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "The city to retrieve info about",
                    },
                    "question": {
                        "type": "string",
                        "description": "The user's question about the city",
                    },
                },
                "required": ["city", "question"],
            },
        },
    },
]


def run_travel_llm(user_prompt: str):
    """Call the LLM with optional tool-calling for flights and city info.

    The function sends the user's prompt to the model. If the model requests a
    tool call, it executes the corresponding function, then performs a follow-up
    call injecting tool results to produce the final answer.

    Args:
        user_prompt: The user's input prompt for the travel assistant.

    Returns:
        The assistant's textual response string.
    """

    GROQ_API_KEY = os.getenv("GROQ_API_KEY")

    client = OpenAI(base_url="https://api.groq.com/openai/v1", api_key=GROQ_API_KEY)

    response = client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=[{"role": "user", "content": user_prompt}],
        tools=tools,
    )

    tool_calls = response.choices[0].message.tool_calls
    if tool_calls:
        tool_results = []

        for tool_call in tool_calls:
            fn_name = tool_call.function.name
            fn_args = eval(tool_call.function.arguments)

            if fn_name == "search_flights":
                result = search_flights(**fn_args)
            elif fn_name == "query_city":
                result = query_city(**fn_args)
            else:
                result = {"error": f"Unknown tool {fn_name}"}

            tool_results.append(
                {"role": "tool", "tool_call_id": tool_call.id, "content": str(result)}
            )

        followup = client.chat.completions.create(
            model="openai/gpt-oss-120b",
            messages=[
                {"role": "user", "content": user_prompt},
                {"role": "assistant", "tool_calls": tool_calls},
                *tool_results,
            ],
        )
        return followup.choices[0].message.content

    else:
        return response.choices[0].message.content
